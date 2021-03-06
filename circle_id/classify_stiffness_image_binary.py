# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 11:00:42 2020

@author: Nachiket Gokhale

Program to classify images of stiffness into two categories 
        - has inclusion 
        - does not have inclusion

"""



import tensorflow as tf
import numpy as np
import matplotlib as mpl
import sys,os

from scriptutils import plot_batch, make_stiffness,generate_data,plotall,\
                        feature_scaling_forward,feature_scaling_inverse
from sklearn.metrics import confusion_matrix, accuracy_score

# double backslashes for escape codes
outputdir=r"G:\\Work\\Misc\\BinaryClass\\"
kerasdir='model_binary.keras.save'

nnodex,nnodey=32,128
ntrain,nval,ntest=1024,205,1024
# ntrain,nval,ntest=32,32,32

nepochs   = 2
min_delta = 1E-4
patience  = 10

# get data
train_data,valid_data,test_data = generate_data(nnodex=nnodex,nnodey=nnodey,
                                                ntrain=ntrain,
                                                nval =nval,
                                                ntest=ntest,
                                                create_homo=True)

train_data,scalers = feature_scaling_forward(train_data,None)
valid_data,scalers = feature_scaling_forward(valid_data,scalers)
test_data,scalers  = feature_scaling_forward(test_data,scalers)
    

if (os.path.exists(outputdir+kerasdir)):
    print('Old model exists...loading old model')
    cnn=tf.keras.models.load_model(outputdir+kerasdir)
   
else:
     # create new model
     # Earlystopping
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                           min_delta=min_delta,
                                                           patience=patience)
    
    # Initialising the CNN
    cnn = tf.keras.models.Sequential()
    # Step 1 - Convolution
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',
                                   input_shape=[nnodey, nnodex, 1]))
    # Step 2 - Pooling
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    # Adding a second convolutional layer
    cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    # Step 3 - Flattening
    cnn.add(tf.keras.layers.Flatten())
    # Step 4 - Full Connection
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    # Step 5 - Output Layer
    cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    
    # plot
    tf.keras.utils.plot_model(
            cnn, to_file=outputdir+'model.png', show_shapes=True, show_layer_names=True,
            rankdir='TB', expand_nested=False, dpi=96
        )
    
    
    # Part 3 - Training the CNN
    
    # Compiling the CNN
    cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    
    # Training the CNN on the Training set and evaluating it on the Test set
    # nhg - how does cnn.fit know that data has finished?
    #     - if I do 'for i in training_set' it keeps yielding forever
    history=cnn.fit(x = train_data[0], y = train_data[1],
                    validation_data = (valid_data[0],valid_data[1]),
                    epochs = nepochs, callbacks=[early_stop_callback])
    
    # also check out: cnn.evaluate
    plotall(history,outputdir)

    cnn.save(outputdir+kerasdir)
    
    
# Summary
cnn.summary()


out   = cnn.predict(test_data[0])      # get prediction
out   = out.ravel() > 0.5              # convert to boolean based on probabilty 0.5
out   = out * 1.0                      # convert to integer

# plot_batch(test_data,' test ',pred_label=out)

cm = confusion_matrix(test_data[1],out)
ac = accuracy_score(test_data[1], out)
print('confusion matrix=\n',cm)
print('accuracy_score=',ac)

# inverse transform train_data
train_data = feature_scaling_inverse(train_data,scalers)

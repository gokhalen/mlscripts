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
import matplotlib.pyplot as plt
import sys,os

from scriptutils import plot_batch, make_stiffness,generate_data,plotall,\
                        feature_scaling_forward,feature_scaling_inverse



nnodex,nnodey=32,128
ntrain,nval,ntest=1024,205,1024
nepochs   = 512
min_delta = 1E-4
patience  = 10

# callbacks removed

outputdir=r"G:\\Work\\Production\\circle_id\\Stiffval\\"
kerasdir='model_stiffval.keras.save'


# get data
train_data,valid_data,test_data = generate_data(nnodex=nnodex,nnodey=nnodey,
                                                ntrain=ntrain,
                                                nval  = nval,
                                                ntest = ntest,
                                                create_homo=False)

# Feature Scaling
train_data,scalers = feature_scaling_forward(train_data,None)
valid_data,scalers = feature_scaling_forward(valid_data,scalers)
test_data,scalers  = feature_scaling_forward(test_data,scalers)

if (os.path.exists(outputdir+kerasdir)):
    print('Old model exists...loading old model')
    cnn=tf.keras.models.load_model(outputdir+kerasdir)

else:   
    # Earlystopping
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', 
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
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    # Step 3 - Flattening
    cnn.add(tf.keras.layers.Flatten())
    # Step 4 - Full Connection
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    # Step 5 - Output Layer - maybe make activation relu later
    cnn.add(tf.keras.layers.Dense(units=1))
    
    # plot
    tf.keras.utils.plot_model(
            cnn, to_file=outputdir+'model.png', show_shapes=True, show_layer_names=True,
            rankdir='TB', expand_nested=False, dpi=256
        )
    
    # Part 3 - Training the CNN
    
    # Compiling the CNN
    cnn.compile(optimizer = 'adam', loss ='mse')
    
    # Training the CNN on the Training set and evaluating it on the Test set
    # nhg - how does cnn.fit know that data has finished?
    #     - if I do 'for i in training_set' it keeps yielding forever
    history=cnn.fit(x = train_data[0], y = train_data[3],
                    validation_data = (valid_data[0],valid_data[3]),
                    epochs = nepochs)
    
    # also check out: cnn.evaluate
    plotall(history,outputdir)
    cnn.save(outputdir+kerasdir)

    
# Summary
cnn.summary()

# prediction
out     = cnn.predict(test_data[0])
out     = scalers[3].inverse_transform(out.reshape((-1,1))).reshape((-1,))
correct = scalers[3].inverse_transform(test_data[3].reshape((-1,1))).reshape((-1,))
# inverse transform out and correct stiffness

plt.figure('Error')
xdata  = np.arange(1,ntest+1)
delta  = (correct-out)
plt.plot(xdata,delta,linewidth='1')
plt.grid(True,which='both')
plt.xlabel('Test example number')
plt.ylabel('Error in stiffness value')
plt.title('Error in stiffness value')
plt.savefig(outputdir+'plotabserror.png')

plt.figure('Correct and predicted stiffness')
xdata  = np.arange(1,ntest+1)
plt.plot(xdata,correct,linewidth='1')
plt.plot(xdata,out,linewidth='1')
plt.grid(True,which='both')
plt.xlabel('Test example number')
plt.ylabel('Stiffness')
plt.legend(['Correct stiffness','Predicted stiffness'])
plt.title('Correct and predicted stiffness')
plt.savefig(outputdir+'plotcomparison.png')

plt.figure('Relative error in stiffness')
xdata  = np.arange(1,ntest+1)
plt.plot(xdata,delta/correct,linewidth='1')
plt.grid(True,which='both')
plt.xlabel('Test example number')
plt.ylabel('Relative error in stiffness')
plt.title('Relative error in stiffness')
plt.savefig(outputdir+'plotrelcomparison.png')
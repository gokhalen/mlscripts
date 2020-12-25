# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:13:10 2020

@author: aa
"""

import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys

from scriptutils import plot_batch, make_stiffness,generate_data,plotall,\
                        feature_scaling_forward,feature_scaling_inverse


nnodex,nnodey     = 64,64
ntrain,nval,ntest = 2048,64,64
nepochs   = 512
min_delta = 1e-4
patience  = 10

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


# Earlystopping
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                       min_delta=min_delta,
                                                        patience=patience)


# Initialising the CNN
cnn = tf.keras.models.Sequential()
# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',
                               input_shape=[nnodex, nnodey, 1]))
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
        cnn, to_file='model.png', show_shapes=True, show_layer_names=True,
        rankdir='TB', expand_nested=False, dpi=96
    )

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss ='mse')

# Summary
cnn.summary()

# Training the CNN on the Training set and evaluating it on the Test set
# nhg - how does cnn.fit know that data has finished?
#     - if I do 'for i in training_set' it keeps yielding forever
history=cnn.fit(x = train_data[0], y = train_data[4],
                validation_data = (valid_data[0],valid_data[4]),
                epochs = nepochs,
                callbacks = [early_stop_callback])

# also check out: cnn.evaluate
plotall(history)

out   = cnn.predict(test_data[0]).ravel() # get prediction
out   = scalers[4].inverse_transform(out.reshape(-1,1)).reshape((-1,))

true = scalers[4].inverse_transform(test_data[4].reshape(-1,1)).reshape((-1,))



plt.figure('Relative Error')
plt.plot((true-out)/(true))
plt.title('Relative Error')



'''
# check inverse scaling
train_data1 = feature_scaling_inverse(train_data1,scalers)
valid_data1 = feature_scaling_inverse(valid_data1,scalers)
test_data1  = feature_scaling_inverse(test_data1,scalers)

for ii in range(5):
    print(np.linalg.norm(train_data1[ii] - train_data[ii]))
    print(np.linalg.norm(valid_data1[ii] - valid_data[ii]))
    print(np.linalg.norm(test_data1[ii]  -  test_data[ii]))
'''
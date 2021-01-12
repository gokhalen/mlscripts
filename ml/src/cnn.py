import tensorflow as tf
import os,numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score

from .datastrc import *
from .plotting import plotall

# Custom activation function
# https://stackoverflow.com/questions/43915482/how-do-you-create-a-custom-activation-function-with-keras
# https://keras.io/api/layers/activations/

def define_cnn(mltype,nnodex,nnodey):
    ndim = 2 # number of displacement components.

    # lookup table to define output layer (units and activation)
    # and loss and other metrics to evaluate
    # look up table seems to be cleaner than using if statements
    units      = {'binary':1,
                  'center':2
                  }
    loss       = {'binary':'binary_crossentropy',
                  'center':'mse'
                  }
    activation = {'binary':'sigmoid',
                  'center':'linear'
                  }
    metrics    = {'binary':['accuracy'],
                  'center':[]
                  }

    # Initialising the CNN
    cnn    = tf.keras.models.Sequential()
    # Step 1 - Convolution
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',
                                       input_shape=[nnodey, nnodex, 2]))
    # Step 2 - Pooling
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    # Another convolutional layer
    cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    
    # Step 3 - Flattening 
    cnn.add(tf.keras.layers.Flatten())
    # Step 4 - Full Connection
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    # Step 5 - Output Layer - mltype is a string which comes out of the params dictionary
    cnn.add(tf.keras.layers.Dense(units=units[mltype], activation=activation[mltype]))
    cnn.compile(optimizer = 'adam', loss = loss[mltype], metrics = metrics[mltype])

    # plot
    tf.keras.utils.plot_model(
            cnn, to_file=f'model_{mltype}.png', show_shapes=True, show_layer_names=True,
            rankdir='TB', expand_nested=False, dpi=256
    )
    
    return cnn

def train_cnn(mltype,cnn,train_data,valid_data,epochs):
    # we're using eval and consistent definition of attributes to escape writing lots of if statements
    history=cnn.fit(x = train_data.images, y = eval(f'train_data.labels.{mltype}'),
                    validation_data = (valid_data.images,eval(f'valid_data.labels.{mltype}')),
                    epochs = epochs)
    
    return (cnn,history)

def load_or_train_and_plot_cnn(mltype,train_data,valid_data,nnodex,nnodey,epochs):
    
    # load old model if exists else create new model
    if (os.path.exists(mltype)):
        print('-'*80,f'\n Old model for mltype={mltype} exists...loading old model\n','-'*80,sep='')
        cnn=tf.keras.models.load_model(mltype)
    else:
        cnn         = define_cnn(mltype,nnodex,nnodey)
        cnn,history = train_cnn(mltype=mltype,
                                cnn=cnn,
                                train_data=train_data,
                                valid_data=valid_data,
                                epochs=epochs
                                )
        plotall(mltype,history)
        # https://github.com/tensorflow/tensorflow/issues/44178 - Deprecation
        # warnings are nothing to worry about
        tf.keras.models.save_model(cnn,mltype,overwrite=True,include_optimizer=True)
        
    return cnn


def predict_and_save_cnn(mltype,cnn,test_data):
    out = cnn.predict(test_data.images)
    if ( mltype == 'binary'):
        out = out > 0.5
        out = out.reshape((-1,))

    np.save(mltype+'_prediction',out)
    return out

def post_process_cnn(mltype,ntrain,nvalid,ntest,prediction,test_data):
    binary_out = None ;    center_out = None;    radius_out = None
    value_out  = None ;    field_out  = None; 
    if (mltype == 'binary'):
        conf_matrix = confusion_matrix(y_pred=prediction,y_true=test_data.labels.binary)
        accu_score  = accuracy_score(y_pred=prediction,y_true=test_data.labels.binary)
        binary_out  = BinaryPostData(accu_score=accu_score,conf_matrix=conf_matrix)
        
        print(f'Confusion Matrix = \n {conf_matrix[0][0]} (TN) {conf_matrix[0][1]} (FP) \n {conf_matrix[1][0]} (FN) {conf_matrix[1][1]} (TP) ')
        print(f'Accurary Score   =  {accu_score}')
        
        # find which examples are not matching 
        boolbin = (prediction != test_data.labels.binary)
        idx,    = np.where(boolbin)
        if ( idx.size > 0 ):
            print(f'Examples ',idx,' in test set and ',idx+ntrain+nvalid,' in global set (0-based indexing) are not classified correctly',sep='')

    if (mltype == 'center'):


        
        pass

    if (mltype == 'radius'):
        pass

    if (mltype == 'value'):
        pass

    if (mltype =='field'):
        pass

    out = PostData(binary=binary_out,
                   center=center_out,
                   radius=radius_out,
                   value=value_out,
                   field=field_out
                   )
    return out

import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score
from .datastrc import *

def define_cnn(params):
    ndim = 2 # number of displacement components.

    nnodex  = params['nelemx']+1;
    nnodey  = params['nelemy']+1;
    mltype  = params['mltype'];

    # lookup table to define output layer (units and activation)
    # and loss and other metrics to evaluate
    # look up table seems to be cleaner than using if statements
    units      = {'binary':1}
    loss       = {'binary':'binary_crossentropy'}
    activation = {'binary':'sigmoid'}
    metrics    = {'binary':['accuracy']}

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

def predict_cnn(mltype,cnn,test_data):
    if ( mltype == 'binary'):
        out = cnn.predict(test_data.images)
        out = out > 0.5
        out = out.reshape((-1,))
    return out

def post_process_cnn(mltype,prediction,test_data):
    binary_out = None ;    center_out = None;    radius_out = None
    value_out  = None ;    field_out  = None; 
    if (mltype == 'binary'):
        conf_matrix = confusion_matrix(y_pred=prediction,y_true=test_data.labels.binary)
        accu_score  = accuracy_score(y_pred=prediction,y_true=test_data.labels.binary)
        binary_out  = BinaryPostData(accu_score=accu_score,conf_matrix=conf_matrix)

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

import tensorflow as tf

def define_cnn(params):
    ndim = 2 # number of displacement components.

    nnodex = params['nelemx']+1;
    nnodey = params['nelemy']+1;
    cnn    = tf.keras.models.Sequential()
    
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',
                                       input_shape=[nnodey, nnodex, 2]))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Flatten())
    # Step 4 - Full Connection
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    # Step 5 - Output Layer
    cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
     # plot
    tf.keras.utils.plot_model(
            cnn, to_file='model.png', show_shapes=True, show_layer_names=True,
            rankdir='TB', expand_nested=False, dpi=256
        )
    cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return cnn

def train_cnn(cnn,train_data,valid_data):
    history=cnn.fit(x = train_data.images, y = train_data.labels.binary,
                    validation_data = (valid_data.images,valid_data.labels.binary),
                    epochs = 10)
    
    return (history,cnn)

def predict_cnn(cnn,test_data):
    pass



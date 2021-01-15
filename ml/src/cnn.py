import tensorflow as tf
import os,numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


from sklearn.metrics import confusion_matrix, accuracy_score

from .datastrc import *
from .plotting import plotall_and_save,plotcurves



# Custom activation function
# https://stackoverflow.com/questions/43915482/how-do-you-create-a-custom-activation-function-with-keras
# https://keras.io/api/layers/activations/

def define_cnn(mltype,nnodex,nnodey,optimizer):
    ndim = 2 # number of displacement components.

    # lookup table to define output layer (units and activation)
    # and loss and other metrics to evaluate
    # look up table seems to be cleaner than using if statements
    units      = {'binary':1,
                  'center':2,
                  'radius':1,
                  'value':1
                  }
    loss       = {'binary':'binary_crossentropy',
                  'center':'mse',
                  'radius':'mse',
                  'value':'mse'
                  }
    activation = {'binary':'sigmoid',
                  'center':'sigmoid',
                  'radius':'linear',
                  'value' :'linear'
                  }
    metrics    = {'binary':['accuracy'],
                  'center':[],
                  'radius':[],
                  'value':[]
                  }

    regularizers = {'binary':None,
                    'center':None,
                    'radius':None,
                    'value':tf.keras.regularizers.l2(0.001),
                   }

    
    
    # Initialising the CNN
    cnn    = tf.keras.models.Sequential()
    
    # Step 1 - Convolution
    conv2d_layer_1 = tf.keras.layers.Conv2D(
                                            filters=32, kernel_size=3, activation='relu',
                                            input_shape=[nnodey, nnodex, 2],
                                            kernel_regularizer=regularizers[mltype]
                                            )

    cnn.add(conv2d_layer_1)
    # Step 2 - Pooling
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    # Another convolutional layer
    conv2d_layer_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',
                                            kernel_regularizer=regularizers[mltype]
                                            )
    cnn.add(conv2d_layer_2)
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    
    # Step 3 - Flattening 
    cnn.add(tf.keras.layers.Flatten())
    
    # Step 4 - Full Connection
    dense_layer_1 = tf.keras.layers.Dense(units=128, activation='relu',
                                          kernel_regularizer=regularizers[mltype])
    cnn.add(dense_layer_1)
    
    # Step 5 - Output Layer - mltype is a string which comes out of the params dictionary
    dense_layer_2 = tf.keras.layers.Dense(units=units[mltype], activation=activation[mltype],
                                          kernel_regularizer=regularizers[mltype])
    cnn.add(dense_layer_2)

    opt = tf.keras.optimizers.Adam()

    cnn.compile(optimizer = opt, loss = loss[mltype], metrics = metrics[mltype])
    
    return cnn

def train_cnn(mltype,cnn,train_data,valid_data,epochs):
    # we're using eval and consistent definition of attributes to escape writing lots of if statements
    history=cnn.fit(x = train_data.images, y = eval(f'train_data.labels.{mltype}'),
                    validation_data = (valid_data.images,eval(f'valid_data.labels.{mltype}')),
                    epochs = epochs)


    return (cnn,history)

def load_or_train_and_plot_cnn(mltype,train_data,valid_data,nnodex,nnodey,epochs,optimizer):
    
    # load old model if exists else create new model,train it and save it
    if (os.path.exists(mltype)):
        print('-'*80,f'\n Old model for mltype={mltype} exists...loading old model\n','-'*80,sep='')
        cnn=tf.keras.models.load_model(mltype)
    else:
        cnn         = define_cnn(mltype,nnodex,nnodey,optimizer)
        cnn,history = train_cnn(mltype=mltype,
                                cnn=cnn,
                                train_data=train_data,
                                valid_data=valid_data,
                                epochs=epochs
                                )
        plotall_and_save(mltype,history)
        
        # https://github.com/tensorflow/tensorflow/issues/44178 - Deprecation
        # warnings are nothing to worry about
        tf.keras.models.save_model(model=cnn,filepath=mltype,overwrite=True,include_optimizer=True)


        # plot
    tf.keras.utils.plot_model(
            cnn, to_file=f'{mltype}_model.png', show_shapes=True, show_layer_names=True,
            rankdir='TB', expand_nested=False, dpi=256
    )
        
    return cnn


def predict_cnn(mltype,cnn,test_data):
    out = cnn.predict(test_data.images)
    if ( mltype == 'binary'):
        out = out > 0.5
        out = out.reshape((-1,))

    if ( mltype == 'center'):
        pass

    if ( mltype == 'radius'):
        out = out.reshape((-1,))

    if ( mltype == 'value'):
        out = out.reshape((-1,))

    return out

def save_prediction(mltype,prediction):
    np.save(mltype+'_prediction',prediction)


def percentages(ytrue,ypred,percen,ntest,msg,logfile):
    # percen = iterable containing percentages
    # for every value pp in percen we compute
    # the number of examples whose relative error
    # is less than pp

    # msg - message to be appened
    wrn = f'{__file__}: percentages Check for zero denominator later in rel error calculation...';
    print(wrn)
    
    absrelerr = np.abs((ypred-ytrue)/ytrue)   # not checking for ytrue = 0
    with open(logfile,'a') as fout:
        fout.write(wrn+'\n')
        for pr in percen:
            nn = np.sum((absrelerr <= pr)*1)
            outstr = msg + f' {nn} examples out of {ntest} have abs(rel error) <= {pr} {(nn/ntest)*100}%'
            print(outstr)
            fout.write(outstr+'\n')
        

def post_process_cnn(mltype,ntrain,nvalid,ntest,prediction,test_data):
    binary_out = None ;    center_out = None;    radius_out = None
    value_out  = None ;    field_out  = None;

    logfile = mltype+'_logfile.txt'
    if(os.path.exists(logfile)):
       os.remove(logfile)
       
    percen = [0.05,0.10,0.15,0.2,0.25,0.3,0.35]
    
    if (mltype == 'binary'):

        conf_matrix = confusion_matrix(y_pred=prediction,y_true=test_data.labels.binary)
        accu_score  = accuracy_score(y_pred=prediction,y_true=test_data.labels.binary)
        binary_out  = BinaryPostData(accu_score=accu_score,conf_matrix=conf_matrix)
        
        conf_message = f'Confusion Matrix = \n {conf_matrix[0][0]} (TN) {conf_matrix[0][1]} (FP) \n {conf_matrix[1][0]} (FN) {conf_matrix[1][1]} (TP)\n'
        accu_message = f'Accurary Score   =  {accu_score}\n'
        notcorrect_message = '\n' 
        
        # find which examples are not matching 
        boolbin = (prediction != test_data.labels.binary)
        idx,    = np.where(boolbin)
        if ( idx.size > 0 ):
            notcorrect_message=f'Examples {idx},in test set and {idx+ntrain+nvalid} in global set (0-based indexing) are not classified correctly'

        with open(logfile,'a') as fout:
            fout.write(conf_message)
            fout.write(accu_message)
            fout.write(notcorrect_message)

        print(conf_message)
        print(accu_message)
        print(notcorrect_message)

    if (mltype == 'center'):
        
        xdata  = np.arange(1,ntest+1)
        delta  = prediction - test_data.labels.center
        cutoff = 0.1
        
        # plot error in x coordinate 
        xlabel = 'No. of test example'
        ylabel = 'Error in x coordinate (units)'
        title  = 'Error in x coordinate for test examples'
        fname  = mltype+'_plot_x_error'+'.png'
        
        plotcurves(xdata=xdata,ydata=[delta[:,0]],xlabel=xlabel,ylabel=ylabel,
                   title=title,legend=None,fname=fname)

        # plot error in y coordinate
        ylabel = 'Error in y coordinate (units)'
        title  = 'Error in y coordinate for test examples'
        fname  = mltype+'_plot_y_error'+'.png'

        plotcurves(xdata=xdata,ydata=[delta[:,1]],xlabel=xlabel,ylabel=ylabel,
                   title=title,legend=None,fname=fname)

        print('cnn.py Check for zero denominator later in rel error calculation...')
               
        absrelx = np.abs(delta[:,0]/test_data.labels.center[:,0])
        absrely = np.abs(delta[:,1]/test_data.labels.center[:,1])

        ylabel = 'Absolute Relative Error in x coordinate'
        title  = 'Absolute Relative Error in x coordinate'
        fname  = mltype+'_plot_x_relerror'+'.png'        

        plotcurves(xdata=xdata,ydata=[absrelx],xlabel=xlabel,ylabel=ylabel,
                   title=title,legend=None,fname=fname)

        ylabel = 'Absolute Relative Error in y coordinate'
        title  = 'Absolute Relative Error in y coordinate'
        fname  = mltype+'_plot_y_relerror'+'.png'        

        plotcurves(xdata=xdata,ydata=[absrely],xlabel=xlabel,ylabel=ylabel,
                   title=title,legend=None,fname=fname)

        # Calculate number of examples below a specified % rel error for x coord
        percentages(ytrue=test_data.labels.center[:,0],ypred=prediction[:,0],
                    percen=percen,ntest=ntest,msg='X-coordinate rel error: ',
                    logfile=logfile)
        # Calculate number of examples below a specified % rel error for y coord
        percentages(ytrue=test_data.labels.center[:,1],ypred=prediction[:,1],
                    percen=percen,ntest=ntest,msg='Y-coordinate rel error: ',
                    logfile=logfile)


    if (mltype == 'radius'):
        xdata  = np.arange(1,ntest+1)
        delta  = prediction - test_data.labels.radius
        abserr = np.abs(delta/test_data.labels.radius)

        xlabel = 'No. of test example'
        ylabel = 'Error in radius (units)'
        title  = 'Error in radius for test examples'
        fname  = mltype+'_plot_error'+'.png'

        plotcurves(xdata=xdata,ydata=[delta],xlabel=xlabel,ylabel=ylabel,
                title=title,legend=None,fname=fname)

        absrelerr = np.abs(delta/test_data.labels.radius)
        ylabel    = 'Absolute relative error in radius'
        title     = 'Absolute relative error in radius vs test example number'
        fname     = mltype+'_plot_abs_rel_error.png'

        plotcurves(xdata=xdata,ydata=[abserr],xlabel=xlabel,ylabel=ylabel,
                   title=title,legend=None,fname=fname)

        percentages(ytrue=test_data.labels.radius,ypred=prediction,
                    percen=percen,ntest=ntest,msg='Radius rel error: ',
                    logfile=logfile)
        
    if (mltype == 'value'):
        xdata     = np.arange(1,ntest+1)
        delta     = prediction - test_data.labels.value
        absrelerr = np.abs(delta/test_data.labels.value)

        xlabel = 'No. of test example'
        ylabel = 'Error in shear modulus value (units)'
        title  = 'Error in shear modulus value (units) for test examples'
        fname  = mltype+'_plot_error'+'.png'

        plotcurves(xdata=xdata,ydata=[delta],xlabel=xlabel,ylabel=ylabel,
                   title=title,legend=None,fname=fname)

        xlabel = 'No. of test example'
        ylabel = 'Absolute relative error in shear modulus value (units)'
        title  = 'Absolute relative error in shear modulus value (units) for test examples'
        fname  = mltype+'_plot_rel_error'+'.png'

        plotcurves(xdata=xdata,ydata=[absrelerr],xlabel=xlabel,ylabel=ylabel,
                   title=title,legend=None,fname=fname)

        percentages(ytrue=test_data.labels.value,ypred=prediction,
                    percen=percen,ntest=ntest,msg='Value rel error: ',
                    logfile=logfile)
        

    if (mltype =='field'):
        pass

    out = PostData(binary=binary_out,
                   center=center_out,
                   radius=radius_out,
                   value=value_out,
                   field=field_out
                   )
    return out

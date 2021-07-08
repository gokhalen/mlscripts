import tensorflow as tf
import os,numpy as np
import matplotlib as mpl
# https://stackoverflow.com/questions/45993879/matplot-lib-fatal-io-error-25-inappropriate-ioctl-for-device-on-x-server-loc See nanounanue's answer
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import json
# import pickle

from sklearn.metrics import confusion_matrix, accuracy_score
from .datastrc import *
from .plotting import plotall_and_save,plotcurves,plotfield,subplotfields
from .data import select_input_comps

# Custom activation function
# https://stackoverflow.com/questions/43915482/how-do-you-create-a-custom-activation-function-with-keras
# https://keras.io/api/layers/activations/

def get_checkpoint(mltype,chkdir):
    # https://keras.io/api/callbacks/model_checkpoint/
    if (mltype == 'binary'):
        monitor = 'val_accuracy'
    else:
        monitor = 'val_loss'

    # save the best model
    chkpnt = tf.keras.callbacks.ModelCheckpoint(filepath=chkdir,
                                                save_weights_only=False,
                                                monitor=monitor,
                                                verbose=0,
                                                mode='auto',
                                                save_best_only=True
                                               )

    # save only the weights of the best model
    chkpnt_wts_only =  tf.keras.callbacks.ModelCheckpoint(filepath=chkdir+'_weights_only',
                                                save_weights_only=True,
                                                monitor=monitor,
                                                verbose=0,
                                                mode='auto',
                                                save_best_only=True
                                               )


    return [chkpnt,chkpnt_wts_only]
    

def define_cnn(mltype,iptype,nnodex,nnodey,mubndmin,mubndmax,activation_arg,optimizer):

    def shift_sigmoid_both(x):
        return tf.constant(mubndmax-1.0)*tf.keras.backend.sigmoid(x) + tf.constant(mubndmin)

    def shift_square_both(x):
        return tf.keras.backend.minimum( x*x + tf.constant(mubndmin), tf.constant(mubndmax) )

    def shift_softplus_both(x):
        return tf.keras.backend.minimum( tf.keras.backend.softplus(x) + tf.constant(mubndmin), tf.constant(mubndmax) )

    def shift_softplus_lower(x):
        return tf.keras.backend.softplus(x) + tf.constant(mubndmin)

    def sigmoid_symmetric(x):
        return 2.0*(tf.keras.backend.sigmoid(x)-0.5)

    def twisted_tanh(x):
        return tf.keras.backend.tanh(x) + tf.constant(0.01)*x
    
    # channels is the number of components of input fields
    # if we have 2 components of displacements then nchannel =2

    channel_dict = {'images':2,
                    'imagesx':1,
                    'imagesy':1,
                    'strain':3,
                    'strainxx':1,
                    'strainyy':1,
                    'strainxxyy':2
    }
    

    # lookup table to define output layer (units and activation)
    # and loss and other metrics to evaluate
    # look up table seems to be cleaner than using if statements
    units      = {'binary':1,
                  'center':2,
                  'radius':1,
                  'value' :1,
                  'field' :(nnodex*nnodey)
                  }
    loss       = {'binary':'binary_crossentropy',
                  'center':'mse',
                  'radius':'mse',
                  'value' :'mse',
                  'field' :'mse'
                  }
    
    activation = {'binary':'sigmoid',
                  'center':'sigmoid',
                  'radius':'softplus',
                  'value' :'softplus',
                  'field' :'softplus'
                  }

    # set the activation arg to either string or function
    if (activation_arg not in ['sigmoid','softplus','tanh']):
        activation['field'] = eval(activation_arg)
    else:
        activation['field'] = activation_arg

    metrics    = {'binary':['accuracy'],
                  'center':[],
                  'radius':[],
                  'value':[],
                  'field':[]
                  }

    regularizers = {'binary':None,
                    'center':None,
                    'radius':None,
                    'value':tf.keras.regularizers.l2(0.0),
                    'field':None
                   }
    
    # Initialising the CNN
    cnn    = tf.keras.models.Sequential()
    
    # Step 1 - Convolution
    conv2d_layer_1 = tf.keras.layers.Conv2D(
                                            filters=32, kernel_size=3, activation='relu',
                                            input_shape=[nnodey, nnodex, channel_dict[iptype]],
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


def train_cnn(mltype,iptype,cnn,train_data,valid_data,epochs,callback_list):
    
    # we're using eval and consistent definition of attributes to escape writing lots of if statements
    # we need to reshape the field data differently from other data

    # depending on the iptype we are choosing the data to be 'images' or 'strain'
    data_dict = {'images':'images',
                 'imagesx':'images',
                 'imagesy':'images',
                 'strain':'strain',
                 'strainxx':'strain',
                 'strainyy':'strain',
                 'strainxxyy':'strain'
                }
    
    if ( mltype != 'field'):
        history=cnn.fit( x = eval(f'train_data.{data_dict[iptype]}'),
                         y = eval(f'train_data.labels.{mltype}'),
                         validation_data = (eval(f'valid_data.{data_dict[iptype]}'),eval(f'valid_data.labels.{mltype}')),
                         epochs    = epochs,
                         callbacks = callback_list
                       )

    if ( mltype == 'field'):
        # for training we require a 2D array
        ntrain = train_data.labels.field.shape[0]
        nvalid = valid_data.labels.field.shape[0]
        
        ytrain = train_data.labels.field[:,:,:,1]
        ytrain = ytrain.reshape((ntrain,-1))
        
        yvalid = valid_data.labels.field[:,:,:,1]
        yvalid = yvalid.reshape((nvalid,-1))

        history=cnn.fit( x = eval(f'train_data.{data_dict[iptype]}'),
                         y = ytrain,
                         validation_data = (eval(f'valid_data.{data_dict[iptype]}'),yvalid),
                         epochs    = epochs,
                         callbacks = callback_list
                       )
        
    return (cnn,history)

def load_or_train_and_plot_cnn(mltype,iptype,train_data,valid_data,nnodex,nnodey,mubndmin,mubndmax,epochs,activation,optimizer,mode,outputdir):
    
    model_dir          = outputdir+'/' + mltype+'_'+iptype+'_model'
    check_dir          = outputdir+'/' + mltype+'_'+iptype+'_check_model'
    history_file       = outputdir+'/' + mltype+'_'+iptype+'_model_history.json'
    best_callback_file = outputdir+'/' + mltype+'_'+iptype+'_best_callback.json' 

    callback_list = get_checkpoint(mltype=mltype,chkdir=check_dir)

    # if mode == 'checkpoint' the load checkpointed model
    # if mode == 'train' check to see if previously saved model exists
    #        if so, continue training for 'nepochs'
    #        else,  define a new model from scratch and train it
    
    if ( mode == 'checkpoint'):
        # we can get rid of this if branch and just return the best model after training
        print('-'*80,f'\n Loading checkpointed model for mltype={mltype} iptype={iptype}\n','-'*80,sep='')
        cnn = tf.keras.models.load_model(check_dir)

    if ( mode == 'train'):
        old_history = {}
        if (os.path.exists(model_dir)):
            print('-'*80,f'\n Old model for mltype={mltype} iptype={iptype} exists...loading old model and retraining \n','-'*80,sep='')
            cnn = tf.keras.models.load_model(model_dir)
            # assume history_file exists if model_dir exists and load history dictionary
            with open(history_file,'r') as fin:
                old_history = json.load(fin)

            # https://stackoverflow.com/questions/59255206/tf-keras-how-to-save-modelcheckpoint-object
            # KawingKelvin's answer
            # get best callback monitored value
            with open(best_callback_file,'r') as fin:
                dd = json.load(fin)

            callback_list[0].best = dd['best']
                
        else:
            print('-'*80,f'\n Training model from scratch for mltype={mltype} iptype={iptype} \n','-'*80,sep='')
            cnn = define_cnn(mltype,iptype,nnodex,nnodey,mubndmin,mubndmax,activation,optimizer)
            # check if better weights exist
            if ( os.path.exists(check_dir+'_weights_only.data-00000-of-00001')):
                 print('weights exist...loading weights')
                 cnn.load_weights(check_dir+'_weights_only')

        cnn,new_history = train_cnn(mltype=mltype,
                                iptype=iptype,
                                cnn=cnn,
                                train_data=train_data,
                                valid_data=valid_data,
                                epochs=epochs,
                                callback_list=callback_list
                                )
        

        print(f'{__file__}: Done training')
        
        
        if not old_history:
            # old_history is empty. we are not restarting from a saved model.
            # create the keys in it which are in history
            old_history = { key:[] for key in new_history.history.keys()}

        # append the new_history to the old_history
        history_sum = { key:(old_history[key]+new_history.history[key]) for key in new_history.history.keys()}


        # save the history_sum dict
        with open(history_file,'w') as fout:
            json.dump(history_sum,fout)


        print(f'{__file__}: before plotall_and_save')
        
        plotall_and_save(mltype=mltype,
                         iptype=iptype,
                         activation=activation,
                         history=history_sum,
                         outputdir=outputdir)

        
        print(f'{__file__}: saving model')
        # https://github.com/tensorflow/tensorflow/issues/44178 - Deprecation warnings are nothing to worry about
        tf.keras.models.save_model(model=cnn,filepath=model_dir,overwrite=True,include_optimizer=True)

        with open(best_callback_file,'w') as fout:
            dd = {'best':callback_list[0].best}
            json.dump(dd,fout) 
        
        # finally, load the best model and return it at the end.
        # during training the best model is available 
        cnn = tf.keras.models.load_model(check_dir)


    # plot model
    tf.keras.utils.plot_model(
            cnn, to_file=f'{outputdir}/{mltype}_{iptype}_model.png', show_shapes=True, show_layer_names=True,
            rankdir='TB', expand_nested=False, dpi=256
    )
        
    return cnn


def predict_cnn(mltype,iptype,cnn,test_data,nnodex,nnodey):
    # depending on the iptype we are choosing the data to be 'images' or 'strain'
    data_dict = {'images':'images',
                 'imagesx':'images',
                 'imagesy':'images',
                 'strain':'strain',
                 'strainxx':'strain',
                 'strainyy':'strain',
                 'strainxxyy':'strain'
                }
    
    inputobj = eval(f'test_data.{data_dict[iptype]}')
    ntest    = inputobj.shape[0]
    out      = cnn.predict(inputobj)
                           
    if ( mltype == 'binary'):
        out = out > 0.5
        out = out.reshape((-1,))

    if ( mltype == 'center'):
        pass

    if ( mltype == 'radius'):
        out = out.reshape((-1,))

    if ( mltype == 'value'):
        out = out.reshape((-1,))

    if ( mltype == 'field'):
        out    = out.reshape((-1,nnodey,nnodex))
        print('-'*80)
        print('Test set loss = ')
        result = cnn.evaluate(x=inputobj,y=test_data.labels.field[...,1].reshape((ntest,-1)))
        print('-'*80)

    return out

def save_prediction_test_data(mltype,iptype,noiseid,prediction,test_data,outputdir):
    np.save(outputdir+'/'+mltype+'_'+iptype+'_noise_'+noiseid+'_prediction',prediction)
    np.save(outputdir+'/'+mltype+'_'+iptype+'_noise_'+noiseid+'_correct',eval(f'test_data.labels.{mltype}'))
    np.save(outputdir+'/'+mltype+'_'+iptype+'_noise_'+noiseid+'_test_images',test_data.images)
    np.save(outputdir+'/'+mltype+'_'+iptype+'_noise_'+noiseid+'_test_strain',test_data.strain)


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
        

def post_process_cnn(mltype,iptype,noiseid,ntrain,nvalid,ntest,prediction,test_data,outputdir,nnodex,nnodey,nimg):
    binary_out = None ;    center_out = None;    radius_out = None
    value_out  = None ;    field_out  = None;

    logfile = outputdir+'/'+mltype+'_'+iptype+'_noise_'+noiseid+'_logfile.txt'

    # need to delete logfile if exists, because we're appending to it
    # if ( os.path.exists(logfile) ):
    #   os.remove(logfile)
        
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
        fname  = mltype+'_'+iptype+'_plot_x_error'+'.png'
        
        plotcurves(xdata=xdata,ydata=[delta[:,0]],xlabel=xlabel,ylabel=ylabel,
                   title=title,outputdir=outputdir,legend=None,fname=fname)

        # plot error in y coordinate
        ylabel = 'Error in y coordinate (units)'
        title  = 'Error in y coordinate for test examples'
        fname  = mltype+'_'+iptype+'_plot_y_error'+'.png'

        plotcurves(xdata=xdata,ydata=[delta[:,1]],xlabel=xlabel,ylabel=ylabel,
                   title=title,outputdir=outputdir,legend=None,fname=fname)

        print('cnn.py Check for zero denominator later in rel error calculation...')
               
        absrelx = np.abs(delta[:,0]/test_data.labels.center[:,0])
        absrely = np.abs(delta[:,1]/test_data.labels.center[:,1])

        ylabel = 'Absolute Relative Error in x coordinate'
        title  = 'Absolute Relative Error in x coordinate'
        fname  = mltype+'_'+iptype+'_plot_x_relerror'+'.png'        

        plotcurves(xdata=xdata,ydata=[absrelx],xlabel=xlabel,ylabel=ylabel,
                   title=title,outputdir=outputdir,legend=None,fname=fname)

        ylabel = 'Absolute Relative Error in y coordinate'
        title  = 'Absolute Relative Error in y coordinate'
        fname  = mltype+'_'+iptype+'_plot_y_relerror'+'.png'        

        plotcurves(xdata=xdata,ydata=[absrely],xlabel=xlabel,ylabel=ylabel,
                   title=title,outputdir=outputdir,legend=None,fname=fname)

        ylabel = 'x-coordinate (units)'
        title  = 'True and predicted x-coordinate (units)'
        fname  = mltype+'_'+iptype+'_plot_x_comparison'+'.png'

        plotcurves(xdata=xdata,ydata=[test_data.labels.center[:,0],prediction[:,0]],
                   xlabel=xlabel,ylabel=ylabel,
                   title=title,outputdir=outputdir,legend=['true','predicted'],fname=fname)

        
        ylabel = 'y-coordinate (units)'
        title  = 'True and predicted y-coordinate (units)'
        fname  = mltype+'_'+iptype+'_plot_y_comparison'+'.png'

        plotcurves(xdata=xdata,ydata=[test_data.labels.center[:,1],prediction[:,1]],
                   xlabel=xlabel,ylabel=ylabel,
                   title=title,outputdir=outputdir,legend=['true','predicted'],fname=fname)

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
        fname  = mltype+'_'+iptype+'_plot_error'+'.png'

        plotcurves(xdata=xdata,ydata=[delta],xlabel=xlabel,ylabel=ylabel,
                   title=title,outputdir=outputdir,legend=None,fname=fname)

        absrelerr = np.abs(delta/test_data.labels.radius)
        ylabel    = 'Absolute relative error in radius'
        title     = 'Absolute relative error in radius vs test example number'
        fname     = mltype+'_'+iptype+'_plot_abs_rel_error.png'

        plotcurves(xdata=xdata,ydata=[abserr],xlabel=xlabel,ylabel=ylabel,
                   title=title,outputdir=outputdir,legend=None,fname=fname)

        ylabel    = 'Radius (units)'
        title     = 'Comparison of true and predicted radii (units)'
        fname     = mltype+'_'+iptype+'_plot_comparison.png'

        plotcurves(xdata=xdata,ydata=[test_data.labels.radius,prediction],
                   xlabel=xlabel,ylabel=ylabel,
                   title=title,outputdir=outputdir,legend=['true','prediction'],fname=fname)
        

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
        fname  = mltype+'_'+iptype+'_plot_error'+'.png'

        plotcurves(xdata=xdata,ydata=[delta],xlabel=xlabel,ylabel=ylabel,
                   title=title,outputdir=outputdir,legend=None,fname=fname)


        xlabel = 'No. of test example'
        ylabel = 'Absolute relative error in shear modulus value (units)'
        title  = 'Absolute relative error in shear modulus value (units) for test examples'
        fname  = mltype+'_'+iptype+'_plot_rel_error'+'.png'

        plotcurves(xdata=xdata,ydata=[absrelerr],xlabel=xlabel,ylabel=ylabel,
                   title=title,outputdir=outputdir,legend=None,fname=fname)

        xlabel = 'No. of test example'
        ylabel = 'Shear modulus'
        title  = 'True and predicted shear modulus (units) for test examples'
        fname  = mltype+'_'+iptype+'_plot_mod_comparison'+'.png'

        plotcurves(xdata=xdata,ydata=[prediction,test_data.labels.value],
                   xlabel=xlabel,ylabel=ylabel,
                   title=title,outputdir=outputdir,legend=['prediction','true'],
                   fname=fname)
                

        percentages(ytrue=test_data.labels.value,ypred=prediction,
                    percen=percen,ntest=ntest,msg='Value rel error: ',
                    logfile=logfile)
        

    if (mltype =='field'):

        minpred = np.min(prediction)
        print(f'{__file__}: min shear modulus = ',minpred)

        coord = np.load('coord.npy')    
        xx    = coord[:,0].reshape(nnodex,nnodey).T
        yy    = coord[:,1].reshape(nnodex,nnodey).T

        # figure out how many zeros to pad
        nzfill = len(str(nimg))

        # JSON structure for output
        json_out = {}
 
        # nimg = test_data.labels.field.shape[0]
        os.system(f'rm {outputdir}/mucomp*.png')
        os.system(f'rm {outputdir}/*.mp4')
        for ifield in range(nimg):
            print(f'plotting test example {ifield+1} out of {nimg}')
            mucorr = test_data.labels.field[ifield,:,:,1]
            mupred = prediction[ifield,:,:]
            # plotfield(xx,yy,field,title,fname,outputdir):
            # plotfield(xx,yy,mucorr,'mu corr'+str(ifield),'mucorr'+str(ifield),outputdir=outputdir)
            # plotfield(xx,yy,mupred,'mu pred'+str(ifield),'mupred'+str(ifield),outputdir=outputdir)
            fname = 'mucomp'+str(ifield).zfill(nzfill)
            subplotfields(xx,yy,[mucorr,mupred],[f'mu correct {ifield}',f'mu pred {ifield}'],fname,outputdir=outputdir)

        os.chdir(outputdir)
        os.system(f'ffmpeg.exe -r 4 -i mucomp%0{nzfill}d.png -vcodec mpeg4 -y {mltype}_{iptype}_noise_{noiseid}_movie.mp4')
        os.system(f'ffmpeg.exe -r 4 -i mucomp%0{nzfill}d.png -c:v libx264 -crf 0 {mltype}_{iptype}_noise_{noiseid}_movie_hires.mp4')
        # hi-res: ffmpeg.exe -r 4 -i mucomp%03d.png -c:v libx264 -crf 0 output.mp4
        # source: https://superuser.com/questions/1429256/producing-lossless-video-from-set-of-png-images-using-ffmpeg

        # write the scaled norm of the error
        # Even if the user specifies nimg, scaled_norm_error is calculated over all test images
        scaled_norm_error = np.linalg.norm(test_data.labels.field[:,:,:,1]-prediction)/np.sqrt(prediction.shape[0])
        json_out['scaled_norm_error'] = scaled_norm_error

        os.chdir('..')
        with open(logfile,'w') as fout:
            json.dump(json_out,fout)
            
    # this isn't really necessary, we're not doing anything with the
    # output of this post_process_cnn
    # only binary_out is not None
    out = PostData(binary=binary_out,
                   center=center_out,
                   radius=radius_out,
                   value=value_out,
                   field=field_out
                   )
    return out



def cnn_summary(cnn,mltype,iptype,noiseid,outputdir):
#   https://stackoverflow.com/questions/41665799/keras-model-summary-object-to-string
    summary_filename = mltype+'_'+iptype+'_noise_'+noiseid+'_summary.txt'
    with open(f'{outputdir}/{summary_filename}','w') as fout:
        cnn.summary(print_fn=lambda x:fout.write(x+'\n'))

    cnn.summary()

# ----------------------------------------------------------------------------------------

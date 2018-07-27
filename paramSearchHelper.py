import numpy as np
#from numpy import genfromtxt
#import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import os
import utils
import NNModel as NN

def hyperparamSearch(log_name, model_type, cwd, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, lr_rng, num_hid_layers_rng, beta_rng, k_p_rng, reg_type, size_hid_layers_rng, num_sims, num_epochs, minibatch_size, log_dir, decay, batch_size, batch_norm, parallel=False, cores=1):
    
    try: # if no log folder exists, create it
        os.mkdir(cwd+"/"+log_dir)
    except: # else start logger
        print("Log directory already created")
    utils.set_logger(os.path.join(cwd+"/"+log_dir,log_dir+'_'+model_type+'_'+log_name+'.log'))
    utils.logging.info("START OF HYPERPARAM SEARCH")
    utils.logging.info("Architecture: %s", model_type)
    
    # compute random values within the ranges for each param of length num_sims
    if model_type == "NN":
        if batch_size != 1:
            (num_batches, m, num_features) = X_train.shape
        else:
            X_train = np.expand_dims(X_train,0) # add a 1 dim to the 0 axis
            Y_train = np.expand_dims(Y_train,0)
            (num_batches, m, num_features) = X_train.shape
        
    num_params = 5
    np.random.seed(13) # set seed for rand
    lower_bounds = [lr_rng[0],num_hid_layers_rng[0],size_hid_layers_rng[0],beta_rng[0],k_p_rng[0]]
    upper_bounds = [lr_rng[1],num_hid_layers_rng[1],size_hid_layers_rng[1],beta_rng[1],k_p_rng[1]]
    sample_size = [num_sims, num_params] # num_sims x number of params in search
    samples_params = np.random.uniform(lower_bounds, upper_bounds, sample_size)

    # modifying the initial random parameters
    lr_samples = 10**samples_params[:,0] # log scale
    hl_samples = samples_params[:,1].astype(int) # rounded down to nearest int
    hu_samples = (samples_params[:,2]*num_features).astype(int) # base of 10 neurons used for each level
    beta = samples_params[:,3]
    k_p = samples_params[:,4]
    
    # save the data for the ranges used to the main sim file
    utils.logging.info("lr_rng = "+str(lr_rng)+" hidden layers rng = "+str(num_hid_layers_rng)+" hidden units rng = "+str(size_hid_layers_rng)+" num sims = %d", num_sims)
    
    results = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0] # allocate array
    
    for i in range(len(lr_samples)):
        if model_type == "NN":
            train_err, train_r2, train_rmse, train_mae, dev_err, test_err, test_rmse, test_mae, min_dev, min_test, min_epoch, test_avg, Y_avg = NN.model(cwd, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, lr_samples[i], beta[i], k_p[i], reg_type, num_epochs, hl_samples[i], hu_samples[i], minibatch_size, log_dir, decay, batch_norm, True, False, False, 10, 100, False, 0)
        
        temp_results = np.array([lr_samples[i], hl_samples[i], hu_samples[i], beta[i], k_p[i], num_epochs, train_err, dev_err, test_err, min_epoch, min_test, min_dev])
        utils.logging.info("START OF NEW MODEL")
        utils.logging.info("learning rate = %f, hidden layers = %d, hidden units = %d, beta = %f, keep_prob = %f, epochs = %d, reg_type = %s", lr_samples[i], hl_samples[i], hu_samples[i], beta[i], k_p[i], num_epochs, reg_type) # add other hyperparams
        utils.logging.info("Train Err = %f, Dev Err = %f, Test Err = %f, Min Dev Err = %f, Min Test Err = %f, Min Epoch = %d", train_err, dev_err, test_err, min_dev, min_test, min_epoch)
        results = np.vstack((results,temp_results)) # get all results in a list

    # results contain an array of the parameters and then the resulting errors
    results = results[1:,:] # get rid of placeholder row
    results= results[results[:,-1].argsort()] # sort by the lowest dev error
    utils.logging.info("RESULTS")
    utils.logging.info("learning rate, num hidden layers, hidden layer size, beta, keep prob, epochs, train err, dev err, test err, min epoch, min test, min dev")
    utils.logging.info(str(results))
    
    return results

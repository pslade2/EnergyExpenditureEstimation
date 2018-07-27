import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import os
import utils
import dataHelp as dh

#maybe add batch norm between layers as an option?
def model(cwd, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, learning_rate, beta, k_p, reg_type, num_epochs, num_hid_layers, size_hid_layers, batch_size, log_dir, decay, batch_norm, saving=True, print_loss=True, plotting = True, print_interval=20, plot_interval=100, parallel=False, queue=0):

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 1                                          # to keep consistent results
    lr = tf.placeholder(tf.float32)
    
    if len(X_train.shape) == 3: # already batched
        (num_batches, m, n_x) = X_train.shape
    else: # add num_batches as 1
        X_train = np.expand_dims(X_train,0)
        Y_train = np.expand_dims(Y_train,0)
        (num_batches, m, n_x) = X_train.shape
    
    losses = []                                       # To keep track of the cost
    train_errs = []                                   # To keep track of the cost
    final_test_pred = []
    dev_errs = []                                     # To keep track of the cost
    epochs = [] 				      # keep track of epoch number
    min_dev = 100.0
    min_test = 100.0
    min_epoch = 0
    
    O = 1 # output size for 1 example
    
    ### ARCHITECTURE CHANGES HERE ###

    X = tf.placeholder(tf.float32,[None, n_x])
    Y = tf.placeholder(tf.float32,[None])
    keep_prob = tf.placeholder(tf.float32)
    
    initializer = tf.glorot_uniform_initializer(seed=1)  # same as xavier
    
    if reg_type == "L2":
        reg = tf.contrib.layers.l2_regularizer(beta)
    elif reg_type == "L1":
        reg = tf.contrib.layers.l1_regularizer(beta)
    else:
        reg = None
    
    for i in range(num_hid_layers):
        if i==0:
            out = tf.layers.dense(inputs=X, units=size_hid_layers, activation=tf.nn.relu, kernel_initializer = initializer, kernel_regularizer=reg)
        else:
            out = tf.layers.dense(inputs=out, units=size_hid_layers, activation=tf.nn.relu, kernel_initializer = initializer, kernel_regularizer=reg)
        out = tf.nn.dropout(out, keep_prob)

    out = tf.layers.dense(inputs=out, units=O, activation=None, kernel_initializer = initializer, kernel_regularizer=reg)
        
    #loss = tf.reduce_mean(tf.squared_difference(out, Y)) # L2 loss
    loss = tf.reduce_mean(tf.losses.absolute_difference(Y,tf.squeeze(out))) # L1 loss
    loss = tf.losses.get_total_loss() # this adds together the regularization losses and loss above

    ### END ARCHITECTURE CHANGES ###
    
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss) # Optimizer, change the learning rate here

    init = tf.global_variables_initializer() # When init is run later (session.run(init))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    
    # Start logging file for this particular model
    log_name = "Model_"+str(learning_rate)+"_"+str(num_epochs)+"_"+str(num_hid_layers)+"_"+str(size_hid_layers)
    if print_loss:
        try: # if no log folder exists, create it
            os.mkdir(cwd+"/"+log_dir)
        except:
            print("directory exists")
        utils.set_logger(os.path.join(cwd+"/"+log_dir,log_name+'.log'))
        utils.logging.info("START OF NEW MODEL")
        utils.logging.info("learning rate = %f, hidden layers = %d, hidden units = %d, epochs = %d", learning_rate, num_hid_layers, size_hid_layers, num_epochs) # add other hyperparams
        utils.logging.info("L2 beta = %f, Dropout Keep Prob = %f", beta, k_p)

    with tf.Session() as sess: # starting tf session --> all computation on tf graph in this with struct
        sess.run(init)
        for epoch in range(num_epochs+1):
            if (decay == True) and (epoch == int(num_epochs/2)):
                learning_rate = learning_rate/5
            for batch in range(num_batches):
                x_batch = np.squeeze(X_train[batch,...])
                y_batch = np.squeeze(Y_train[batch,:])
                _, loss_val = sess.run([optimizer, loss], feed_dict={keep_prob: k_p, X: x_batch, Y: y_batch, lr: learning_rate})

            if (epoch % (num_epochs/print_interval) == 0): # print loss and error rates
                train_pred = sess.run(out, feed_dict={X: x_batch, keep_prob: 1})
                train_err = np.mean(abs((np.squeeze(train_pred) - y_batch)/y_batch))
                dev_pred = sess.run(out, feed_dict={X: X_dev, keep_prob: 1})
                dev_err = np.mean(abs((np.squeeze(dev_pred) - Y_dev)/Y_dev))
                    
                if print_loss:
                    utils.logging.info("Epoch %d loss: %f", epoch, loss_val)   
                    utils.logging.info("Train error: %f", train_err)
                    utils.logging.info("Dev error: %f", dev_err)
                losses.append(loss_val)
                train_errs.append(train_err)
                dev_errs.append(dev_err)
                epochs.append(epoch)
                
                # min dev error check and update (save model if best)
                if dev_err < min_dev:
                    min_dev = dev_err
                    min_epoch = epoch
                    test_pred = sess.run(out, feed_dict={X: X_test, keep_prob: 1}) # test prediction values
                    min_test = np.mean(abs((np.squeeze(test_pred) - Y_test)/Y_test)) # absolute error
                    if saving:
                        save_name = cwd+"/"+log_dir+"/"+log_name+".ckpt"
                        save_path = saver.save(sess, save_name)

                if epoch == num_epochs: # last one, print test data
                    test_pred = sess.run(out, feed_dict={X: X_test, keep_prob: 1}) # test prediction values
                    test_mae = np.mean(abs(np.squeeze(test_pred) - Y_test))
                    test_err = np.mean(abs((np.squeeze(test_pred) - Y_test)/Y_test)) # absolute error
                    test_rmse = np.sqrt(mean_squared_error(Y_test, np.squeeze(test_pred)))
                    final_test_pred = np.squeeze(test_pred)

                    train_r2 = r2_score(y_batch, np.squeeze(train_pred))
                    train_rmse = np.sqrt(mean_squared_error(y_batch, np.squeeze(train_pred)))
                    train_mae = np.mean(abs(np.squeeze(train_pred) - y_batch))
                    
                    if print_loss:
                        utils.logging.info("Final train error: %f", train_err)
                        utils.logging.info("Final dev error: %f", dev_err)
                        utils.logging.info("Final test error: %f", test_err)
                        
                        utils.logging.info("Test MAE: %f", test_mae)
                        utils.logging.info("Test RMSE: %f", test_rmse)
                        utils.logging.info("Train R^2: %f", train_r2)
                        
                        utils.logging.info("Min epoch: %d", min_epoch)
                        utils.logging.info("Min dev error: %f", min_dev)
                        utils.logging.info("Min test error: %f", min_test)
                      
            elif plotting and (epoch % (num_epochs/plot_interval) == 0): # save data at higher rate than print
                losses.append(loss_val)
                train_errs.append(train_err)
                dev_errs.append(dev_err)
                epochs.append(epoch)
                #train_pred = sess.run(out, feed_dict={X: X_train, keep_prob: k_p})
                dev_pred = sess.run(out, feed_dict={X: X_dev, keep_prob: 1})
                #train_err = np.mean(abs((train_pred - Y_train)/Y_train)) # absolute error
                dev_err = np.mean(abs((np.squeeze(dev_pred) - Y_dev)/Y_dev))
        
        if plotting:
            # Plot losses during iterations
            plt.plot(np.squeeze(losses))
            plt.ylabel('loss')
            plt.xlabel('iterations every '+str(num_epochs/plot_interval))
            plt.xlim(0, plot_interval)
            plt.ylim(0, losses[1])
            plt.title("Loss for learning rate = " + str(learning_rate))
            plt.show()

            # Plot percent errors during iterations
            plt.plot(np.squeeze(train_errs))
            plt.plot(np.squeeze(dev_errs))
            plt.xlim(0, plot_interval)
            plt.ylim(0, min(train_errs)*10)
            plt.ylabel('Percent error')
            plt.xlabel('iterations every '+str(num_epochs/plot_interval))
            plt.title("Error for learning rate = " + str(learning_rate))
            plt.show()

            # compare the dev vectors visually
            plt.plot(np.squeeze(dev_pred))
            plt.plot(np.squeeze(Y_dev))
            plt.show()
            
        test_avg, Y_avg = dh.avgSubjectCond(np.squeeze(final_test_pred), Y_test)

        results = train_err, train_r2, train_rmse, train_mae, dev_err, test_err, test_rmse, test_mae, min_dev, min_test, min_epoch, test_avg, Y_avg
        
        if parallel:
            queue.put(results)
        else:
            return results


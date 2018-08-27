import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import os
cwd = os.getcwd()

# have (# data examples, inpute_size)
# want (# data examples - num_steps - 1, num_steps, input_size)
# assumes y is 2D
def reshape_sequences(data, labels, sequence_length):
    data_size, features = data.shape
    out_data = np.zeros((data_size-sequence_length-1, sequence_length, features))
    out_y = np.zeros((data_size-sequence_length-1, labels.shape[1]))
    for i in range(out_data.shape[0]):
        out_data[i,:,:] = data[i:sequence_length+i,:]
        out_y[i] = labels[sequence_length+i+1,:]
    return out_data, out_y

# have (# data examples, inpute_size) # takes step and that's how many inputs it moves between the next sample it creates
# want (# data examples - num_steps - 1, num_steps, input_size)
# assumes y is 2D
def reshape_sequences_step(data, labels, sequence_length, step):
    data_size, features = data.shape
    out_samples = int((data_size-sequence_length)/step)
    out_data = np.zeros((out_samples-1, sequence_length, features))
    out_y = np.zeros((out_samples-1, labels.shape[1]))
    for i in range(out_data.shape[0]):
        ind = i*step
        out_data[i,:,:] = data[ind:sequence_length+ind,:]
        out_y[i] = labels[sequence_length+ind,:]
    return out_data, out_y

# reshape (data, features) to stacked (new_data, sequences, features) at steps of
def timeToSamples(data, data_y, sequence_length, features, step=-1):
    if step == -1: # no redundant samples used
        step = sequence_length
        new_length = data.shape[0]//step
        features = data.shape[1]
        new_y = np.ones((new_length,data_y.shape[1]))*data_y[0]
        return np.reshape(data[:new_length*step,:],(new_length,-1,features)), new_y
    # add the feature to get many different samples of similar data

# downsamples time series data assumes data in is the (num_examples x features)
def downsampleData(cwd, data_folder, new_name, sample_rate):
    data_dir = cwd+"/"+data_folder
    new_dir = cwd+"/"+new_name
    goDirectory(cwd,new_name)
    os.chdir(data_dir) # in data dir
    foldernames = os.listdir(data_dir)
    for folder in foldernames:
        data_folder = data_dir+"/"+folder
        os.chdir(data_folder)
        x_orig = genfromtxt('x.csv', delimiter=',')
        y_orig = genfromtxt('y.csv', delimiter=',')
        y_orig = np.reshape(y_orig,[y_orig.shape[0],-1])
        new_length = x_orig.shape[0]//sample_rate
        features = x_orig.shape[1]
        x_down = rebin(x_orig[:new_length*sample_rate,:], [new_length, features])
        y_down = rebin(y_orig[:new_length*sample_rate,:], [new_length, y_orig.shape[1]])
        # save to new_folder_dir
        os.chdir(new_dir)
        goDirectory(new_dir, folder)
        np.savetxt('x.csv', x_down, delimiter=',')
        np.savetxt('y.csv', y_down, delimiter=',')
        print(x_orig.shape,x_down.shape)

def goDirectory(path, name):
    new_dir = path+"/"+name
    os.chdir(path)
    try:
        os.mkdir(name)
        os.chdir(new_dir)
    except:
        os.chdir(new_dir)

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

# Make confusion matrix, center is bool for reshuffle conditions or not
def confusionMultiMat(pred_list, actual_list, center):
    size = len(pred_list[0]) # number of conditions
    mat = np.zeros((size,size))
    for i in range(len(pred_list)): # loop through subjects
        if center:
            order = np.argsort(actual_list[i])
            pred_temp = np.asarray(pred_list[i])
            actual_temp = np.asarray(actual_list[i])
            pred = np.argsort(pred_temp[order])
            actual = np.argsort(actual_temp[order])

        else:
            pred = np.argsort(pred_list[i])
            actual = np.argsort(actual_list[i])

        for j in range(size): # loop through conditions
            mat[pred[j],actual[j]] += 1
    mat = mat / size # normalizing
    return mat

# Make confusion matrix, center is bool for reshuffle conditions or not
def confusionMat(pred_list, actual_list, center, path, filename, saving=True):
    size = len(pred_list[0]) # number of conditions
    mat = np.zeros((size,size))
    for i in range(len(pred_list)): # loop through subjects
        if center:
            order = np.argsort(actual_list[i])
            pred_temp = np.asarray(pred_list[i])
            actual_temp = np.asarray(actual_list[i])
            pred = np.argsort(pred_temp[order])
            actual = np.argsort(actual_temp[order])

        else:
            pred = np.argsort(pred_list[i])
            actual = np.argsort(actual_list[i])

        for j in range(size): # loop through conditions
            mat[pred[j],actual[j]] += 1
    mat = mat / size # normalizing
    #print(np.diag(mat))
    plt.matshow(mat)
    plt.xlabel('Predicted Order')
    plt.ylabel('Conditions Ordered by Metabolic Effort')
    if saving:
        plt.colorbar()
        os.chdir(path)
        plt.savefig(filename+'.png', bbox_inches='tight')
        plt.savefig(filename+'.eps', format='eps', dpi=1200, bbox_inches='tight')
        plt.savefig(filename+'.svg', format='svg', dpi=1200, bbox_inches='tight')


def avgSubjectCond(pred, label):
    size = len(label)
    pred_avg = []
    label_avg = [label[0]]
    cur_label = label[0]
    ind = [0] # store indeces where value of label changes
    for i in range(size):
        if (label[i] != cur_label) or (i == size-1):
            if (i == size-1): # end of values --> only add to pred_avg
                pred_avg.append(np.mean(pred[ind[-1]:])) # last ind:end
            else:
                cur_label = label[i]
                label_avg.append(cur_label)
                ind.append(i) # add change index to list
                pred_avg.append(np.mean(pred[ind[-2]:ind[-1]]))
    return pred_avg, label_avg

def loadData(data_type, cwd, seed, y_ind, train_size, test_size, features, norm, folder_name, sequence_length, signals, avg_num_steps=1):
    if data_type == "subjects":
        x_train, y_train, x_dev, y_dev, x_test, y_test = splitSubjects(cwd,
            folder_name, seed, y_ind, test_size, features, True, norm, sequence_length, signals, avg_num_steps)
    elif data_type == "subjects_time":
        x_train, y_train, x_dev, y_dev, x_test, y_test = splitSubjects(cwd,
            folder_name, seed, y_ind, test_size, features, False, norm, sequence_length, signals)
    elif data_type == "conditions":
        x_train, y_train, x_dev, y_dev, x_test, y_test = splitConditions(cwd,
            folder_name, seed, y_ind, train_size, test_size, features, True, norm)
    elif data_type == "conditions_time":
        x_train, y_train, x_dev, y_dev, x_test, y_test = splitConditions(cwd,
            folder_name, seed, y_ind, train_size, test_size, features, False, norm, sequence_length)
    elif data_type == "data":
        x_train, y_train, x_dev, y_dev, x_test, y_test = splitAvgData(cwd,
            folder_name, seed, y_ind, train_size, test_size, features, False, norm) # "filt_norm_data_30bins"
    if data_type[0:8] == "subjcond":
        x_train, y_train, x_dev, y_dev, x_test, y_test = splitSubjcond(cwd,
            folder_name, seed, y_ind, train_size, test_size, features, True, norm, data_type[8:])
    return x_train, y_train, x_dev, y_dev, x_test, y_test

def groupedAvg(myArray, N=2):
    result = np.cumsum(myArray, 0)[N-1::N]/float(N)
    result[1:] = result[1:] - result[:-1]
    return result

def loadFiles(path, avg_num_steps=1):
    os.chdir(path)
    x_data = genfromtxt('x.csv', delimiter=',')
    y_data = genfromtxt('y.csv', delimiter=',')
    y_data = np.reshape(y_data,(y_data.shape[0],-1)) #(num,) -> (num,1) so vstack doesn't throw error
    if avg_num_steps != 1: # average N steps and chop off remainder
        x_data = groupedAvg(x_data,avg_num_steps)
        y_data = groupedAvg(y_data,avg_num_steps)
    return x_data, y_data

def addData(x_data, y_data, x_temp, y_temp):
    if len(x_data) == 0:
        x_data = x_temp
        y_data = y_temp
    else:
        x_data = np.vstack((x_data,x_temp))
        y_data = np.vstack((y_data,y_temp))
    return x_data, y_data

# if input data is in (num_examples, features*bins) this returns (num_examples, bins, features)
def chunkData(x_train, x_dev, x_test, features):
    sequence_length = int(data_train.shape[1]/features)
    X_train = np.swapaxes(np.reshape(x_train,(x_train.shape[0],features,sequence_length)),1,2)
    X_dev = np.swapaxes(np.reshape(x_dev,(x_dev.shape[0],features,sequence_length)),1,2)
    X_test = np.swapaxes(np.reshape(x_test,(x_test.shape[0],features,sequence_length)),1,2)
    return X_train, X_dev, X_test

def batch(data, labels, batch_size):
    data_length, sequence_length, features = data.shape
    new_data_length = data_length - (data_length % batch_size)
    num_batches = int(data_length/batch_size)
    data_out = np.reshape(data[:new_data_length,:,:],(num_batches, -1, sequence_length, features))
    labels_out = np.reshape(labels[:new_data_length],(num_batches, -1)) # would need to add dim for num_outputs
    return data_out, labels_out

def batch_nonseq(data, labels, batch_size):
    data_length, features = data.shape
    new_data_length = data_length - (data_length % batch_size)
    num_batches = int(data_length/batch_size)
    data_out = np.reshape(data[:new_data_length,:],(num_batches, -1, features))
    labels_out = np.reshape(labels[:new_data_length],(num_batches, -1)) # would need to add dim for num_outputs
    return data_out, labels_out

def normalizeData(x_train, x_dev, x_test, cycle=True, num_bins=30):
    if cycle:
        features = x_train.shape[1]//num_bins # assumes all features are time series
        mu = np.zeros((1,features))
        cov = np.zeros((1,features))
        for i in range(features):
            mu[0,i]=x_train[:,i*num_bins:(i+1)*num_bins].mean()
            cov[0,i]=x_train[:,i*num_bins:(i+1)*num_bins].std()
        mu = np.repeat(mu,num_bins)
        mu = mu.reshape((1,-1))
        cov = np.repeat(cov,num_bins)
        cov = cov.reshape((1,-1))
    else:
        mu = np.mean(x_train,axis=0) # compute the mean along axis = 0 (num_samples for raw data)
        cov = np.std(x_train,axis=0) # using std instead of variance seems to be best
    #print(mu.shape, cov.shape, features, x_train.shape, cycle, num_bins)
    X_train = (x_train - mu)/cov
    X_dev = (x_dev - mu)/cov
    X_test = (x_test - mu)/cov

    return X_train, X_dev, X_test


# In[56]:
# train/test_split should be a percentage
def splitAvgData(cwd, data_folder, seed, y_ind, train_split, test_split, features, cycle=True, norm=True):
    path = cwd + '/' + data_folder
    os.chdir(path) # cd to given data path
    folder = os.listdir(path) # get folders in path

    folder_path = path +'/'+ folder[0]
    x_data, y_data = loadFiles(folder_path)


    output_samples = x_data.shape[0]
    split_1 = int(train_split * output_samples)
    split_2 = int((train_split+test_split) * output_samples)
    np.random.seed(seed)
    np.random.shuffle(x_data) # randomize the samples
    np.random.seed(seed)
    np.random.shuffle(y_data) # randomize the samples

    x_train = x_data[:split_1, :]
    y_train = y_data[:split_1, y_ind]

    x_dev = x_data[split_1:split_2, :]
    y_dev = y_data[split_1:split_2, y_ind]

    x_test = x_data[split_2:, :]
    y_test = y_data[split_2:, y_ind]

    if norm:
        num_bins = int(x_train.shape[1]/features)
        X_train, X_dev, X_test = normalizeData(x_train, x_dev, x_test, cycle, num_bins)
    else:
        X_train = x_train
        X_dev = x_dev
        X_test = x_test

    return X_train, y_train, X_dev, y_dev, X_test, y_test

# train/test_split should be a percentage
def splitData(cwd, data_folder, seed, y_ind, train_split, test_split, features, cycle=True, norm=True):
    path = cwd + '/' + data_folder
    os.chdir(path) # cd to given data path
    folder = os.listdir(path) # get folders in path

    folder_path = path +'/'+ folder[0]
    x_data, y_data = loadFiles(folder_path)

    output_samples = x_data.shape[0]
    split_1 = int(train_split * output_samples)
    split_2 = int((train_split+test_split) * output_samples)
    np.random.seed(seed)
    np.random.shuffle(x_data) # randomize the samples
    np.random.seed(seed)
    np.random.shuffle(y_data) # randomize the samples

    x_train = x_data[:split_1, :]
    y_train = y_data[:split_1, y_ind]

    x_dev = x_data[split_1:split_2, :]
    y_dev = y_data[split_1:split_2, y_ind]

    x_test = x_data[split_2:, :]
    y_test = y_data[split_2:, y_ind]

    if norm:
        num_bins = int(x_train.shape[1]/features)
        X_train, X_dev, X_test = normalizeData(x_train, x_dev, x_test, cycle, num_bins)
    else:
        X_train = x_train
        X_dev = x_dev
        X_test = x_test

    return X_train, y_train, X_dev, y_dev, X_test, y_test


# In[49]:


## TODO: need to add function by add data that breaks it up into x, time length chunks
# here train and subject splits are int
# test_ind is the indeces to put in test for specific subjects
def splitSubjects(cwd, data_folder, seed, y_ind, test_size, features=20, cycle=True, norm=True, sequence_length=30, signals="all", avg_num_steps=1):
    #print(signals)
    path = cwd + '/' + data_folder
    os.chdir(path) # cd to given data path
    folders = os.listdir(path) # get folders in path
    folders.sort()
    if cycle:
        subj_list = list(range(len(folders))) # list 0 : n-1 for randomizing subj
    else:
        subj_list = []
        for folder in folders:
            cur_subj = folder[1:3]
            if cur_subj not in subj_list:
                subj_list.append(cur_subj)

    np.random.seed(seed)
    np.random.shuffle(subj_list)
    split_1 = len(subj_list) - test_size
    #print(split_1)
    #split_2 = split_1 + test_size
    train_subj = subj_list[:split_1]
    dev_subj = subj_list[split_1:]
    test_subj = subj_list[split_1:]

    print('Test subjects: ', test_subj)
    print('Dev subjects: ', dev_subj)
    print('Train subjects: ', train_subj)
    x_train = []
    y_train = []
    x_dev = []
    y_dev = []
    x_test = []
    y_test = []

    for i,folder in enumerate(folders):
        folder_path = path +'/'+ folder
        x_temp, y_temp = loadFiles(folder_path, avg_num_steps)
        if i==0:
            num_bins = int(x_temp.shape[1]/features)
        #print(x_temp.shape)

        if signals=="EMG" or signals=="emg":
            x_temp = x_temp[:,6*num_bins:]
            features = int(x_temp.shape[1]/num_bins)
        elif signals=="force" or signals=="forces":
            x_temp = x_temp[:,0:6*num_bins]
            features = int(x_temp.shape[1]/num_bins)
        #elif signals=="allvert": # add something here for vertical forces
        #    x_temp = x_temp[:,

        #print(x_temp.shape)
        #print(features)
        if not cycle: # chunk by time sequence length
            if folder[1:3] in train_subj:
                #x_temp, y_temp = reshape_sequences_step(x_temp, y_temp, sequence_length, sequence_length)
                x_temp, y_temp = timeToSamples(x_temp, y_temp, sequence_length, features, step=-1)
                x_train, y_train = addData(x_train, y_train, x_temp, y_temp)
            if folder[1:3] in dev_subj:
                x_temp, y_temp = timeToSamples(x_temp, y_temp, sequence_length, features, step=-1)
                x_dev, y_dev = addData(x_dev, y_dev, x_temp, y_temp)
                x_test, y_test = addData(x_test, y_test, x_temp, y_temp)
        else:
            if i in train_subj:
                x_train, y_train = addData(x_train, y_train, x_temp, y_temp)
            if i in dev_subj:
                x_dev, y_dev = addData(x_dev, y_dev, x_temp, y_temp)
            if i in test_subj: # subject(s) for test folder
                x_test, y_test = addData(x_test, y_test, x_temp, y_temp)

    if norm:
        X_train, X_dev, X_test = normalizeData(x_train, x_dev, x_test, cycle, num_bins)
    else:
        X_train = x_train
        X_dev = x_dev
        X_test = x_test

    return X_train, y_train[:,y_ind], X_dev, y_dev[:,y_ind], X_test, y_test[:,y_ind]

def splitSubjcond(cwd, data_folder, seed, y_ind, train_size,
                    test_size, features=20, cycle=True, norm=True, test_conds_str="1",sequence_length=30):
    path = cwd + '/' + data_folder
    os.chdir(path) # cd to given data path
    folders = os.listdir(path) # get folders in path
    folders.sort()
    #print(folders)
    cond_list = list(range(len(folders))) # list 0 : n-1 for randomizing subj
    num_conditions = len(cond_list)

    ### old implementation below
    #test_conds = list(map(int, str.split(test_conds_str)))
    #if len(test_conds) == 0:
    #    test_conds = [1]

    # hard coding the number of subjects and conditions to make it easier to hold some out
    if data_folder[0:12] == "incline-load":
        subjs = 13
        conds = 12
    elif data_folder[0:8] == "assisted":
        subjs = 8
        conds = 9

    if cycle:
        subj_list = list(range(subjs)) # list 0 : n-1 for randomizing subj
    else:
        print("error because cycle is not true?")

    np.random.seed(seed)
    conds_holdout = int(test_conds_str) # convert num of test conds to int
    test_conds = []#np.random.randint(1,conds+1, conds_holdout)

    new_cond = np.random.randint(0,conds)
    for i in range(conds_holdout):
        while new_cond in test_conds:
            new_cond = np.random.randint(0,conds)
        test_conds.append(new_cond)

    np.random.seed(seed)
    np.random.shuffle(subj_list)
    split_1 = len(subj_list) - test_size
    #print(len(subj_list), test_size)
    #print(split_1)
    #split_2 = split_1 + test_size
    train_subj = subj_list[:split_1]
    dev_subj = subj_list[split_1:]
    test_subj = subj_list[split_1:]

    print('Test subjects: ', test_subj)
    print('Test conditions (held out from train): ', test_conds)
    print('Dev subjects: ', dev_subj)
    print('Train subjects: ', train_subj)

    x_train = []
    y_train = []
    x_dev = []
    y_dev = []
    x_test = []
    y_test = []

    # loop through folders, check which data file to add to
    #for i,folder in enumerate(folders):
    for subj in range(subjs):
        for cond in range(conds):
            cond_ind = subj*conds + cond
            #print(subj, cond, cond_ind, len(folders))
            folder_path = path +'/'+ folders[cond_ind]
            x_temp, y_temp = loadFiles(folder_path)

            # check if condition is in a (train subject && !test_condition) --> train ...  or if test_condition -->test
            if (subj in train_subj) and (cond not in test_conds):
                #print(subj, cond, "train subj/cond")
                x_train, y_train = addData(x_train, y_train, x_temp, y_temp)
            elif (subj in test_subj) and (cond in test_conds):
                #print(subj, cond, "test subj/cond")
                x_dev, y_dev = addData(x_dev, y_dev, x_temp, y_temp)
                x_test, y_test = addData(x_test, y_test, x_temp, y_temp)
    if norm:
        num_bins = int(x_train.shape[1]/features)
        X_train, X_dev, X_test = normalizeData(x_train, x_dev, x_test, cycle, num_bins)
    else:
        X_train = x_train
        X_dev = x_dev
        X_test = x_test
    return X_train, y_train[:,y_ind], X_dev, y_dev[:,y_ind], X_test, y_test[:,y_ind]



def splitConditions(cwd, data_folder, seed, y_ind, train_size,
                    test_size, features=20, cycle=True, norm=True, sequence_length=30):
    path = cwd + '/' + data_folder
    os.chdir(path) # cd to given data path
    folders = os.listdir(path) # get folders in path
    folders.sort()
    cond_list = list(range(len(folders))) # list 0 : n-1 for randomizing subj
    num_conditions = len(cond_list)
    split_1 = int(np.ceil(train_size * num_conditions))
    if (split_1)%2 != 0: # if there is an odd number left
        split_1 = split_1+1
    split_2 = int(split_1 + (num_conditions-split_1)/2)

    np.random.seed(seed)
    np.random.shuffle(cond_list)

    if sequence_length != 30:
        train_conds = cond_list[:split_2]
        dev_conds = cond_list[split_2:]
        test_conds = cond_list[split_2:]
    else:
        train_conds = cond_list[:split_1]
        dev_conds = cond_list[split_1:split_2]
        test_conds = cond_list[split_2:]

    print('Test conditions: ', test_conds)
    x_train = []
    y_train = []
    x_dev = []
    y_dev = []
    x_test = []
    y_test = []

    # loop through folders, check which data file to add to
    for i,folder in enumerate(folders):
        folder_path = path +'/'+ folder
        x_temp, y_temp = loadFiles(folder_path)
        if not cycle: # chunk by time sequence length
            if i in train_conds:
                #x_temp, y_temp = reshape_sequences_step(x_temp, y_temp, sequence_length, sequence_length//5)
                x_temp, y_temp = timeToSamples(x_temp, y_temp, sequence_length, features, step=-1)
                x_train, y_train = addData(x_train, y_train, x_temp, y_temp)
            elif i in dev_conds:
                x_temp, y_temp = timeToSamples(x_temp, y_temp, sequence_length, features, step=-1)
                x_dev, y_dev = addData(x_dev, y_dev, x_temp, y_temp)
            else: # conditions for test folder
                x_temp, y_temp = timeToSamples(x_temp, y_temp, sequence_length, features, step=-1)
                x_test, y_test = addData(x_test, y_test, x_temp, y_temp)
        else: # cycle data
            if i in train_conds:
                x_train, y_train = addData(x_train, y_train, x_temp, y_temp)
            elif i in dev_conds:
                x_dev, y_dev = addData(x_dev, y_dev, x_temp, y_temp)
            else: # conditions for test folder
                x_test, y_test = addData(x_test, y_test, x_temp, y_temp)
    if norm:
        num_bins = int(x_train.shape[1]/features)
        X_train, X_dev, X_test = normalizeData(x_train, x_dev, x_test, cycle, num_bins)
    else:
        X_train = x_train
        X_dev = x_dev
        X_test = x_test
    return X_train, y_train[:,y_ind], X_dev, y_dev[:,y_ind], X_test, y_test[:,y_ind]

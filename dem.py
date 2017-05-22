import pickle
#from hep_ml.reweight import BinsReweighter
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

def get_weight(energy1_, energy2_):
    reweighter = BinsReweighter()
    reweighter.fit(energy1_, energy2_) #energy
    X1_weights = reweighter.predict_weights(energy1_)
    sampler = np.random.choice(np.arange(len(energy1_)),
                               p = X1_weights/X1_weights.sum(),
                               replace=False,
                               size=len(energy2_))

    energy2_ = numpy.array(energy2_)
    energy1_ = numpy.array(energy1_)
    X1_w = energy1_[sampler]

    weight2 = np.ones(len(energy2_))
    print type(weight2), type(X1_weights)
    weights_ = np.concatenate((X1_weights, weight2), axis = 0)
    plt.hist(energy2, bins = 50,alpha=0.5)
    plt.hist(X1_w, bins = 50,alpha=0.5)
    plt.show()
    return weights_

def rev_(X_train):
    X_train_rev = list()
    for elem in X_train:
        X_train_rev.append(elem.ravel())
    X_train_rev = np.array(X_train_rev)
    return X_train_rev

def get_fpr(tpr_val, tpr_, fpr_):
    new_trp = [abs(tpr_-0.9)]
    index = np.argmin(new_trp)
    return fpr_[index]

def write_data(str1, str2):
    with open(str1) as f_in:
        X1, hypo1, y1, energy1 = pickle.load(f_in)

    with open(str2) as f_in:
        X2, hypo2, y2, energy2 = pickle.load(f_in)
    return np.array(X1), np.array(hypo1), np.array(y1), np.array(energy1), np.array(X2), np.array(hypo2), np.array(y2), np.array(energy2)

def preprocess(str1, str2):
    X1, hypo1, y1, energy1, X2, hypo2, y2, energy2 =  write_data(str1, str2)
    X_all = np.concatenate((X1, X2), axis=0)
    print "X_all shape", X_all.shape
    Y_all = np.concatenate((y1, y2), axis=0)
    #weights = get_weight(energy1, energy2)
    weights = np.ones(len(X_all))
    X_train,X_val,y_train,y_val, w_train, w_test = train_test_split(X_all,Y_all,weights)
    return X_train,X_val,y_train,y_val, w_train, w_test
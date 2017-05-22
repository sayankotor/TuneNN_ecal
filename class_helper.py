import time
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import theano
theano.config.floatX = 'float32'
import theano.tensor as T


from sklearn.base import BaseEstimator
import custom_nn

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
        

def single_train(train_fun1, predict_fun, X_train,y_train,  X_val,y_val, n_epoch):       
    test_curve = list()
    train_curve = list()
    pr_ = list() 
    tpr_=list() 
    thresholds = list()
    initial = 0

    for epoch in range(n_epoch):
        for Xb,yb in iterate_minibatches(X_train,y_train,100,shuffle=True):
            _ = train_fun1(Xb, yb)
        y_proba_1 = predict_fun(X_train)
        y_proba_2 = predict_fun(X_val)
        roc_auc_test = roc_auc_score(y_val, y_proba_2[:,1])
        test_curve.append(roc_auc_test)
        roc_auc_train = roc_auc_score(y_train, y_proba_1[:,1])
        train_curve.append(roc_auc_train)
        if (initial == 0 or initial == 99 or initial == 299):
            print "test", roc_auc_test
            print "train", roc_auc_train 
        initial = initial + 1
        fpr_, tpr_, thresholds_ = roc_curve(y_val, y_proba_2[:,1])
    return train_curve, test_curve


class neuronka(BaseEstimator):
    functions = ()
    def __init__(self, ecal_units, ps_units, drop_, drop_ps, n_epoch):
        self.ecal_units = ecal_units
        self.ps_units = ps_units
        self.drop_ = drop_
        self.drop_ps = drop_ps
        self.n_epoch = n_epoch
        self.train_fun,self.accuracy_fun,self.predict_fun = custom_nn.make_nn2(-2, "All", self.ecal_units, self.ps_units, self.drop_, self.drop_ps)
    def fit(self,X, y):
        for epoch in range(self.n_epoch):
            for Xb,yb in iterate_minibatches(X,y,100,shuffle=True):
                _ = self.train_fun(Xb, yb)
                
    def predict_proba(self,X_test):
        return self.predict_fun(X_test)
    
    def predict(self,X_test):
        return self.predict_proba(X_test).argmax(-1)
    
class neuronka1(BaseEstimator):
    functions = ()
    def __init__(self, ecal_units, ps_units, drop_, drop_ps, n_epoch):
        self.ecal_units = ecal_units
        self.ps_units = ps_units
        self.drop_ = drop_
        self.drop_ps = drop_ps
        self.n_epoch = n_epoch
        self.train_fun,self.accuracy_fun,self.predict_fun = custom_nn.make_nn1(-2, "All", self.ecal_units, self.ps_units, self.drop_, self.drop_ps)
    def fit(self,X, y):
        for epoch in range(self.n_epoch):
            for Xb,yb in iterate_minibatches(X,y,100,shuffle=True):
                _ = self.train_fun(Xb, yb)
                
    def predict_proba(self,X_test):
        return self.predict_fun(X_test)
    
    def predict(self,X_test):
        return self.predict_proba(X_test).argmax(-1)

class MyOtherNeuronka(neuronka):
    functions = ()
    def __init__(self, ecal_units, ps_units, drop_, drop_ps, n_epoch):
        self.ecal_units = ecal_units
        self.ps_units = ps_units
        self.drop_ = drop_
        self.drop_ps = drop_ps
        self.n_epoch = n_epoch
        
        
    def fit(self,X, y):
        self.train_fun,self.accuracy_fun,self.predict_fun = custom_nn.make_nn2(-2, "All", self.ecal_units, self.ps_units, self.drop_, self.drop_ps)
        for epoch in range(self.n_epoch):
            for Xb,yb in iterate_minibatches(X,y,100,shuffle=True):
                _ = self.train_fun(Xb, yb)
                
class MyOtherNeuronka1(neuronka):
    functions = ()
    def __init__(self, ecal_units, ps_units, drop_, drop_ps, n_epoch):
        self.ecal_units = ecal_units
        self.ps_units = ps_units
        self.drop_ = drop_
        self.drop_ps = drop_ps
        self.n_epoch = n_epoch
        
        
    def fit(self,X, y):
        self.train_fun,self.accuracy_fun,self.predict_fun = custom_nn.make_nn1(-2, "All", self.ecal_units, self.ps_units, self.drop_, self.drop_ps)
        for epoch in range(self.n_epoch):
            for Xb,yb in iterate_minibatches(X,y,100,shuffle=True):
                _ = self.train_fun(Xb, yb)
                
class MyOtherNeuronka3(neuronka):
    functions = ()
    def __init__(self, ecal_units, ps_units, drop_, drop_ps, n_epoch):
        self.ecal_units = ecal_units
        self.ps_units = ps_units
        self.drop_ = drop_
        self.drop_ps = drop_ps
        self.n_epoch = n_epoch
        
        
    def fit(self,X, y):
        self.train_fun,self.accuracy_fun,self.predict_fun = custom_nn.make_nn3(-2, "All", self.ecal_units, self.ps_units, self.drop_, self.drop_ps)
        for epoch in range(self.n_epoch):
            for Xb,yb in iterate_minibatches(X,y,100,shuffle=True):
                _ = self.train_fun(Xb, yb)
 

class MyOtherNeuronka4(neuronka):
    functions = ()
    def __init__(self, ecal_units, ps_units, drop_, drop_ps, n_epoch):
        self.ecal_units = ecal_units
        self.ps_units = ps_units
        self.drop_ = drop_
        self.drop_ps = drop_ps
        self.n_epoch = n_epoch
        
        
    def fit(self,X, y):
        self.train_fun,self.accuracy_fun,self.predict_fun = custom_nn.make_nn4(-2, "All", self.ecal_units, self.ps_units, self.drop_, self.drop_ps)
        for epoch in range(self.n_epoch):
            for Xb,yb in iterate_minibatches(X,y,100,shuffle=True):
                _ = self.train_fun(Xb, yb)
                
import time
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import theano
theano.config.floatX = 'float32'
import theano.tensor as T

import theano
import theano.tensor as T
import lasagne

from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params

def make_nn2(seed_, part='All', ecal_units=800, ps_units = 50, drop_ = 0, drop_ps = 0, l1_reg = 0, l2_reg=0):
    print "nn2"
    if (seed_ > -1):
        np.random.seed(seed_)
    theano.config.floatX = 'float32'
    input_X = T.tensor4("X")

    input_shape = [None,5,5]

    target_y = T.vector("target Y integer",dtype='int32')
    weights = T.vector("weights",dtype='float64')


    input_layer_calo = lasagne.layers.InputLayer(shape = input_shape,input_var=input_X[:,0,:,:])
    input_layer_preshower = lasagne.layers.InputLayer(shape = input_shape,input_var=input_X[:,1,:,:])

    dense_1_calo = lasagne.layers.DenseLayer(input_layer_calo,num_units=ecal_units,
                               nonlinearity = lasagne.nonlinearities.tanh,
                               name = "hidden_dense_layer_calo")
    drop_out_calo = lasagne.layers.dropout(dense_1_calo, p=drop_)
    dense_2_calo = lasagne.layers.DenseLayer(drop_out_calo,num_units=ecal_units,
                               nonlinearity = lasagne.nonlinearities.tanh,
                               name = "hidden_dense_layer_calo")
    dense_1_ps = lasagne.layers.DenseLayer(input_layer_preshower,num_units=ps_units,
                               nonlinearity = lasagne.nonlinearities.tanh,
                               name = "hidden_dense_layer preshower")
    #drop_out_ps = lasagne.layers.dropout(dense_1_ps, p=drop_ps)
    dense_2_ps = lasagne.layers.DenseLayer(drop_out_ps,num_units=ps_units,
                               nonlinearity = lasagne.nonlinearities.tanh,
                               name = "hidden_dense_layer preshower")


    dense_2 = lasagne.layers.ConcatLayer([dense_2_calo,dense_2_ps],name='concatenated dense layers')
    
    if (part == 'All'):
        #print 'All'
        dense_output = lasagne.layers.DenseLayer(dense_2, num_units = 2,
                                        nonlinearity = lasagne.nonlinearities.softmax,
                                        name='output')
    elif (part == 'PS'):
        dense_output = lasagne.layers.DenseLayer(dense_2_ps, num_units = 2,
                                        nonlinearity = lasagne.nonlinearities.softmax,
                                        name='output')

    elif (part == 'Calo'):
        dense_output = lasagne.layers.DenseLayer(dense_2_calo, num_units = 2,
                                        nonlinearity = lasagne.nonlinearities.softmax,
                                        name='output')
    else:
        raise Exception('strange parametr')

    y_predicted = lasagne.layers.get_output(dense_output)
    all_weights = lasagne.layers.get_all_params(dense_output)

    l2_penalty = regularize_layer_params(dense_2, l2)
    l1_penalty = regularize_layer_params(dense_2, l1)

    loss_func = lasagne.objectives.categorical_crossentropy(y_predicted,target_y)+l1_reg*l1_penalty

    loss = loss_func.mean() + 1e-4*l2_penalty
    #loss = loss_func.mean()


    accuracy = lasagne.objectives.categorical_accuracy(y_predicted,target_y).mean()
    updates_sgd = lasagne.updates.adagrad(loss, all_weights, learning_rate=0.05)
    train_fun1 = theano.function([input_X,target_y],[loss,accuracy,y_predicted],updates= updates_sgd, allow_input_downcast=True)
    accuracy_fun1 = theano.function([input_X,target_y],accuracy)
    predict_fun = theano.function([input_X],y_predicted)
    
    return train_fun1, accuracy_fun1, predict_fun


def make_nn1(seed_, part='All', ecal_units=800, ps_units = 50, drop_ = 0, drop_ps = 0, l1_reg = 0, l2_reg=0):
    if (seed_ > -1):
        np.random.seed(seed_)
    theano.config.floatX = 'float32'
    input_X = T.tensor4("X")

    input_shape = [None,5,5]

    target_y = T.vector("target Y integer",dtype='int32')
    weights = T.vector("weights",dtype='float64')


    input_layer_calo = lasagne.layers.InputLayer(shape = input_shape,input_var=input_X[:,0,:,:])
    input_layer_preshower = lasagne.layers.InputLayer(shape = input_shape,input_var=input_X[:,1,:,:])

    dense_1_calo = lasagne.layers.DenseLayer(input_layer_calo,num_units=ecal_units,
                               nonlinearity = lasagne.nonlinearities.tanh,
                               name = "hidden_dense_layer_calo")
    drop_out_calo = lasagne.layers.dropout(dense_1_calo, p=drop_)

    dense_1_ps = lasagne.layers.DenseLayer(input_layer_preshower,num_units=ps_units,
                               nonlinearity = lasagne.nonlinearities.tanh,
                               name = "hidden_dense_layer preshower")
    #drop_out_ps = lasagne.layers.dropout(dense_1_ps, p=drop_ps)


    dense_1 = lasagne.layers.ConcatLayer([dense_1_calo,dense_1_ps],name='concatenated dense layers')
    
    if (part == 'All'):
        #print 'All'
        dense_output = lasagne.layers.DenseLayer(dense_1, num_units = 2,
                                        nonlinearity = lasagne.nonlinearities.softmax,
                                        name='output')
    elif (part == 'PS'):
        dense_output = lasagne.layers.DenseLayer(dense_1_ps, num_units = 2,
                                        nonlinearity = lasagne.nonlinearities.softmax,
                                        name='output')

    elif (part == 'Calo'):
        dense_output = lasagne.layers.DenseLayer(dense_1_calo, num_units = 2,
                                        nonlinearity = lasagne.nonlinearities.softmax,
                                        name='output')
    else:
        raise Exception('strange parametr')

    y_predicted = lasagne.layers.get_output(dense_output)
    all_weights = lasagne.layers.get_all_params(dense_output)

    l2_penalty = regularize_layer_params(dense_1, l2)
    l1_penalty = regularize_layer_params(dense_1, l1)

    loss_func = lasagne.objectives.categorical_crossentropy(y_predicted,target_y)+l1_reg*l1_penalty

    loss = loss_func.mean() + 1e-4*l2_penalty
    #loss = loss_func.mean()


    accuracy = lasagne.objectives.categorical_accuracy(y_predicted,target_y).mean()
    updates_sgd = lasagne.updates.adagrad(loss, all_weights, learning_rate=0.05)
    train_fun1 = theano.function([input_X,target_y],[loss,accuracy,y_predicted],updates= updates_sgd, allow_input_downcast=True)
    accuracy_fun1 = theano.function([input_X,target_y],accuracy)
    predict_fun = theano.function([input_X],y_predicted)
    
    return train_fun1, accuracy_fun1, predict_fun


def make_nn_250_10(seed_, part='All', drop_ = 0, drop_ps = 0):
    if (seed_ > -1):
        np.random.seed(seed_)
    theano.config.floatX = 'float32'
    input_X = T.tensor4("X")

    input_shape = [None,5,5]

    target_y = T.vector("target Y integer",dtype='int32')
    weights = T.vector("weights",dtype='float64')


    input_layer_calo = lasagne.layers.InputLayer(shape = input_shape,input_var=input_X[:,0,:,:])
    input_layer_preshower = lasagne.layers.InputLayer(shape = input_shape,input_var=input_X[:,1,:,:])

    dense_1_calo = lasagne.layers.DenseLayer(input_layer_calo,num_units=250,
                               nonlinearity = lasagne.nonlinearities.tanh,
                               name = "hidden_dense_layer_calo")
    drop_out_calo = lasagne.layers.dropout(dense_1_calo, p=drop_)
    dense_2_calo = lasagne.layers.DenseLayer(drop_out_calo,num_units=250,
                               nonlinearity = lasagne.nonlinearities.tanh,
                               name = "hidden_dense_layer_calo")
    dense_1_ps = lasagne.layers.DenseLayer(input_layer_preshower,num_units=10,
                               nonlinearity = lasagne.nonlinearities.tanh,
                               name = "hidden_dense_layer preshower")
    drop_out_ps = lasagne.layers.dropout(dense_1_ps, p=drop_ps)
    dense_2_ps = lasagne.layers.DenseLayer(drop_out_ps,num_units=10,
                               nonlinearity = lasagne.nonlinearities.tanh,
                               name = "hidden_dense_layer preshower")


    dense_2 = lasagne.layers.ConcatLayer([dense_2_calo,dense_2_ps],name='concatenated dense layers')
    
    if (part == 'All'):
        print 'All'
        dense_output = lasagne.layers.DenseLayer(dense_2, num_units = 2,
                                        nonlinearity = lasagne.nonlinearities.softmax,
                                        name='output')
    elif (part == 'PS'):
        dense_output = lasagne.layers.DenseLayer(dense_2_ps, num_units = 2,
                                        nonlinearity = lasagne.nonlinearities.softmax,
                                        name='output')

    elif (part == 'Calo'):
        dense_output = lasagne.layers.DenseLayer(dense_2_calo, num_units = 2,
                                        nonlinearity = lasagne.nonlinearities.softmax,
                                        name='output')
    else:
        raise Exception('strange parametr')

    y_predicted = lasagne.layers.get_output(dense_output)
    all_weights = lasagne.layers.get_all_params(dense_output)

    l2_penalty = regularize_layer_params(dense_2, l2)
    l1_penalty = regularize_layer_params(dense_2, l1)

    loss_func = lasagne.objectives.categorical_crossentropy(y_predicted,target_y)

    loss = loss_func.mean() + 1e-4*l2_penalty
    #loss = loss_func.mean()


    accuracy = lasagne.objectives.categorical_accuracy(y_predicted,target_y).mean()
    updates_sgd = lasagne.updates.adagrad(loss, all_weights)
    train_fun1 = theano.function([input_X,target_y],[loss,accuracy,y_predicted],updates= updates_sgd, allow_input_downcast=True)
    accuracy_fun1 = theano.function([input_X,target_y],accuracy)
    predict_fun = theano.function([input_X],y_predicted)
    
    return train_fun1, accuracy_fun1, predict_fun

def make_nn3(seed_, part='All', ecal_units=300, ps_units = 50, drop_ = 0.3, drop_ps = 0, l1_reg = 0, l2_reg=0):
    if (seed_ > -1):
        np.random.seed(seed_)
    theano.config.floatX = 'float32'
    input_X = T.tensor4("X")

    input_shape = [None,5,5]

    target_y = T.vector("target Y integer",dtype='int32')
    weights = T.vector("weights",dtype='float64')


    input_layer_calo = lasagne.layers.InputLayer(shape = input_shape,input_var=input_X[:,0,:,:])
    input_layer_preshower = lasagne.layers.InputLayer(shape = input_shape,input_var=input_X[:,1,:,:])

    dense_1_calo = lasagne.layers.DenseLayer(input_layer_calo,num_units=ecal_units,
                               nonlinearity = lasagne.nonlinearities.tanh,
                               name = "hidden_dense_layer_calo")
    drop_out_calo = lasagne.layers.dropout(dense_1_calo, p=drop_)
    dense_2_calo = lasagne.layers.DenseLayer(drop_out_calo,num_units=ecal_units,
                               nonlinearity = lasagne.nonlinearities.tanh,
                               name = "hidden_dense_layer_calo")
    drop_out_calo2 = lasagne.layers.dropout(dense_2_calo, p=drop_)
    dense_3_calo = lasagne.layers.DenseLayer(drop_out_calo2,num_units=ecal_units,
                               nonlinearity = lasagne.nonlinearities.tanh,
                               name = "hidden_dense_layer_calo")
    #dense_4_calo = lasagne.layers.DenseLayer(dense_3_calo,num_units=ecal_units,
                               #nonlinearity = lasagne.nonlinearities.tanh,
                               #name = "hidden_dense_layer_calo")
    dense_1_ps = lasagne.layers.DenseLayer(input_layer_preshower,num_units=ps_units,
                               nonlinearity = lasagne.nonlinearities.tanh,
                               name = "hidden_dense_layer preshower")
    #drop_out_ps = lasagne.layers.dropout(dense_1_ps, p=drop_ps)
    dense_2_ps = lasagne.layers.DenseLayer(dense_1_ps,num_units=ps_units,
                               nonlinearity = lasagne.nonlinearities.tanh,
                               name = "hidden_dense_layer preshower")
    dense_3_ps = lasagne.layers.DenseLayer(dense_2_ps,num_units=ps_units,
                               nonlinearity = lasagne.nonlinearities.tanh,
                               name = "hidden_dense_layer preshower")
    #dense_4_ps = lasagne.layers.DenseLayer(dense_3_ps,num_units=ps_units,
                               #nonlinearity = lasagne.nonlinearities.tanh,
                               #name = "hidden_dense_layer preshower")


    dense_3 = lasagne.layers.ConcatLayer([dense_3_calo,dense_3_ps],name='concatenated dense layers')
    
    if (part == 'All'):
        print 'All'
        dense_output = lasagne.layers.DenseLayer(dense_3, num_units = 2,
                                        nonlinearity = lasagne.nonlinearities.softmax,
                                        name='output')
    elif (part == 'PS'):
        dense_output = lasagne.layers.DenseLayer(dense_3_ps, num_units = 2,
                                        nonlinearity = lasagne.nonlinearities.softmax,
                                        name='output')

    elif (part == 'Calo'):
        dense_output = lasagne.layers.DenseLayer(dense_3_calo, num_units = 2,
                                        nonlinearity = lasagne.nonlinearities.softmax,
                                        name='output')
    else:
        raise Exception('strange parametr')

    y_predicted = lasagne.layers.get_output(dense_output)
    all_weights = lasagne.layers.get_all_params(dense_output)

    l2_penalty = regularize_layer_params(dense_3, l2)
    l1_penalty = regularize_layer_params(dense_3, l1)

    loss_func = lasagne.objectives.categorical_crossentropy(y_predicted,target_y)+l1_reg*l1_penalty

    loss = loss_func.mean() + 1e-4*l2_penalty
    #loss = loss_func.mean()


    accuracy = lasagne.objectives.categorical_accuracy(y_predicted,target_y).mean()
    updates_sgd = lasagne.updates.adagrad(loss, all_weights, learning_rate=0.05)
    train_fun1 = theano.function([input_X,target_y],[loss,accuracy,y_predicted],updates= updates_sgd, allow_input_downcast=True)
    accuracy_fun1 = theano.function([input_X,target_y],accuracy)
    predict_fun = theano.function([input_X],y_predicted)
    
    return train_fun1, accuracy_fun1, predict_fun

def make_nn4(seed_, part='All', ecal_units=300, ps_units = 50, drop_ = 0.3, drop_ps = 0, l1_reg = 0, l2_reg=0):
    if (seed_ > -1):
        np.random.seed(seed_)
    theano.config.floatX = 'float32'
    input_X = T.tensor4("X")

    input_shape = [None,5,5]

    target_y = T.vector("target Y integer",dtype='int32')
    weights = T.vector("weights",dtype='float64')


    input_layer_calo = lasagne.layers.InputLayer(shape = input_shape,input_var=input_X[:,0,:,:])
    input_layer_preshower = lasagne.layers.InputLayer(shape = input_shape,input_var=input_X[:,1,:,:])

    dense_1_calo = lasagne.layers.DenseLayer(input_layer_calo,num_units=ecal_units,
                               nonlinearity = lasagne.nonlinearities.tanh,
                               name = "hidden_dense_layer_calo")
    drop_out_calo = lasagne.layers.dropout(dense_1_calo, p=drop_)
    dense_2_calo = lasagne.layers.DenseLayer(drop_out_calo,num_units=ecal_units,
                               nonlinearity = lasagne.nonlinearities.tanh,
                               name = "hidden_dense_layer_calo")
    drop_out_calo2 = lasagne.layers.dropout(dense_2_calo, p=drop_)
    dense_3_calo = lasagne.layers.DenseLayer(drop_out_calo2,num_units=ecal_units,
                               nonlinearity = lasagne.nonlinearities.tanh,
                               name = "hidden_dense_layer_calo")
    dense_4_calo = lasagne.layers.DenseLayer(dense_3_calo,num_units=ecal_units,
                               nonlinearity = lasagne.nonlinearities.tanh,
                               name = "hidden_dense_layer_calo")
    dense_1_ps = lasagne.layers.DenseLayer(input_layer_preshower,num_units=ps_units,
                               nonlinearity = lasagne.nonlinearities.tanh,
                               name = "hidden_dense_layer preshower")
    #drop_out_ps = lasagne.layers.dropout(dense_1_ps, p=drop_ps)
    dense_2_ps = lasagne.layers.DenseLayer(dense_1_ps,num_units=ps_units,
                               nonlinearity = lasagne.nonlinearities.tanh,
                               name = "hidden_dense_layer preshower")
    dense_3_ps = lasagne.layers.DenseLayer(dense_2_ps,num_units=ps_units,
                               nonlinearity = lasagne.nonlinearities.tanh,
                               name = "hidden_dense_layer preshower")
    dense_4_ps = lasagne.layers.DenseLayer(dense_3_ps,num_units=ps_units,
                               nonlinearity = lasagne.nonlinearities.tanh,
                               name = "hidden_dense_layer preshower")


    dense_4 = lasagne.layers.ConcatLayer([dense_4_calo,dense_4_ps],name='concatenated dense layers')
    
    if (part == 'All'):
        print 'All'
        dense_output = lasagne.layers.DenseLayer(dense_4, num_units = 2,
                                        nonlinearity = lasagne.nonlinearities.softmax,
                                        name='output')
    elif (part == 'PS'):
        dense_output = lasagne.layers.DenseLayer(dense_4_ps, num_units = 2,
                                        nonlinearity = lasagne.nonlinearities.softmax,
                                        name='output')

    elif (part == 'Calo'):
        dense_output = lasagne.layers.DenseLayer(dense_4_calo, num_units = 2,
                                        nonlinearity = lasagne.nonlinearities.softmax,
                                        name='output')
    else:
        raise Exception('strange parametr')

    y_predicted = lasagne.layers.get_output(dense_output)
    all_weights = lasagne.layers.get_all_params(dense_output)

    l2_penalty = regularize_layer_params(dense_4, l2)
    l1_penalty = regularize_layer_params(dense_4, l1)

    loss_func = lasagne.objectives.categorical_crossentropy(y_predicted,target_y)+l1_reg*l1_penalty

    loss = loss_func.mean() + 1e-4*l2_penalty
    #loss = loss_func.mean()


    accuracy = lasagne.objectives.categorical_accuracy(y_predicted,target_y).mean()
    updates_sgd = lasagne.updates.adagrad(loss, all_weights, learning_rate=0.05)
    train_fun1 = theano.function([input_X,target_y],[loss,accuracy,y_predicted],updates= updates_sgd, allow_input_downcast=True)
    accuracy_fun1 = theano.function([input_X,target_y],accuracy)
    predict_fun = theano.function([input_X],y_predicted)
    
    return train_fun1, accuracy_fun1, predict_fun


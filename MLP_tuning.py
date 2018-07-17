# Author: Xu Yizhi <yzxu@zju.edu.cn>
# Date: 2018/6/27
# A tool for multi layer perceptron hyper-parameter fine-tuning

import numpy as np
import pandas as pd
import time
from scipy import sparse
from collections import Counter
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import np_utils
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from keras.callbacks import Callback
np.random.seed(2018)

num_cores = 30
CPU = True
GPU = not CPU

if GPU:
    num_GPU = 1
    num_CPU = 1
if CPU:
    num_CPU = 30
    num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
        				inter_op_parallelism_threads=num_cores, allow_soft_placement=True,
        				device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)

def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

roc_auc = as_keras_metric(tf.metrics.auc)
recall = as_keras_metric(tf.metrics.recall)
precision = as_keras_metric(tf.metrics.precision)

class MLP_tuning:

	def __init__(self, X, y, n_jobs=-1, hidden_layers=1, cv=5, epochs=30, batch_size=128, random_state=2018):
		self.X = X
		self.y = np_utils.to_categorical(y, 2)
		self.n_jobs = n_jobs
		self.hidden_layers = hidden_layers
		self.cv = cv
		# self.layer_num = layer_num
		# self.neuron_num = neuron_num
		self.epochs = epochs
		self.batch_size = batch_size
		self.random_state = random_state
		# self.optimization = optimization
		# self.learning_rate = learning_rate
		# self.momentum = momentum
		# self.weight_init = weight_init
		# self.activation = activation
		# self.dropout = dropout
		#self.early_stopping = EarlyStopping(monitor='val_loss', patience=2)
		self.param_grid = dict(neuron_num=-1, epochs=30, 
							   batch_size=128, activation='sigmoid', 
							   optimizer='rmsprop', dropout=0.2)

	def create_model(self, 
					 neuron_num=-1, optimizer='adam', learning_rate=0.1, 
				 	 momentum=0, weight_init='uniform', activation='sigmoid', 
				 	 dropout=0.2):
		if(neuron_num == -1):
			neuron_num=[int(np.sqrt(len(self.y)))]*self.hidden_layers
		inputs = Input(shape=(len(self.X[0]), ))
		x = Dense(neuron_num[0], activation=activation)(inputs)
		for i in range(self.hidden_layers - 1):
			x = Dropout(dropout)(x)
			x = Dense(neuron_num[i + 1], activation=activation)(x)
		predictions = Dense(2, activation='softmax')(x)
		model = Model(inputs=inputs, outputs=predictions)
		model.compile(optimizer=optimizer,
					  loss='binary_crossentropy',
					  metrics=['accuracy', roc_auc, recall])
		return model

	def fit_cv(self, 
			   neuron_num=-1, optimizer='rmsprop', learning_rate=0.1, 
			   momentum=0, weight_init='uniform', activation='sigmoid', 
			   dropout=0.2, batch_size=128, epochs=30):
		K.set_session(session)
		X = self.X
		labels = self.y
		n_split = self.cv
		scores = []
		if(neuron_num == -1):
			neuron_num=[int(np.sqrt(len(self.X[0])))]*self.hidden_layers
		inputs = Input(shape=(len(self.X[0]), ))
		x = Dense(neuron_num[0], activation=activation, kernel_initializer=weight_init)(inputs)
		for i in range(self.hidden_layers - 1):
			x = Dropout(dropout)(x)
			x = Dense(neuron_num[i + 1], activation=activation, kernel_initializer=weight_init)(x)
		predictions = Dense(2, activation='softmax')(x)
		model = Model(inputs=inputs, outputs=predictions)
		model.save_weights(r'G:\xuyizhi\app\gender_analysis\1\code\ITNN_empty.h5')

		early_stopping = EarlyStopping(monitor='val_loss', patience=2)
		tensorboard = TensorBoard(log_dir=r'G:\xuyizhi\app\gender_analysis\1\code\ITNN_log')
		callback_lists = [tensorboard, early_stopping]

		kf = KFold(n_splits=n_split, shuffle = True, random_state=self.random_state)
		for train_index, test_index in kf.split(labels):
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = labels[train_index], labels[test_index]
			model = Model(inputs=inputs, outputs=predictions)
			model.load_weights(r'G:\xuyizhi\app\gender_analysis\1\code\ITNN_empty.h5')
			model.compile(optimizer=optimizer,
						loss='categorical_crossentropy',
						metrics=['accuracy', roc_auc, recall, precision])
			model.fit(X_train, y_train,
					  epochs=epochs,
					  batch_size=batch_size,
					  validation_data=(X_test, y_test),
					  verbose=0, 
					  callbacks=callback_lists)
			scores.append(model.evaluate(X_test, y_test, verbose=0)[1])
		
		params_score = dict(neuron_num=neuron_num, epochs=epochs, 
					  		batch_size=batch_size, activation=activation, 
					  		optimizer=optimizer, dropout=dropout,
					  		score=np.array(scores).mean())

		return params_score

	def grid_search(self, param_grid):
		params_scores = []
		params_list = ['neuron_num', 'epochs', 'batch_size', 'activation', 'optimizer', 'dropout']

		def to_list(x):
			if(type(x) == list):
				return x
			else:
				tmp_list = []
				tmp_list.append(x)
				return tmp_list

		neuron_num = to_list(param_grid['neuron_num'])
		if(type(neuron_num[0]) == list):
			pass
		else:
			neuron_num = []
			neuron_num.append(to_list(param_grid['neuron_num']))
		
		epochs = to_list(param_grid['epochs'])
		batch_size = to_list(param_grid['batch_size'])
		activation = to_list(param_grid['activation'])
		optimizer = to_list(param_grid['optimizer'])
		dropout = to_list(param_grid['dropout'])

		for i0 in range(len(neuron_num)):
			for i1 in range(len(epochs)):
				for i2 in range(len(batch_size)):
					for i3 in range(len(activation)):
						for i4 in range(len(optimizer)):
							for i5 in range(len(dropout)):
								params_scores.append(self.fit_cv(neuron_num=neuron_num[i0], epochs=epochs[i1], 
														  batch_size=batch_size[i2], activation=activation[i3], 
														  optimizer=optimizer[i4], dropout=dropout[i5]))
		
		params_scores.sort(key=lambda x: x['score'], reverse=True)
		return params_scores[0]


	def tune_neuron_num(self, param_grid):
		print('start tuning neuron numbers......')
		start_time = time.time()
		power2 = list(map(lambda x: 2**x, np.arange(10)))
		top_layer = int(2*np.sqrt(len(self.X[0])))
		for i in range(len(power2)):
			if(top_layer <= power2[i]):
				top_layer = power2[i]/2
				break
		
		print('top_layer_num...%d'%(top_layer))
		neuron_num = []
		if(self.hidden_layers == 1):
			for i in range(2, int(np.log2(top_layer) + 1)):
				neuron_num.append([2**i])
		elif(self.hidden_layers == 2):
			for i in range(2, int(np.log2(top_layer) + 1)):
				for j in range(2, i + 1):
					neuron_num.append([2**i, 2**j])
		elif(self.hidden_layers == 3):
			for i in range(2, int(np.log2(top_layer) + 1)):
				for j in range(2, i + 1):
					for k in range(2, j + 1):
						neuron_num.append([2**i, 2**j, 2**k])

		param_grid['neuron_num'] = neuron_num
		results = self.grid_search(param_grid)
		print('finish tuning neuron number in %d seconds'%(time.time() - start_time))
		print('the best parameters and score is %s'%(results))
		del results['score']
		return results

	def tune_epochs_batch_size(self, param_grid):
		print('start tuning epochs and batch_size......')
		start_time = time.time()
		epochs = [30, 50]
		batch_size = [32, 64, 128]
		param_grid['epochs'] = epochs
		param_grid['batch_size'] = batch_size
		results = self.grid_search(param_grid)
		print('finish tuning epochs and batch_size in %d seconds'%(time.time() - start_time))
		print('the best parameters and score is %s'%(results))
		del results['score']
		return results

	def tune_activation(self, param_grid):
		print('start tuning activation......')
		start_time = time.time()
		activation = ['sigmoid', 'relu', 'tanh', 'hard_sigmoid', 'linear']
		param_grid['activation'] = activation
		results = self.grid_search(param_grid)
		print('finish tuning activation in %d seconds'%(time.time() - start_time))
		print('the best parameters and score is %s'%(results))
		del results['score']
		return results

	def tune_optimizer(self, param_grid):
		print('start tuning optimizer......')
		start_time = time.time()
		optimizer = ['sgd', 'rmsprop', 'adam', 'nadam']
		param_grid['optimizer'] = optimizer
		results = self.grid_search(param_grid)
		print('finish tuning optimizer in %d seconds'%(time.time() - start_time))
		print('the best parameters and score is %s'%(results))
		del results['score']
		return results

	def tune_dropout(self, param_grid):
		print('start tuning dropout......')
		start_time = time.time()
		dropout = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
		param_grid['dropout'] = dropout
		results = self.grid_search(param_grid)
		print('finish tuning dropout in %d seconds'%(time.time() - start_time))
		print('the best parameters and score is %s'%(results))
		del results['score']
		return results
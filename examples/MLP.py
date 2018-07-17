from my_tools.MLP_tuning import MLP_tuning
from my_tools.GBDT_tuning import GBDT_tuning
from sklearn.preprocessing import MinMaxScaler, scale
import numpy as np
from scipy import sparse
import os

if __name__ == '__main__':
	path = os.path.abspath('.')
	np.random.seed(2018)
	X = np.random.randint(10, size=(1000, 200))
	y = np.concatenate((np.zeros(500), np.ones(500)))

	#load data
	user_vector = np.load(r'G:\xuyizhi\app\gender_analysis\1\data\user_vector.npy')
	user_vector_5 = np.load(r'G:\xuyizhi\app\gender_analysis\1\data\user_vector_5.npy')
	user_color_bins = np.load(r'G:\xuyizhi\app\gender_analysis\1\data\user_color_bins.npy')
	X_ig_sparse = sparse.load_npz(r'G:\xuyizhi\app\gender_analysis\1\data\X_ig_sparse.npz')
	X_ig = X_ig_sparse.toarray()
	minmaxscaler = MinMaxScaler()

	X_lda100 = np.load(r'G:\xuyizhi\app\gender_analysis\1\data\X_lda100.npy')
	X_lda200 = np.load(r'G:\xuyizhi\app\gender_analysis\1\data\X_lda200.npy')
	X_lda300 = np.load(r'G:\xuyizhi\app\gender_analysis\1\data\X_lda300.npy')
	X_lda400 = np.load(r'G:\xuyizhi\app\gender_analysis\1\data\X_lda400.npy')
	X_lda500 = np.load(r'G:\xuyizhi\app\gender_analysis\1\data\X_lda500.npy')
	#user_vector = minmaxscaler.fit_transform(user_vector)
	#user_vector_5 = minmaxscaler.fit_transform(user_vector_5)
	user_color_bins = minmaxscaler.fit_transform(user_color_bins)
	y = np.concatenate((np.ones(7500), np.zeros(7500)))
	X = np.concatenate((X_ig, user_vector, user_color_bins), axis=1)
	params = dict(neuron_num=-1, epochs=30, 
					  batch_size=128, activation='sigmoid', 
					  optimizer='rmsprop', dropout=0.2)

	MLPTuning = MLP_tuning(X=np.concatenate((X_ig, X_lda300), axis=1), y=y, n_jobs=1, hidden_layers=3, cv=5, random_state=2018)
	params = MLPTuning.tune_neuron_num(param_grid=params)
	params = MLPTuning.tune_activation(param_grid=params)
	#params = MLPTuning.tune_epochs_batch_size(param_grid=params)
	params = MLPTuning.tune_optimizer(param_grid=params)
	params = MLPTuning.tune_dropout(param_grid=params)

	# GBDTTuning = GBDT_tuning(X=user_vector, y=y, cv=5, n_jobs=20)
	# # grid_scores_, best_params_, best_score_ = GBDTTuning.randomsearch()
	# # print(grid_scores_)
	# # print("Best: %f using %s" % (best_score_, best_params_))
	# GBDTTuning.tune()

from my_tools.MLP_tuning import MLP_tuning
from my_tools.GBDT_tuning import GBDT_tuning
from my_tools.LR_tuning import LR_tuning
import numpy as np
from scipy import sparse

if __name__ == '__main__':
	np.random.seed(2018)
	X = np.random.randint(10, size=(1000, 200))
	y = np.concatenate((np.zeros(500), np.ones(500)))

	#load data
	user_vector = np.load(r'G:\xuyizhi\app\gender_analysis\1\data\user_vector.npy')
	user_vector_5 = np.load(r'G:\xuyizhi\app\gender_analysis\1\data\user_vector_5.npy')
	user_color_bins = np.load(r'G:\xuyizhi\app\gender_analysis\1\data\user_color_bins.npy')
	X_ig_sparse = sparse.load_npz(r'G:\xuyizhi\app\gender_analysis\1\data\X_ig_sparse.npz')
	X_ig = X_ig_sparse.toarray()
	y = np.concatenate((np.ones(7500), np.zeros(7500)))
	X = np.concatenate((X_ig, user_vector_5), axis=1)

	# MLPTuning = MLP_tuning(X=X, y=y, n_jobs=30, hidden_layers=1, cv=3)
	# print(MLPTuning.tune_epochs_batch_size())

	
	# grid_scores_, best_params_, best_score_ = GBDTTuning.randomsearch()
	# print(grid_scores_)
	# print("Best: %f using %s" % (best_score_, best_params_))
	GBDTTuning = GBDT_tuning(X=np.concatenate((X_ig, user_vector, user_color_bins), axis=1), y=y, cv=5, n_jobs=30, random_state=2018)
	GBDTTuning.tune()

	#LRTuning = LR_tuning(X=user_vector_5, y=y, cv=5, n_jobs=30, random_state=2018)
	#LRTuning.tune()

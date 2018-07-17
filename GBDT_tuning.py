# Author: Xu Yizhi <yzxu@zju.edu.cn>
# Date: 2018/6/27
# A tool for GBDT hyper-parameter fine-tuning

from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from scipy import sparse
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import time
np.random.seed(1024)

class GBDT_tuning:

	def __init__(self, X, y, cv=5, n_jobs=30, random_state=2018):
		self.X = X
		self.y = y
		self.cv = cv
		self.n_jobs = n_jobs
		self.random_state = random_state

	def gridsearch(self, param_grid):
		gsearch = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=1000,
                                min_samples_leaf=20, max_features='sqrt', subsample=0.8, random_state=self.random_state), 
                       			param_grid=param_grid, scoring='accuracy', cv=self.cv, n_jobs = self.n_jobs, 
                       			verbose=2)
		gsearch.fit(self.X, self.y)
		return gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_

	def randomsearch(self, param_distributions):
		gsearch1 = RandomizedSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=1000,
                                min_samples_leaf=20, max_features='sqrt', subsample=0.8, random_state=self.random_state), 
                       			param_distributions=param_distributions, scoring='accuracy', cv=self.cv, n_jobs = self.n_jobs, 
                       			n_iter=20, verbose=2)
		rsearch.fit(self.X, self.y)
		return rsearch.grid_scores_, rsearch.best_params_, rsearch.best_score_

	def step0(self, search_type, params):
		params['n_estimators'] = list(np.arange(100, 1001, 100))
		params['learning_rate'] = [0.01, 0.02, 0.05, 0.1]

		if(search_type == 'grid'):
			return self.gridsearch(params)
		elif(search_type == 'random'):
			return self.randomsearch(params)

	def step1(self, search_type, params):
		params['max_depth'] = range(3, 11, 1)
		params['min_samples_split'] = (len(self.y)/np.arange(10, 310, 30)).astype(int)

		if(search_type == 'grid'):
			return self.gridsearch(params)
		elif(search_type == 'random'):
			return self.randomsearch(params)

	def step2(self, search_type, params):
		params['min_samples_leaf'] = (len(self.y)/np.arange(100, 3100, 300)).astype(int)
		params['min_samples_split'] = (len(self.y)/np.arange(10, 310, 30)).astype(int)

		if(search_type == 'grid'):
			return self.gridsearch(params)
		elif(search_type == 'random'):
			return self.randomsearch(params)

	def step3(self, search_type, params):
		params['subsample'] = [0.6, 0.7, 0.8 ,0.9, 1]

		if(search_type == 'grid'):
			return self.gridsearch(params)
		elif(search_type == 'random'):
			return self.randomsearch(params)

	def step4(self, search_type, params):
		params['max_features'] = (np.array([0.25, 0.5, 1, 1.5, 2])*np.sqrt(len(self.X[0]))).astype(int)

		if(search_type == 'grid'):
			return self.gridsearch(params)
		elif(search_type == 'random'):
			return self.randomsearch(params)

	def tune(self):
		start_time = time.time()
		params = dict()
		for i in range(5):
			_0, best_params_, best_scores_ = eval('self.step%d(\'grid\', params)'%(i))
			for key, value in best_params_.items():
				if(type(value) == str):
					params[key] = eval('[\'%s\']'%(value))
				else:
					params[key] = eval('[%s]'%(value))
		print(params, best_scores_)
		print('finish tuning GBDT model in %d seconds'%(time.time() - start_time))
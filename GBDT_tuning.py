# Author: Xu Yizhi <yzxu@zju.edu.cn>
# Date: 2018/6/27
# A tool for GBDT hyper-parameter fine-tuning
# Also as an implementation of https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
# TODO Change hyper-parameters according to the traits of the dataset
# TODO Search wider range of parameters if the best one is on the margin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from scipy import sparse
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import time
np.random.seed(1024)

class GBDT_tuning:

	def __init__(self, X, y, cv=5, n_jobs=30, random_state=2018, model_type='classification', scoring='accuracy'):
		self.X = X
		self.y = y
		self.cv = cv
		self.n_jobs = n_jobs
		self.random_state = random_state
		self.model_type = model_type
		self.scoring = scoring

	def gridsearch(self, param_grid):
		if(self.model_type == 'classification' or self.model_type == 'clf'):
			estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=1000,
                                				   min_samples_leaf=20, max_features='sqrt', subsample=0.8, random_state=self.random_state)
		elif(self.model_type == 'regression' or self.model_type == 'reg'):
			estimator = GradientBoostingRegressor(learning_rate=0.1, min_samples_split=1000,
                                				  min_samples_leaf=20, max_features='sqrt', subsample=0.8, random_state=self.random_state)

		gsearch = GridSearchCV(estimator=estimator, 
                       		   param_grid=param_grid, scoring=self.scoring, cv=self.cv, n_jobs = self.n_jobs, 
                       		   verbose=2)
		gsearch.fit(self.X, self.y)
		return gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_

	def randomsearch(self, param_distributions):
		if(self.model_type == 'classification' or self.model_type == 'clf'):
			estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=1000,
                                				   min_samples_leaf=20, max_features='sqrt', subsample=0.8, random_state=self.random_state)
		elif(self.model_type == 'regression' or self.model_type == 'reg'):
			estimator = GradientBoostingRegressor(learning_rate=0.1, min_samples_split=1000,
                                				  min_samples_leaf=20, max_features='sqrt', subsample=0.8, random_state=self.random_state)

		rsearch = RandomizedSearchCV(estimator=estimator, 
                       				  param_distributions=param_distributions, scoring=self.scoring, cv=self.cv, n_jobs = self.n_jobs, 
                       				  n_iter=20, verbose=2)
		rsearch.fit(self.X, self.y)
		return rsearch.grid_scores_, rsearch.best_params_, rsearch.best_score_

	def search(self, search_type, params):
		if(search_type == 'grid'):
			return self.gridsearch(params)
		elif(search_type == 'random'):
			return self.randomsearch(params)

	def step0(self, search_type, params):
		params['n_estimators'] = list(np.arange(20, 71, 10))
		params['learning_rate'] = [0.05, 0.1, 0.2]

		return self.search(search_type, params)

	def step1(self, search_type, params):
		params['max_depth'] = range(5, 15, 2)
		params['min_samples_split'] = (len(self.y)*np.arange(0.002, 0.01, 0.002)).astype(int)

		return self.search(search_type, params)

	def step2(self, search_type, params):
		params['min_samples_split'] = (len(self.y)*np.arange(0.01, 0.02, 0.002)).astype(int)
		params['min_samples_split'] = (len(self.y)/np.arange(30, 71, 10)).astype(int)

		return self.search(search_type, params)

	def step3(self, search_type, params):
		params['max_features'] = (np.array([0.25, 0.5, 1, 1.5, 2])*np.sqrt(len(self.X[0]))).astype(int)

		return self.search(search_type, params)

	def step4(self, search_type, params):
		params['subsample'] = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]

		return self.search(search_type, params)

	def step5(self, search_type, params):
		params['n_estimators'] = list(params['n_estimators'][0] * np.array([2, 10, 20]))
		params['learning_rate'] = list(params['learning_rate'][0] * np.array([0.5, 0.1, 0.05]))

		return self.search(search_type, params)

	def tune(self):
		steps = 6
		start_time = time.time()
		params = dict()
		for i in range(steps):
			_0, best_params_, best_scores_ = eval('self.step%d(\'grid\', params)'%(i))
			for key, value in best_params_.items():
				if(type(value) == str):
					params[key] = eval('[\'%s\']'%(value))
				else:
					params[key] = eval('[%s]'%(value))
		print(params, best_scores_)
		print('finish tuning GBDT model in %d seconds'%(time.time() - start_time))
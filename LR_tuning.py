# Author: Xu Yizhi <yzxu@zju.edu.cn>
# Date: 2018/6/27
# A tool for logistic regression hyper-parameter fine-tuning

from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy import sparse
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import time
np.random.seed(1024)

class LR_tuning:

	def __init__(self, X, y, cv=5, n_jobs=30, random_state=2018):
		self.X = X
		self.y = y
		self.cv = cv
		self.n_jobs = n_jobs
		self.random_state = random_state

	def gridsearch(self, param_grid):
		gsearch = GridSearchCV(estimator=LogisticRegression(penalty='l2', 
							   solver='liblinear', C = 1.0, 
                               random_state=self.random_state), 
                       		   param_grid=param_grid, scoring='accuracy', 
                       		   cv=self.cv, n_jobs = self.n_jobs, 
                       		   verbose=2)
		gsearch.fit(self.X, self.y)
		return gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_

	def randomsearch(self, param_distributions):
		gsearch1 = RandomizedSearchCV(estimator=LogisticRegression(penalty='l2', 
                                	  solver='liblinear', C = 1.0, 
                                	  random_state=self.random_state), 
                       				  param_distributions=param_distributions, scoring='accuracy', 
                       				  cv=self.cv, n_jobs = self.n_jobs, 
                       				  n_iter=20, verbose=2)
		rsearch.fit(self.X, self.y)
		return rsearch.grid_scores_, rsearch.best_params_, rsearch.best_score_

	def step0(self, search_type, params):
		params['C'] = np.logspace(-4, 3, 20).astype(np.float16)

		if(search_type == 'grid'):
			return self.gridsearch(params)
		elif(search_type == 'random'):
			return self.randomsearch(params)

	def step1(self, search_type, params):
		#params['penalty'] = ['l1', 'l2']#TODO l1 penalty support
		params['solver'] = ['newton-cg', 'lbfgs', 'sag', 'saga', 'liblinear']

		if(search_type == 'grid'):
			return self.gridsearch(params)
		elif(search_type == 'random'):
			return self.randomsearch(params)


	def tune(self):
		start_time = time.time()
		params = dict()
		for i in range(2):
			_0, best_params_, best_scores_ = eval('self.step%d(\'grid\', params)'%(i))
			for key, value in best_params_.items():
				if(type(value) == str):
					params[key] = eval('[\'%s\']'%(value))
				else:
					params[key] = eval('[%s]'%(value))
		print(params, best_scores_)
		print('finish tuning LR model in %d seconds'%(time.time() - start_time))
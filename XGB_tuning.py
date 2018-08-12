# Author: Xu Yizhi <yzxu@zju.edu.cn>
# Date: 2018/7/23
# A tool for XGBoost hyper-parameter fine-tuning
# Also as an implementation of https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
# TODO Change hyper-parameters according to the traits of the dataset
# TODO Search wider range of parameters if the best one is on the margin
import xgboost as xgb
import numpy as np
from scipy import sparse
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import *
import time
np.random.seed(1024)

class XGB_tuning:

	def __init__(self, X, y, cv=5, n_jobs=30, random_state=2018, model_type='classification', scoring='accuracy'):
		self.X = X
		self.y = y
		self.cv = cv
		self.n_jobs = n_jobs
		self.random_state = random_state
		self.model_type = model_type
		self.scoring = scoring

	def choose_model_type(self):
		if(self.model_type == 'classification' or self.model_type == 'clf'):
			estimator = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=10000, 
										  min_child_weight=int(np.sqrt(len(self.y))), gamma=0, scale_pos_weight=1, 
										  subsample=0.8, colsample_bytree=0.8, random_state=self.random_state)
		elif(self.model_type == 'regression' or self.model_type == 'reg'):
			estimator = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=10000, 
										 min_child_weight=int(np.sqrt(len(self.y))), gamma=0, scale_pos_weight=1, 
										 subsample=0.8, colsample_bytree=0.8, random_state=self.random_state)

		return estimator

	def fit_cv_adhoc(self, param_grid):
		estimator = self.choose_model_type()
		X = self.X
		labels = self.y
		n_split = self.cv
		scores = []
		print(len(X) == len(labels))
		kf = KFold(n_splits=n_split, shuffle = True, random_state=self.random_state)
		for train_index, test_index in kf.split(labels):
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = labels[train_index], labels[test_index]
			estimator.fit(X_train, y_train)
			y_pred = estimator.predict(X_test)
			y_pred[np.isnan(y_pred)] = 0
			y_pred = np.array([x if x > 0 else 0 for x in y_pred])
			scores.append(mean_squared_log_error(y_test, y_pred))
		
		params_score = param_grid
		params_score['score'] = np.array(scores).mean()

		return params_score

	def gridsearch_adhoc(self, param_grid):
		params_scores = []
		params_list = ['max_depth', 'min_child_weight', 'gamma', 'subsample', 'colsample_bytree', 'reg_alpha', 'learning_rate']

		def to_list(x):
			if(type(x) == list):
				return x
			else:
				tmp_list = []
				tmp_list.append(x)
				return tmp_list
		
		max_depth = to_list(param_grid['max_depth'])
		min_child_weight = to_list(param_grid['min_child_weight'])
		gamma = to_list(param_grid['gamma'])
		subsample = to_list(param_grid['subsample'])
		colsample_bytree = to_list(param_grid['colsample_bytree'])
		reg_alpha = to_list(param_grid['reg_alpha'])
		learning_rate = to_list(param_grid['learning_rate'])

		for i0 in range(len(max_depth)):
			for i1 in range(len(min_child_weight)):
				for i2 in range(len(gamma)):
					for i3 in range(len(subsample)):
						for i4 in range(len(colsample_bytree)):
							for i5 in range(len(reg_alpha)):
								for i6 in range(len(learning_rate)):
									params_scores.append(self.fit_cv_adhoc({'max_depth': max_depth[i0], 'min_child_weight': min_child_weight[i1], 
														 	 		 		'gamma': gamma[i2], 'subsample': subsample[i3], 
														  			 		'colsample_bytree': colsample_bytree[i4], 'reg_alpha': reg_alpha[i5], 
														  			 		'learning_rate': learning_rate[i6]}))
		
		params_scores.sort(key=lambda x: x['score'], reverse=True)
		best_score_ = params_scores[0]['score']
		del params_scores[0]['score']
		return True, params_scores[0], best_score_


	def gridsearch(self, param_grid):
		estimator = self.choose_model_type()

		gsearch = GridSearchCV(estimator=estimator, 
                       		   param_grid=param_grid, scoring=self.scoring, cv=self.cv, n_jobs = self.n_jobs, 
                       		   verbose=2)
		gsearch.fit(self.X, self.y)
		return gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_

	def randomsearch(self, param_distributions):
		estimator = self.choose_model_type()

		rsearch = RandomizedSearchCV(estimator=estimator, 
                       				 param_distributions=param_distributions, scoring=self.scoring, cv=self.cv, n_jobs = self.n_jobs, 
                       				 n_iter=20, verbose=2)
		rsearch.fit(self.X, self.y)
		return rsearch.grid_scores_, rsearch.best_params_, rsearch.best_score_

	def search(self, search_type, params):
		if(self.scoring == 'neg_mean_squared_log_error'):
			return self.gridsearch_adhoc(params)
		elif(search_type == 'grid'):
			return self.gridsearch(params)
		elif(search_type == 'random'):
			return self.randomsearch(params)

	def fix_n_estimators(self):
		alg = self.choose_model_type()
		xgb_param = alg.get_xgb_params()
		early_stopping_rounds = 50
		metrics = {'roc_auc':'auc', 
				   'neg_mean_squared_error':'rmse',
				   'neg_mean_squared_log_error':'rmse',  
				   'neg_logloss':'log_loss', 
				   'accuracy':'error'}
		maximize = {'roc_auc':True, 
				   'neg_mean_squared_error':False, 
				   'neg_mean_squared_log_error':False, 
				   'neg_logloss':False, 
				   'accuracy':False}
		cvresult = xgb.cv(xgb_param, xgb.DMatrix(self.X, self.y), num_boost_round=alg.get_params()['n_estimators'], maximize = maximize[self.scoring], 
						  nfold=self.cv, metrics=metrics[self.scoring], early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
		return cvresult.shape[0]

	def step0(self, search_type, params):
		params['max_depth'] = range(3, 10, 2)
		params['min_child_weight'] = (np.array([0.1, 0.5, 1, 2, 5]) * np.sqrt(len(self.y))).astype(int)

		return self.search(search_type, params)

	def step1(self, search_type, params):
		params['gamma'] = [0.0, 0.1, 0.2, 0.3, 0.4]

		return self.search(search_type, params)

	def step2(self, search_type, params):
		params['subsample'] = [0.7, 0.75, 0.8, 0.85, 0.9]
		params['colsample_bytree'] = [0.7, 0.75, 0.8, 0.85, 0.9]

		return self.search(search_type, params)

	def step3(self, search_type, params):
		params['reg_alpha'] = [0.00001, 0.01, 0.1, 1, 100]

		return self.search(search_type, params)

	def step4(self, search_type, params):
		params['learning_rate'] = [0.05, 0.01]

		return self.search(search_type, params)

	def tune(self):
		steps = 5
		start_time = time.time()
		params = self.choose_model_type().get_params()
		params['n_estimators'] = [self.fix_n_estimators()]
		for i in range(steps):
			_0, best_params_, best_scores_ = eval('self.step%d(\'grid\', params)'%(i))
			print(best_params_)
			for key, value in best_params_.items():
				if(type(value) == str):
					params[key] = eval('[\'%s\']'%(value))
				else:
					params[key] = eval('[%s]'%(value))
		print(params, best_scores_)
		print('finish tuning XGB model in %d seconds'%(time.time() - start_time))
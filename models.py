from numpy.random import seed
seed(1)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, HistGradientBoostingRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoLars, BayesianRidge, PoissonRegressor, PassiveAggressiveRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.svm import SVR, LinearSVR
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def predict_regression(model_types, X_train, y_train, X_test, cv_num, trial_num):
    
    models = {'HGBR': HistGradientBoostingRegressor(), 'RF': RandomForestRegressor(), 'SVM': SVR(), 'GBR': GradientBoostingRegressor(), 'ABR': AdaBoostRegressor(),\
              'GPR': GaussianProcessRegressor(), 'R': Ridge(),'LL': LassoLars(), 'BR': BayesianRidge(), 'KNN': KNeighborsRegressor(),\
              'L': Lasso(), 'MLP': MLPRegressor(), 'LSVM': LinearSVR(), 'ET': ExtraTreeRegressor(), 'LR': LinearRegression(), \
              'PR': PoissonRegressor(), 'PAR': PassiveAggressiveRegressor(), 'XGB': XGBRegressor(), 'XGBRF': XGBRFRegressor(), 'LGBM': LGBMRegressor(),\
              'CBR': CatBoostRegressor(), 'BaR': BaggingRegressor()}
    y_pred = {}
    studies = {}
    finished_models = {}
    
    for model_type in model_types:
        pred, study, model = get_prediction(models, model_type, X_train, y_train, X_test, cv_num, trial_num) 
        y_pred[model_type] = pred
        studies[model_type] = study
        finished_models[model_type] = model
        print(f'done with {model_type}')
    
    return y_pred, studies, finished_models

def get_prediction(models, model_type, X_train, y_train, X_test, cv_num, trial_num):
    try:
        model = models[model_type]
    except:
        return f'No model with name {model_type} found!'
    
    study = optimize_hyperparameters(model_type, X_train, y_train, cv_num, trial_num)
    print(study.best_params)
    model.set_params(**study.best_params)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return y_pred, study, model

def objective(trial, model, X, y, cv_num):
    if model == 'ABR':
        abr_loss = trial.suggest_categorical('loss', ['linear', 'square', 'exponential'])
        abr_n_estimators = trial.suggest_int('n_estimators', 5, 200, step=5)
        abr_learning_rate = trial.suggest_float('learning_rate', 0.01, 10)
        
        classifier_obj = AdaBoostRegressor(loss=abr_loss, n_estimators=abr_n_estimators, learning_rate=abr_learning_rate)
    
    elif model == 'BaR':
        br_n_estimators = trial.suggest_int('n_estimators', 5, 200, step=5)
        br_max_samples = trial.suggest_int('max_samples', 1, 200)
        br_max_features = trial.suggest_int('max_features', 1, 50)
        br_bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        br_bootstrap_features = trial.suggest_categorical('bootstrap_features', [True, False])
        
        classifier_obj = BaggingRegressor(n_estimators=br_n_estimators, max_samples=br_max_samples, max_features=br_max_features, bootstrap=br_bootstrap,\
            bootstrap_features=br_bootstrap_features)
    
    elif model == 'BR':
        br_n_iter = trial.suggest_int('n_iter', 10, 510, step=50)
        br_alpha_1 = trial.suggest_float('alpha_1', 1e-10, 1e10, log=True)
        br_alpha_2 = trial.suggest_float('alpha_2', 1e-10, 1e10, log=True)
        br_lambda_1 = trial.suggest_float('lambda_1', 1e-10, 1e10, log=True)
        br_lambda_2 = trial.suggest_float('lambda_2', 1e-10, 1e10, log=True)
        br_fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
        
        classifier_obj = BayesianRidge(n_iter=br_n_iter, alpha_1=br_alpha_1, alpha_2=br_alpha_2, lambda_1=br_lambda_1, lambda_2=br_lambda_2, fit_intercept=br_fit_intercept)
        
    elif model == 'CBR':
        cbr_iterations = trial.suggest_int('iterations', 5, 120, step=5)
        cbr_depth = trial.suggest_int('depth', 1, 12)
        cbr_learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1)
        
        classifier_obj = CatBoostRegressor(iterations=cbr_iterations, depth=cbr_depth, learning_rate=cbr_learning_rate)
        
    elif model == 'ET':
        et_max_depth = trial.suggest_int('max_depth', 10, 100, step=10)
        et_min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        et_min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        et_max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
        
        classifier_obj = ExtraTreeRegressor(max_depth=et_max_depth, min_samples_leaf=et_min_samples_leaf, min_samples_split=et_min_samples_split, max_features=et_max_features)
        
    elif model == 'GBR':
        gbr_n_estimators = trial.suggest_int('n_estimators', 10, 250, step=10)
        gbr_max_depth = trial.suggest_int('max_depth', 10, 100, step=10)
        gbr_min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        gbr_min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        
        classifier_obj = GradientBoostingRegressor(n_estimators=gbr_n_estimators, max_depth=gbr_max_depth, min_samples_leaf=gbr_min_samples_leaf, min_samples_split=gbr_min_samples_split)
        
    elif model == 'GPR':
        gpr_alpha = trial.suggest_float('alpha', 1e-10, 1e10, log=True)
        
        classifier_obj = GaussianProcessRegressor(alpha=gpr_alpha)
        
    elif model == 'HGBR':
        hgbr_learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1)
        hgbr_max_depth = trial.suggest_int('max_depth', 10, 100, step=10)
        hgbr_min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 40)
        hgbr_max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 2, 80, step=3)
        
        classifier_obj = HistGradientBoostingRegressor(learning_rate=hgbr_learning_rate, max_depth=hgbr_max_depth, min_samples_leaf=hgbr_min_samples_leaf,\
            max_leaf_nodes=hgbr_max_leaf_nodes)
        
    elif model == 'KNN':
        knn_n_neighbors = trial.suggest_int('n_neighbors', 1, 20)
        knn_weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        knn_metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'chebyshev', 'minkowski'])
        knn_algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
        knn_leaf_size = trial.suggest_int('leaf_size', 5, 150, step=5)
        
        classifier_obj = KNeighborsRegressor(n_neighbors=knn_n_neighbors, weights=knn_weights, algorithm=knn_algorithm, metric=knn_metric, leaf_size=knn_leaf_size)
        
    elif model == 'L':
        l_alpha = trial.suggest_float('alpha', 1e-10, 1e10, log=True)
        l_fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
        
        classifier_obj = Lasso(alpha=l_alpha, fit_intercept=l_fit_intercept)
        
    elif model == 'LGBM':
        lgbm_n_estimators = trial.suggest_int('n_estimators', 5, 200, step=5)
        lgbm_max_depth = trial.suggest_int('max_depth', 10, 100, step=10)
        lgbm_num_leaves = trial.suggest_int('num_leaves', 10, 100, step=5)
        
        classifier_obj = LGBMRegressor(n_estimators=lgbm_n_estimators, max_depth=lgbm_max_depth, num_leaves=lgbm_num_leaves)
        
    elif model == 'LL':
        ll_alpha = trial.suggest_float('alpha', 1e-10, 1e10, log=True)
        ll_fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
        
        classifier_obj = LassoLars(alpha=ll_alpha, fit_intercept=ll_fit_intercept)
        
    elif model == 'LSVM':
        lsvm_c = trial.suggest_float('C', 1e-10, 1e10, log=True)
        lsvm_fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
        
        classifier_obj = LinearSVR(C=lsvm_c, fit_intercept=lsvm_fit_intercept)
        
    elif model == 'LR':
        lr_fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
        
        classifier_obj = LinearRegression(fit_intercept=lr_fit_intercept)
        
    elif model == 'MLP':
        mlp_activation = trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu'])
        mlp_solver = trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam'])
        mlp_alpha = trial.suggest_float('alpha', 1e-10, 1e10, log=True)
        mlp_batch_size = trial.suggest_int('batch_size', 8, 64, step=8)
        mlp_max_iter = trial.suggest_int('max_iter', 100, 500, step=100)
        
        classifier_obj = MLPRegressor(activation=mlp_activation, solver=mlp_solver, alpha=mlp_alpha, batch_size=mlp_batch_size, max_iter=mlp_max_iter)
        
    elif model == 'PAR':
        par_c = trial.suggest_float('C', 1e-10, 1e10, log=True)
        par_fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
        
        classifier_obj = PassiveAggressiveRegressor(C=par_c, fit_intercept=par_fit_intercept)
        
    elif model == 'PR':
        pr_alpha = trial.suggest_float('alpha', 1e-10, 1e10, log=True)
        pr_fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
        
        classifier_obj = PoissonRegressor(alpha=pr_alpha, fit_intercept=pr_fit_intercept)
        
    elif model == 'R':
        r_alpha = trial.suggest_float('alpha', 1e-10, 1e10, log=True)
        r_fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
        
        classifier_obj = Ridge(alpha=r_alpha, fit_intercept=r_fit_intercept)
        
    elif model == 'RF':
        rf_n_estimators = trial.suggest_int('n_estimators', 5, 200, step=5)
        rf_max_depth = trial.suggest_int('max_depth', 10, 150, step=10)
        rf_min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        rf_min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        rf_max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
        
        classifier_obj = RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=rf_max_depth, min_samples_split=rf_min_samples_split, min_samples_leaf=rf_min_samples_leaf,\
            max_features=rf_max_features, n_jobs=-1)
        
    elif model == 'SVM':
        svr_kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        svr_degree = trial.suggest_int('degree', 1, 10)
        svr_c = trial.suggest_float('C', 1e-10, 1e10, log=True)
        
        classifier_obj = SVR(kernel=svr_kernel, degree=svr_degree, C=svr_c)
        
    elif model == 'XGB':
        xgb_n_estimators = trial.suggest_int('n_estimators', 5, 200, step=5)
        xgb_max_dept = trial.suggest_int('max_depth', 3, 18)
        xgb_gamma = trial.suggest_int('gamma', 1, 10)
        xgb_reg_alpha = trial.suggest_int('reg_alpha', 4, 180)
        xgb_reg_lambda = trial.suggest_uniform('reg_lambda', 0, 1)
        xgb_min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
        xgb_eta = trial.suggest_float('eta', 0.001, 1)
        
        classifier_obj = XGBRegressor(n_estimators=xgb_n_estimators ,max_depth=xgb_max_dept, gamma=xgb_gamma, reg_alpha=xgb_reg_alpha, reg_lambda=xgb_reg_lambda,\
            min_child_weight=xgb_min_child_weight, eta=xgb_eta)
        
    elif model == 'XGBRF':
        xgbrf_n_estimators = trial.suggest_int('n_estimators', 5, 200, step=5)
        xgbrf_max_dept = trial.suggest_int('max_depth', 3, 18)
        xgbrf_gamma = trial.suggest_int('gamma', 1, 10)
        xgbrf_reg_alpha = trial.suggest_int('reg_alpha', 4, 180)
        xgbrf_reg_lambda = trial.suggest_uniform('reg_lambda', 0, 1)
        xgbrf_min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
        xgbrf_eta = trial.suggest_float('eta', 0.001, 1)
        
        classifier_obj = XGBRFRegressor(n_estimators=xgbrf_n_estimators ,max_depth=xgbrf_max_dept, gamma=xgbrf_gamma, reg_alpha=xgbrf_reg_alpha, reg_lambda=xgbrf_reg_lambda,\
            min_child_weight=xgbrf_min_child_weight, eta=xgbrf_eta)

    score = cross_val_score(classifier_obj, X, y, n_jobs=2, cv=cv_num, scoring='neg_mean_squared_error')
    mse = score.mean()
    
    return mse

def optimize_hyperparameters(model, X, y, cv_num, trial_num):
    study = optuna.create_study(direction='maximize') # maximize because it maximizes the negative MSE thus minimizing the MSE
    study.optimize(lambda trial: objective(trial, model, X, y, cv_num), n_trials=trial_num)
    
    return study
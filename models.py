from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, LassoLarsCV, BayesianRidge, PoissonRegressor, PassiveAggressiveRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.svm import SVR, LinearSVR
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def predict_regression(model_types, X_train, y_train, X_test):
    
    models = {'HGBR': HistGradientBoostingRegressor(), 'RF': RandomForestRegressor(), 'SVM': SVR(), 'GBR': GradientBoostingRegressor(), 'ABR': AdaBoostRegressor(),\
              'GPR': GaussianProcessRegressor(), 'RCV': RidgeCV(),'LLCV': LassoLarsCV(), 'BR': BayesianRidge(), 'KNN': KNeighborsRegressor(),\
              'LCV': LassoCV(), 'MLP': MLPRegressor(), 'LSVM': LinearSVR(), 'ET': ExtraTreeRegressor(), 'LR': LinearRegression(), \
              'PR': PoissonRegressor(), 'PAR': PassiveAggressiveRegressor(), 'XGB': XGBRegressor(), 'XGBRF': XGBRFRegressor(), 'LGBM': LGBMRegressor(),\
              'CBR': CatBoostRegressor(), 'RNR': RadiusNeighborsRegressor()}
    y_pred = {}
    
    for model_type in model_types:
        y_pred[model_type] = get_prediction(models, model_type, X_train, y_train, X_test)
        print(f'done with {model_type}')
    
    return y_pred

def get_prediction(models, model_type, X_train, y_train, X_test):
    try:
        model = models[model_type]
    except:
        return f'No model with name {model_type} found!'
    
    params = optimize_hyperparameters(model_type, X_train, y_train)
    print(params)
    model.set_params(**params)
        
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print(model)
    
    return y_pred

def objective(trial, model, X, y):
    if model == 'ABR':
        abr_loss = trial.suggest_categorical('loss', ['linear', 'square', 'exponential'])
        
        classifier_obj = AdaBoostRegressor(loss=abr_loss)
        
    elif model == 'BR':
        
        
        classifier_obj = BayesianRidge()
        
    elif model == 'CBR':
        
        
        classifier_obj = CatBoostRegressor()
        
    elif model == 'ET':
        
        
        classifier_obj = ExtraTreeRegressor()
        
    elif model == 'GBR':
        
        
        classifier_obj = GradientBoostingRegressor()
        
    elif model == 'GPR':
        
        
        classifier_obj = GaussianProcessRegressor()
        
    elif model == 'HGBR':
        
        
        classifier_obj = HistGradientBoostingRegressor()
        
    elif model == 'KNN':
        
        
        classifier_obj = KNeighborsRegressor()
        
    elif model == 'LCV':
        
        
        classifier_obj = LassoCV()
        
    elif model == 'LGBM':
        lgbm_n_estimators = trial.suggest_int('n_estimators', 5, 200, step=5)
        
        classifier_obj = LGBMRegressor(n_estimators=lgbm_n_estimators)
        
    elif model == 'LLCV':
        
        
        classifier_obj = LassoLarsCV()
        
    elif model == 'LSVM':
        
        
        classifier_obj = LinearSVR()
        
    elif model == 'MLP':
        
        
        classifier_obj = MLPRegressor()
        
    elif model == 'PAR':
        
        
        classifier_obj = PassiveAggressiveRegressor()
        
    elif model == 'PR':
        
        
        classifier_obj = PoissonRegressor()
        
    elif model == 'RCV':
        
        
        classifier_obj = RidgeCV()
        
    elif model == 'RF':
        rf_n_estimators = trial.suggest_int('n_estimators', 5, 200, step=5)
        rf_max_depth = trial.suggest_int('max_depth', 10, 100, step=10)
        
        classifier_obj = RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=rf_max_depth)
        
    elif model == 'RNR':
        
        
        classifier_obj = RadiusNeighborsRegressor()
        
    elif model == 'SVM':
        
        
        classifier_obj = SVR()
        
    elif model == 'XGB':
        
        
        classifier_obj = XGBRegressor()
        
    elif model == 'XGBRF':
        
        
        classifier_obj = XGBRFRegressor()

    score = cross_val_score(classifier_obj, X, y, n_jobs=2, cv=3, scoring='neg_mean_squared_error')
    mse = score.mean()
    
    return mse

def optimize_hyperparameters(model, X, y):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, model, X, y), n_trials=25)
    
    return study.best_params
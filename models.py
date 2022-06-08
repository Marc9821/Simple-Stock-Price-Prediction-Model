from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, LassoLarsCV, BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.svm import SVR, LinearSVC


def predict_regression(model_types, X_train, y_train, X_test):
    
    models = {'LR': LinearRegression(), 'RF': RandomForestRegressor(), 'SVM': SVR(), 'GBR': GradientBoostingRegressor(), 'ABR': AdaBoostRegressor(),
              'GPR': GaussianProcessRegressor(), 'RCV': RidgeCV(),'LLCV': LassoLarsCV(), 'BR': BayesianRidge(), 'KNN': KNeighborsRegressor(), 
              'LCV': LassoCV(), 'MLP': MLPRegressor(), 'LSVM': LinearSVC(), 'ET': ExtraTreeRegressor()}
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
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    return y_pred
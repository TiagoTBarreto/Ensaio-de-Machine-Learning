# 0.0 Funções

def load_dataset():
# Essa função carrega os dados.
    # treino
    X_training = pd.read_csv('../dataset/regression/X_training.csv')
    y_training = pd.read_csv('../dataset/regression/y_training.csv')
    
    # validacao
    X_validation = pd.read_csv('../dataset/regression/X_validation.csv')
    y_validation = pd.read_csv('../dataset/regression/y_val.csv')
    
    # teste
    X_test = pd.read_csv('../dataset/regression/X_test.csv')
    y_test = pd.read_csv('../dataset/regression/y_test.csv')

    return X_training, y_training, X_validation, y_validation, X_test, y_test

def metrics(true, pred):
# Essa função calcula as principais métricas dos algoritmos de regressão.
    r2 = mt.r2_score(true, pred)
    mse = mt.mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    mae = mt.mean_absolute_error(true, pred)
    mape = mt.mean_absolute_percentage_error(true, pred)

    return r2, mse, rmse, mae, mape

def fit_model_linear(model):
# Essa função treina os modelos lineares.
    # fit 
    model.fit(X_training, y_training)
    # performance
    yhat_train = model.predict(X_training)
    yhat_val = model.predict(X_validation)

    return yhat_train, yhat_val

def refit_model_linear(model):
# Essa função retreina os modelos lineares.
    # refit 
    model.fit(np.concatenate((X_training, X_validation)), np.concatenate((y_training, y_validation)))
    # performance
    yhat_test = model.predict(X_test)

    return yhat_test

def fit_model_poly(model, degree):
# Essa função treina os modelos polinomiais.
    # define
    poly = PolynomialFeatures(degree = degree)

    # transform
    X_train_poly = poly.fit_transform(X_training)
    X_val_poly = poly.transform(X_validation)
    
    # fit 
    model.fit(X_train_poly, y_training)
    
     # predict
    yhat_train = model.predict(X_train_poly) 
    yhat_val = model.predict(X_val_poly)

    return yhat_train, yhat_val

def refit_model_poly(model, degree):
# Essa função retreina os modelos polinomiais.
    # define
    poly = PolynomialFeatures(degree = degree)
    
    # transform
    X_train_poly = poly.fit_transform(X_training)
    X_val_poly = poly.transform(X_validation)
    
    # refit
    model.fit(np.concatenate((X_train_poly, X_val_poly)), np.concatenate((y_training, y_validation)))
    X_test_poly = poly.transform(X_test)
    
    # predict
    yhat_test = model.predict(X_test_poly)

    return yhat_test

def fit_tree(model):
# Essa função treina os modelos de árvore.
    # fit 
    model.fit(X_training, y_training)
    # performance
    yhat_train = model.predict(X_training)
    yhat_val = model.predict(X_validation)

    return yhat_train, yhat_val

def refit_tree(model):
# Essa função retreina os modelos de árvore.
    # refit 
    model.fit(np.concatenate((X_training, X_validation)), np.concatenate((y_training, y_validation)))
    # performance
    yhat_test = model.predict(X_test)

    return yhat_test

def dataframe(tipo_do_dado):
# Essa função gera o dataframe do ensaio. Podendo receber 4 variavéis:
# 1.0 'treino': Só mostra o dataset de treino
# 2.0 'validacao': Só mostra o dataset de validacao
# 3.0 'teste': Só mostra o dataset de teste
# 4.0 'total': Mostra todos os datasets
    data1 = [['Linear Regression', r2_lr_train, mse_lr_train, rmse_lr_train, mae_lr_train, mape_lr_train, 'Treino'], ['Lasso', r2_lasso_train, mse_lasso_train, rmse_lasso_train, mae_lasso_train, mape_lasso_train, 'Treino'],['Ridge', r2_ridge_train, mse_ridge_train, rmse_ridge_train, mae_ridge_train, mape_ridge_train,'Treino'],['Elastic Net',r2_en_train, mse_en_train, rmse_en_train, mae_en_train, mape_en_train,'Treino'],['Polynomial',r2_pr_train, mse_pr_train, rmse_pr_train, mae_pr_train, mape_pr_train,'Treino'],['Polynomial Lasso',r2_plasso_train, mse_plasso_train, rmse_plasso_train, mae_plasso_train, mape_plasso_train,'Treino'],['Polynomial Ridge',r2_pridge_train, mse_pridge_train, rmse_pridge_train, mae_pridge_train, mape_pridge_train,'Treino'],['Polynomial Elastic',r2_pen_train, mse_pen_train, rmse_pen_train, mae_pen_train, mape_pen_train,'Treino'],['Decision Tree Regressor',r2_dt_train, mse_dt_train, rmse_dt_train, mae_dt_train, mape_dt_train,'Treino'],['Random Forest Regressor',r2_rf_train, mse_rf_train, rmse_rf_train, mae_rf_train, mape_rf_train,'Treino']]
    
    df1 = pd.DataFrame(data1, columns=['Model', 'R2','MSE','RMSE','MAE','MAPE','Dataset'])
    
    data2 = [['Linear Regression', r2_lr_val, mse_lr_val, rmse_lr_val, mae_lr_val, mape_lr_val , 'Validação'], ['Lasso', r2_lasso_val, mse_lasso_val, rmse_lasso_val, mae_lasso_val, mape_lasso_val , 'Validação'],['Ridge', r2_ridge_val, mse_ridge_val, rmse_ridge_val, mae_ridge_val, mape_ridge_val ,'Validação'],['Elastic Net', r2_en_val, mse_en_val, rmse_en_val, mae_en_val, mape_en_val,'Validação'],['Polynomial', r2_pr_val, mse_pr_val, rmse_pr_val, mae_pr_val, mape_pr_val,'Validação'],['Polynomial Lasso', r2_plasso_val, mse_plasso_val, rmse_plasso_val, mae_plasso_val, mape_plasso_val,'Validação'],['Polynomial Ridge', r2_pridge_val, mse_pridge_val, rmse_pridge_val, mae_pridge_val, mape_pridge_val,'Validação'],['Polynomial Elastic', r2_pen_val, mse_pen_val, rmse_pen_val, mae_pen_val, mape_pen_val,'Validação'],['Decision Tree Regressor', r2_dt_val, mse_dt_val, rmse_dt_val, mae_dt_val, mape_dt_val,'Validação'],['Random Forest Regressor',r2_rf_val, mse_rf_val, rmse_rf_val, mae_rf_val, mape_rf_val,'Validação']]
    
    df2 = pd.DataFrame(data2, columns=['Model', 'R2','MSE','RMSE','MAE','MAPE','Dataset'])
    
    data3 = [['Linear Regression', r2_lr_test, mse_lr_test, rmse_lr_test, mae_lr_test, mape_lr_test , 'Teste'], ['Lasso', r2_lasso_test, mse_lasso_test, rmse_lasso_test, mae_lasso_test, mape_lasso_test , 'Teste'],['Ridge', r2_ridge_test, mse_ridge_test, rmse_ridge_test, mae_ridge_test, mape_ridge_test ,'Teste'],['Elastic Net', r2_en_test, mse_en_test, rmse_en_test, mae_en_test, mape_en_test,'Teste'],['Polynomial', r2_pr_test, mse_pr_test, rmse_pr_test, mae_pr_test, mape_pr_test,'Teste'],['Polynomial Lasso', r2_plasso_test, mse_plasso_test, rmse_plasso_test, mae_plasso_test, mape_plasso_test,'Teste'],['Polynomial Ridge', r2_pridge_test, mse_pridge_test, rmse_pridge_test, mae_pridge_test, mape_pridge_test,'Teste'],['Polynomial Elastic', r2_pen_test, mse_pen_test, rmse_pen_test, mae_pen_test, mape_pen_test,'Teste'],['Decision Tree Regressor', r2_dt_test, mse_dt_test, rmse_dt_test, mae_dt_test, mape_dt_test,'Teste'],['Random Forest Regressor',r2_rf_test, mse_rf_test, rmse_rf_test, mae_rf_test, mape_rf_test,'Teste']]
    
    df3 = pd.DataFrame(data3, columns=['Model', 'R2','MSE','RMSE','MAE','MAPE','Dataset'])
    
    df = pd.concat([df1, df2, df3], axis = 0).reset_index(drop = True)
    
    
    list_columns = ['R2', 'MSE', 'RMSE', 'MAE', 'MAPE']
    for c in list_columns:    
        df[c] = df[c].round(2)
    if tipo_do_dado == 'treino': 
        return df1
    elif tipo_do_dado == 'validacao':
        return df2
    elif tipo_do_dado == 'teste':
        return df3
    elif tipo_do_dado == 'total':
        return df

# 1.0 Libraries
import pandas as pd
import warnings
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics as mt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
from IPython.display import display

# Ignorar os warnings
warnings.filterwarnings("ignore")

# 2.0 Dataset
X_training, y_training, X_validation, y_validation, X_test, y_test = load_dataset()

# 3.0 Model Linear Regression

# 3.0.1 Normal

# define
linear = LinearRegression()
# fit
yhat_train, yhat_val = fit_model_linear(linear)

# performance
r2_lr_train, mse_lr_train, rmse_lr_train, mae_lr_train, mape_lr_train = metrics(y_training, yhat_train)
r2_lr_val, mse_lr_val, rmse_lr_val, mae_lr_val, mape_lr_val = metrics(y_validation, yhat_val)

# refit
yhat_test = refit_model_linear(linear)

# performance
r2_lr_test, mse_lr_test, rmse_lr_test, mae_lr_test, mape_lr_test = metrics(y_test, yhat_test)

# 3.0.2 Lasso

# define
lasso = Lasso()
# fit 
yhat_train, yhat_val = fit_model_linear(lasso)

r2_lasso_train, mse_lasso_train, rmse_lasso_train, mae_lasso_train, mape_lasso_train = metrics(y_training, yhat_train)
r2_lasso_val, mse_lasso_val, rmse_lasso_val, mae_lasso_val, mape_lasso_val = metrics(y_validation, yhat_val)

# refit
yhat_test = refit_model_linear(lasso)

# performance
r2_lasso_test, mse_lasso_test, rmse_lasso_test, mae_lasso_test, mape_lasso_test = metrics(y_test, yhat_test)

# 3.0.3 Ridge

# define
ridge = Ridge(alpha = 1)

# fit 
yhat_train, yhat_val = fit_model_linear(ridge)

r2_ridge_train, mse_ridge_train, rmse_ridge_train, mae_ridge_train, mape_ridge_train = metrics(y_training, yhat_train)
r2_ridge_val, mse_ridge_val, rmse_ridge_val, mae_ridge_val, mape_ridge_val = metrics(y_validation, yhat_val)

# refit
yhat_test = refit_model_linear(ridge)

# performance
r2_ridge_test, mse_ridge_test, rmse_ridge_test, mae_ridge_test, mape_ridge_test = metrics(y_test, yhat_test)

# 3.0.4 ElasticNet

# define
elastic_net = ElasticNet()

# fit 
yhat_train, yhat_val = fit_model_linear(elastic_net)

r2_en_train, mse_en_train, rmse_en_train, mae_en_train, mape_en_train = metrics(y_training, yhat_train)
r2_en_val, mse_en_val, rmse_en_val, mae_en_val, mape_en_val = metrics(y_validation, yhat_val)

# refit
yhat_test = refit_model_linear(elastic_net)

# performance
r2_en_test, mse_en_test, rmse_en_test, mae_en_test, mape_en_test = metrics(y_test, yhat_test)

# 3.1 Model Polinomial Regression

# 3.1.1 Normal

# define
linear = LinearRegression()
# fit
yhat_train, yhat_val = fit_model_poly(linear, 2)

# performance
r2_pr_train, mse_pr_train, rmse_pr_train, mae_pr_train, mape_pr_train = metrics(y_training, yhat_train)
r2_pr_val, mse_pr_val, rmse_pr_val, mae_pr_val, mape_pr_val = metrics(y_validation, yhat_val)

# refit
yhat_test = refit_model_poly(linear, 2)

# performance
r2_pr_test, mse_pr_test, rmse_pr_test, mae_pr_test, mape_pr_test = metrics(y_test, yhat_test)

# 3.1.2 Lasso

# define
lasso = Lasso()

# fit
yhat_train, yhat_val = fit_model_poly(lasso, 2)

# performance
r2_plasso_train, mse_plasso_train, rmse_plasso_train, mae_plasso_train, mape_plasso_train = metrics(y_training, yhat_train)
r2_plasso_val, mse_plasso_val, rmse_plasso_val, mae_plasso_val, mape_plasso_val = metrics(y_validation, yhat_val)

# refit
yhat_test = refit_model_poly(linear, 2)

# performance
r2_plasso_test, mse_plasso_test, rmse_plasso_test, mae_plasso_test, mape_plasso_test = metrics(y_test, yhat_test)

# 3.1.3 Ridge

# define
ridge = Ridge()

# fit
yhat_train, yhat_val = fit_model_poly(ridge, 2)

# performance
r2_pridge_train, mse_pridge_train, rmse_pridge_train, mae_pridge_train, mape_pridge_train = metrics(y_training, yhat_train)
r2_pridge_val, mse_pridge_val, rmse_pridge_val, mae_pridge_val, mape_pridge_val = metrics(y_validation, yhat_val)

# refit
yhat_test = refit_model_poly(ridge, 2)

# performance
r2_pridge_test, mse_pridge_test, rmse_pridge_test, mae_pridge_test, mape_pridge_test = metrics(y_test, yhat_test)

# 3.1.4 ElasticNet

# define
elastic_net = ElasticNet()

# fit
yhat_train, yhat_val = fit_model_poly(elastic_net, 2)

# performance
r2_pen_train, mse_pen_train, rmse_pen_train, mae_pen_train, mape_pen_train = metrics(y_training, yhat_train)
r2_pen_val, mse_pen_val, rmse_pen_val, mae_pen_val, mape_pen_val = metrics(y_validation, yhat_val)

# refit
yhat_test = refit_model_poly(elastic_net, 2)

# performance
r2_pen_test, mse_pen_test, rmse_pen_test, mae_pen_test, mape_pen_test = metrics(y_test, yhat_test)

# 3.2 Model Decision Tree Regressor

# define
model = DecisionTreeRegressor(max_depth = 35)
# fit
yhat_train, yhat_val = fit_tree(model)
# performance
r2_dt_train, mse_dt_train, rmse_dt_train, mae_dt_train, mape_dt_train = metrics(y_training, yhat_train)
r2_dt_val, mse_dt_val, rmse_dt_val, mae_dt_val, mape_dt_val = metrics(y_validation, yhat_val)

# refit
yhat_test = refit_tree(model)

# performance
r2_dt_test, mse_dt_test, rmse_dt_test, mae_dt_test, mape_dt_test = metrics(y_test, yhat_test)

# 3.3 Model Random Forest Regressor

# define
tree = RandomForestRegressor(max_depth = 29)
# fit
yhat_train, yhat_val = fit_tree(tree)
# performance
r2_rf_train, mse_rf_train, rmse_rf_train, mae_rf_train, mape_rf_train = metrics(y_training, yhat_train)
r2_rf_val, mse_rf_val, rmse_rf_val, mae_rf_val, mape_rf_val = metrics(y_validation, yhat_val)

# refit
yhat_test = refit_tree(tree)

# performance
r2_rf_test, mse_rf_test, rmse_rf_test, mae_rf_test, mape_rf_test = metrics(y_test, yhat_test)

# 4.0 DataFrame

result = dataframe('total')
display(result)
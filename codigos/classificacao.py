# 1.0 Funções

def load_dataset():
# Essa função carrega os dados e os transforma em uma array do numpy.
    # treino
    X_training = pd.read_csv('../dataset/classification/X_training.csv')
    y_training = pd.read_csv('../dataset/classification/y_training.csv')
    
    # validacao
    X_validation = pd.read_csv('../dataset/classification/X_validation.csv')
    y_validation = pd.read_csv('../dataset/classification/y_validation.csv')
    
    # teste
    X_test = pd.read_csv('../dataset/classification/X_test.csv')
    y_test = pd.read_csv('../dataset/classification/y_test.csv')

    X_training = np.array(X_training)
    y_training = np.array(y_training).ravel()

    X_validation = np.array(X_validation)
    y_validation = np.array(y_validation).ravel()
    
    X_test = np.array(X_test)
    y_test = np.array(y_test).ravel()

    return X_training, y_training, X_validation, y_validation, X_test, y_test

def metrics(true, pred):
# Essa função calcula as principais métricas de algoritmos de classificação
    acc = mt.accuracy_score(true, pred)
    precision = mt.precision_score(true, pred)
    recall = mt.recall_score(true, pred)
    f1_score = mt.f1_score(true, pred)

    return acc, precision, recall, f1_score

def dataframe():
# Essa função gera o dataframe comparando os resultados do ensaio.
    data = [['KNN', acc_knn_train, precision_knn_train, recall_knn_train, f1_score_knn_train, 'Treino'], ['Decision Tree', acc_dt_train, precision_dt_train, recall_dt_train, f1_score_dt_train, 'Treino'], ['Random Forest', acc_rf_train, precision_rf_train, recall_rf_train, f1_score_rf_train, 'Treino'], ['Logistic Regression', acc_lreg_train, precision_lreg_train, recall_lreg_train, f1_score_lreg_train, 'Treino'],['KNN', acc_knn_val, precision_knn_val, recall_knn_val, f1_score_knn_val, 'Validação'], ['Decision Tree', acc_dt_val, precision_dt_val, recall_dt_val, f1_score_dt_val, 'Validação'], ['Random Forest', acc_rf_val, precision_rf_val, recall_rf_val, f1_score_rf_val, 'Validação'], ['Logistic Regression', acc_lreg_val, precision_lreg_val, recall_lreg_val, f1_score_lreg_val, 'Validação'], ['KNN', acc_knn_test, precision_knn_test, recall_knn_test, f1_score_knn_test, 'Teste'], ['Decision Tree', acc_dt_test, precision_dt_test, recall_dt_test, f1_score_dt_test, 'Teste'], ['Random Forest', acc_rf_test, precision_rf_test, recall_rf_test, f1_score_rf_test, 'Teste'], ['Logistic Regression', acc_lreg_test, precision_lreg_test, recall_lreg_test, f1_score_lreg_test, 'Teste']]
    df = pd.DataFrame(data, columns=['Model', 'Accuracy','Precision','Recall','F1-Score','Dataset'])
    list_columns = ['Accuracy','Precision','Recall','F1-Score'] 
    for c in list_columns:    
            df[c] = df[c].round(2)
    return df
    
# import libraries

import pandas as pd
import numpy as np
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics as mt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from IPython.display import display

# Excluindo os avisos
warnings.filterwarnings("ignore")

# 2.0 Import Dataset
X_training, y_training, X_validation, y_validation, X_test, y_test = load_dataset()

# 3.0 Model KNN

# Treino
# define
model = KNeighborsClassifier(n_neighbors = 3)
# fit
model.fit(X_training, y_training)
# performance
yhat_train = model.predict(X_training)

acc_knn_train, precision_knn_train, recall_knn_train, f1_score_knn_train = metrics(y_training, yhat_train) 

# Validacao
# performance
yhat_val = model.predict(X_validation)

acc_knn_val, precision_knn_val, recall_knn_val, f1_score_knn_val = metrics(y_validation, yhat_val) 

# Teste
model.fit(np.concatenate((X_training, X_validation)), np.concatenate((y_training, y_validation)))

yhat_test = model.predict(X_test)

acc_knn_test, precision_knn_test, recall_knn_test, f1_score_knn_test = metrics(y_test, yhat_test) 

# 3.1 Model Decision Tree

# Treino
tree = DecisionTreeClassifier(max_depth = 13)
# fit
tree.fit(X_training, y_training)   
# performance
yhat_train = tree.predict(X_training)

acc_dt_train, precision_dt_train, recall_dt_train, f1_score_dt_train = metrics(y_training, yhat_train) 

# Validação
# performance
yhat_val = tree.predict(X_validation)

acc_dt_val, precision_dt_val, recall_dt_val, f1_score_dt_val = metrics(y_validation, yhat_val) 

# Treino
# fit
tree.fit(np.concatenate((X_training, X_validation)), np.concatenate((y_training, y_validation)))
    
# performance
yhat_test = tree.predict(X_test)

acc_dt_test, precision_dt_test, recall_dt_test, f1_score_dt_test = metrics(y_test, yhat_test) 

# 3.2 Model Random Forest

# Treino
tree = RandomForestClassifier(n_estimators = 100, max_depth = 20)
# fit
tree.fit(X_training, y_training)
    
# performance
yhat_train = tree.predict(X_training)

acc_rf_train, precision_rf_train, recall_rf_train, f1_score_rf_train = metrics(y_training, yhat_train) 

# Validação
# performance
yhat_val = tree.predict(X_validation)

acc_rf_val, precision_rf_val, recall_rf_val, f1_score_rf_val = metrics(y_validation, yhat_val)

# Teste
# fit
tree.fit(np.concatenate((X_training, X_validation)), np.concatenate((y_training, y_validation)))  
# performance
yhat_test = tree.predict(X_test)

acc_rf_test, precision_rf_test, recall_rf_test, f1_score_rf_test = metrics(y_test, yhat_test) 

# 3.3 Model Logistic Regression

# Treino
# define
logistic = LogisticRegression(solver= 'newton-cg', C= 1)
# fit
logistic.fit(X_training, y_training)
    
# performance
yhat_train = logistic.predict(X_training)

acc_lreg_train, precision_lreg_train, recall_lreg_train, f1_score_lreg_train = metrics(y_training, yhat_train) 

# Validação
# performance
yhat_val = logistic.predict(X_validation)

acc_lreg_val, precision_lreg_val, recall_lreg_val, f1_score_lreg_val = metrics(y_validation, yhat_val)

# Teste
# fit
logistic.fit(np.concatenate((X_training, X_validation)), np.concatenate((y_training, y_validation)))
    
# performance
yhat_test = logistic.predict(X_test)

acc_lreg_test, precision_lreg_test, recall_lreg_test, f1_score_lreg_test = metrics(y_test, yhat_test) 

# 4.0 DataFrame
df = dataframe()
display(df)
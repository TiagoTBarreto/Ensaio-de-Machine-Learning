# 0.0 Funções

def fit_predict(model):
# Essa função faz com o que o modelo treine e faça a predição.
    model.fit( X )
    labels = model.predict( X )
        # Metric
    ss = mt.silhouette_score( X, labels )
    n_clusters = len(np.unique(labels))

    return ss, n_clusters

def dataframe():
# Essa função mostra o dataframe
    data = [['KMeans', n_clusters, ss], ['Affinity Propagation', n_clusters_ap, ss_ap]]
    df = pd.DataFrame(data, columns=['Model','Número de Clusters', 'Silhouette Score'])
    return df

# 1.0 Libraries

import pandas as pd
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn import metrics as mt
from matplotlib import pyplot as plt
from IPython.display import display

# 2.0 Dataset
X = pd.read_csv('../dataset/cluster/X_dataset.csv')

# 3.0 Model Affinity Propagation
model = AffinityPropagation( preference= -47 )
ss_ap, n_clusters_ap = fit_predict(model)

# 3.1 Model KMeans
model = KMeans(n_clusters = 3, n_init = 10)
ss, n_clusters = fit_predict(model)

# 4.0 DataFrame
df = dataframe()
display(df)
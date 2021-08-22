import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',1000)

df = pd.read_csv('in/df.csv', encoding='utf-8')

scaler = StandardScaler()
x_sc = scaler.fit_transform(df.drop(columns = ['Персона']))

linked = linkage(x_sc, method = 'ward')
plt.figure(figsize=(10, 15))
dendrogram(linked, orientation='top')
plt.show()

km = KMeans(n_clusters = 2) # задаём число кластеров, равное 2
labels = km.fit_predict(x_sc) # применяем алгоритм к данным и формируем вектор кластеров
df['cluster'] = labels

df.to_csv('out/df_after_classification.csv')



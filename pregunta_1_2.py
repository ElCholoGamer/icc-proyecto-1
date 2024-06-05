import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


data = pd.read_csv('1_1_smogon_agrupados.csv')

data.drop(columns='Cluster', inplace=True)

print('Datos sin cluster:')
print(data)
print()


data.drop(columns=data.columns[0], inplace=True)

print('Datos sin doble índice:')
print(data)
print()

PCA_COMPONENTS = 18

pca = PCA(PCA_COMPONENTS)
x_pca = pca.fit_transform(data)

print(f'Tamaño del DataFrame original: {
      len(data.axes[0])} filas x {len(data.axes[1])} columnas')
print(f'Tamaño de la matriz PCA: {len(x_pca)} filas x {
      len(x_pca[0])} columnas')


columns = [f'PCA{i}' for i in range(1, PCA_COMPONENTS + 1)]
pca_mat = pd.DataFrame(data=x_pca, columns=columns)

print('Matriz PCA:')
print(pca_mat)
print()


km = KMeans(18)
clusters_list = km.fit_predict(pca_mat)

pca_mat['Cluster'] = clusters_list
pca_mat.to_csv('1_2_smogon_agrupados_pca.csv')

smogon_data = pd.read_csv('datos/smogon.csv')
pca_mat['Pokemon'] = smogon_data['Pokemon']
pca_mat.to_csv('1_2_smogon_agrupados_pca_friendly.csv')

print('CSVs generados')

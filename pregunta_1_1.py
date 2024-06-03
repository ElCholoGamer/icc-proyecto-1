import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def process_move_set(move_set: str) -> str:
    move_set = move_set.replace('.', ' ')

    header_str_index = move_set.find('Moves')
    if header_str_index != -1:
        move_set = move_set[header_str_index + 5:]

    return move_set


data = pd.read_csv('datos/smogon.csv')
data['moves'] = data['moves'].apply(process_move_set)


vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
tfidf_mat = vec.fit_transform(data['moves'])

print('Número total de tokens:', len(vec.vocabulary_))
print('Vocabulario:')
print(sorted(vec.vocabulary_))
print()

headings = sorted(vec.vocabulary_)
freq_table = pd.DataFrame(data=tfidf_mat.toarray(), columns=headings)

print('Matriz TF-IDF:')
print(freq_table)
print()

km = KMeans(n_clusters=18, n_init=40)
clusters_list = km.fit_predict(freq_table)

freq_table['Cluster'] = clusters_list
freq_table.to_csv('1_1_smogon_agrupados.csv')

data['Cluster'] = clusters_list
data.to_csv('1_1_smogon_agrupados_friendly.csv')

print('CSVs generados')

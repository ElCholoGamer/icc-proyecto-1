import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def process_move_set(move_set: str) -> str:
    move_set = move_set.replace('.', ' ')

    header_str_index = move_set.find('Moves')
    if header_str_index != -1:
        move_set = move_set[header_str_index + 5:]

    return move_set


data = pd.read_csv('smogon.csv')
data['moves'] = data['moves'].apply(process_move_set)


vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
tfidf_x = vec.fit_transform(data['moves'])

print('Número total de tokens:', len(vec.vocabulary_))
print('Vocabulario:')
print(sorted(vec.vocabulary_))
print()

headings = sorted(vec.vocabulary_)
tfidf_mat = pd.DataFrame(data=tfidf_x.toarray(), columns=headings)

print('Matriz TF-IDF:')
print(tfidf_mat)
print()

km = KMeans(n_clusters=18, n_init=100)
clusters_list = km.fit_predict(tfidf_mat)

tfidf_mat['Cluster'] = clusters_list
tfidf_mat.to_csv('1_1_smogon_agrupados.csv')

data['Cluster'] = clusters_list
data.to_csv('1_1_smogon_agrupados_friendly.csv')

print('CSVs generados')
print()

print('Matriz TF-IDF con los clusters:')
print(tfidf_mat)

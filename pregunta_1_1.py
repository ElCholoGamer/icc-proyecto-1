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
x_tfidf = vec.fit_transform(data['moves'])

print('Número total de tokens:', len(vec.vocabulary_))
print('Vocabulario:')
print(sorted(vec.vocabulary_))
print()

headings = sorted(vec.vocabulary_)
tfidf_df = pd.DataFrame(data=x_tfidf.toarray(), columns=headings)

print('Matriz TF-IDF:')
print(x_tfidf.toarray())
print()

km = KMeans(n_clusters=18, n_init=100)
clusters_list = km.fit_predict(tfidf_df)

tfidf_df['Cluster'] = clusters_list
tfidf_df.to_csv('1_1_smogon_agrupados.csv')

data['Cluster'] = clusters_list
data.to_csv('1_1_smogon_agrupados_friendly.csv')

print('CSVs generados')
print()

print('DataFrame TF-IDF con los clusters (guardado en 1_1_smogon_agrupados.csv):')
print(tfidf_df)

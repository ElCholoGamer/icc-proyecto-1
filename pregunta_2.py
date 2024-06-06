import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

POKEMON_KEYWORDS = [
    'bug',
    'dark',
    'dragon',
    'electric',
    'fairy',
    'fighting',
    'fire',
    'flying',
    'ghost',
    'grass',
    'ground',
    'ice',
    'normal',
    'poison',
    'psychic',
    'rock',
    'steel',
    'water'
]


def keep_keywords(s: str) -> str:
    s = s.lower()

    result = []
    i = 0

    while i < len(s):
        for keyword in POKEMON_KEYWORDS:
            if s[i:i+len(keyword)] == keyword:
                result.append(keyword)
                i += len(keyword) - 1
                break

        i += 1

    return ' '.join(result)


data = pd.read_csv('smogon.csv')

data['moves'] = data['moves'].apply(keep_keywords)

vec = TfidfVectorizer(ngram_range=(1, 1))
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

km = KMeans(n_clusters=18, n_init=100)
clusters_list = km.fit_predict(freq_table)

freq_table['Cluster'] = clusters_list
freq_table.to_csv('2_smogon_agrupados_keywords.csv')

data['Cluster'] = clusters_list
data.to_csv('2_smogon_agrupados_keywords_friendly.csv')

print('CSVs generados')

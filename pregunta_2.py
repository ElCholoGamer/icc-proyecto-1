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

    result = ''
    done = False
    start = 0

    while not done:
        done = True

        for keyword in POKEMON_KEYWORDS:
            next = s.find(keyword, start)

            if next != -1:
                result += keyword + ' '
                start = next + len(keyword)
                done = False

    return result.strip()


datos = pd.read_csv('datos/smogon.csv')

datos['moves'] = datos['moves'].apply(keep_keywords)


# Generar la matriz TF-IDF
# Usa los stop_words en inglés provistos por sklearn
vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
tfidf_mat = vec.fit_transform(datos['moves'])

# Mostrar el número total de tokens e imprimirlos
print('Número total de tokens:', len(vec.vocabulary_))
print('Vocabulario:')
print(sorted(vec.vocabulary_))
print()

# Generar un DataFrame con la matriz TF-IDF con el vocabulario como cabecera
headings = sorted(vec.vocabulary_)
freq_table = pd.DataFrame(data=tfidf_mat.toarray(), columns=headings)

print('Matriz TF-IDF:')
print(freq_table)
print()

# Agrupar las filas del DataFrame en base a sus puntuaciones TF-IDF
km = KMeans(n_clusters=18, n_init=40)
clusters_list = km.fit_predict(freq_table)

# Generar un archivo CSV con la matriz TF-IDF y los clusters
freq_table['Cluster'] = clusters_list
freq_table.to_csv('csv/2_smogon_agrupados.csv')

# CSV human-friendly
datos['Cluster'] = clusters_list
datos.to_csv('csv/2_smogon_agrupados_friendly.csv')

print('CSVs generados')

import nltk
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import FreqDist
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer


def run():
    corpus, corpus_lematizado = obtener_corpus('CorpusLenguajes.txt')
    print("\nCorpus:", corpus)

    matriz, vocabulario = aplicar_tfidf(corpus)
    print("\nMatriz TF-IDF:", matriz.toarray())
    print("\nVocabulario: ", vocabulario)

    frecuencia = obtener_frecuencia(corpus_lematizado)
    mostrar_6_palabras_mas_frecuentes(frecuencia)
    mostrar_palabra_menos_frecuente(frecuencia)
    mostrar_palabra_mas_repetida_por_cada_oracion(corpus_lematizado)
    graficar_distancia_de_frecuencia(frecuencia)


def obtener_corpus(file_name):
    contexto = globals().copy()
    with open(file_name, 'r', encoding='utf-8') as f:
        codigo = f.read()
    exec(codigo, contexto)
    corpus_lematizado = contexto['corpus']
    corpus = [" ".join(sublist) for sublist in corpus_lematizado]
    return corpus, corpus_lematizado


def quitarStopwords_eng(text):
    eng = stopwords.words("english")
    clean_text = [w.lower() for w in text if w.lower() not in eng
                  and w not in string.punctuation
                  and w not in ["'s", '|', '--', "''", "``", '.-']]
    return clean_text


def lematizar(text):
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in text]
    return lemmatized_text


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def aplicar_tfidf(corpus):
    vectorizer = TfidfVectorizer()
    matriz = vectorizer.fit_transform(corpus)  
    vocabulario = vectorizer.vocabulary_
    return matriz, vocabulario


def obtener_frecuencia(corpus_lematizado):
    corpus_tokenizado = [item for sublist in corpus_lematizado for item in sublist]
    frecuencia = FreqDist(corpus_tokenizado)
    return frecuencia


def mostrar_6_palabras_mas_frecuentes(frecuencia):
    top6 = frecuencia.most_common(6)
    print("\nJerarquía de 6 palabras más usadas:")
    for palabra, cantidad in top6:
        print(f"{palabra}: {cantidad} apariciones")


def mostrar_palabra_menos_frecuente(frecuencia):
    menos_usada = frecuencia.most_common()[-1]
    print(f"\nLa palabra menos utilizada es '{menos_usada[0]}' con {menos_usada[1]} aparición/es.")


def mostrar_palabra_mas_repetida_por_cada_oracion(corpus_lematizado):
    print("\nPalabra mas repetida por cada oración:")
    for oracion in corpus_lematizado:
        freq = FreqDist(oracion)
        palabra, repeticiones = freq.most_common(1)[0]  # palabra más repetida en esa oración
        print(f"'{palabra}' se repitió {repeticiones} veces en la oración: '{" ".join(oracion)}'")
    

def graficar_distancia_de_frecuencia(frecuencia):
    frecuencia.plot(20, show=True)


if __name__ == '__main__':
    run()
from __future__ import division
import nltk
from nltk.corpus import stopwords
import string
import numpy
from collections import Counter
import math

stopwords = stopwords.words('english')
punct = string.punctuation


def compute_frequency_scores(docs):

    def term_freq(document):
        freq = Counter(document)
        word_freq = []
        for word in unique_words:
            word_freq.append(freq[word]/len(document))
        return word_freq

    def inverse_doc_freq(word):
        matches = len(list(True for text in text_collection._texts if word in text))

        if matches > 0:
            idf = 1.0 + math.log(float(len(text_collection._texts)) / matches)
        else:
            idf = 1.0

        return idf

    def tf_idf(document):
        freq = Counter(document)
        word_tfidf = []
        for word in unique_words:
            tfidf = (freq[word] / len(document)) * inverse_doc_freq(word)
            word_tfidf.append(tfidf)
        return word_tfidf

    texts = []
    frequencies = dict()
    for doc in docs:
        f = open(doc).read()
        words = nltk.word_tokenize(f)
        words = [word.lower() for word in words if word not in stopwords and word not in punct]
        text = nltk.Text(words)
        texts.append(text)

    text_collection = nltk.TextCollection(texts)
    unique_words = list(set(text_collection))

    tfidf_vectors = [numpy.array(tf_idf(text)) for text in texts]
    frequencies['tf_idf'] = []
    frequencies['tf_idf'] = tfidf_vectors

    freq_vectors = [numpy.array(term_freq(text)) for text in texts]
    frequencies['term_freq'] = []
    frequencies['term_freq'] = freq_vectors

    return frequencies


def cos_sim(v1, v2):
    return numpy.dot(v1, v2) / (numpy.sqrt(numpy.dot(v1, v1)) * numpy.sqrt(numpy.dot(v2, v2)))


def main():
    docs = ['JobDesc.txt', 'JobDesc.txt']
    frequencies = compute_frequency_scores(docs)
    tfidf_vectors = frequencies['tf_idf']
    print tfidf_vectors
    cosine_similarity =  cos_sim(tfidf_vectors[0], tfidf_vectors[1])
    print cosine_similarity

    freq_vectors = frequencies['term_freq']
    #print cos_sim(freq_vectors[0], freq_vectors[1])


if __name__ == '__main__':


    main()
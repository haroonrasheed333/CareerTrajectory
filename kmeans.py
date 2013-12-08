from sklearn.feature_extraction.text import TfidfVectorizer
from careertrajectory import ResumeCorpus
from sklearn.cluster import KMeans
import logging
from optparse import OptionParser
import sys
from time import time

traintest_corpus = ResumeCorpus('/Users/divyakarthikeyan/Desktop/COURSES/Fall 2013/NLP/Project/testing/samples_text')
docs=[]
filename = {}
i= 0
for resume in traintest_corpus.resumes[:3]:
    docs.append(resume[0])
    filename[i] = resume[2]
    i += 1

#docs =["space star sky spaceship", "math cosine similarity", "space astronaut meteor", "add subtract math calculate"]

vectorizer = TfidfVectorizer(min_df=1)
tfidf = vectorizer.fit_transform(docs)
km = KMeans(n_clusters=2, init='k-means++', max_iter=100, n_init=1)
km.fit(tfidf)
for i in range (0,len(tfidf.toarray())):
    print str(filename[i]) + str(km.predict(tfidf.toarray()[i]))
    i +=1
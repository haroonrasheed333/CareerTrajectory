import nltk
from nltk.tokenize import word_tokenize
import lxml
import pprint, random, pickle
import itertools
import os, sys, re
import random
from pprint import pprint as pp
from itertools import islice
from nltk import FreqDist
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize
from nltk.classify import apply_features
from nltk.corpus import brown, names, movie_reviews 
from nltk.corpus import stopwords as sw
from nltk.collocations import  BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import       BigramAssocMeasures,     TrigramAssocMeasures


#!/usr/bin/python

#punctuation = re.compile(r'[<>+&.?!\]\[,":;()\/\\|0-9]')
punctuation = re.compile(r'[^A-Za-z0-9\' ]+')
stopwords   = sw.words('english')
lemma = nltk.WordNetLemmatizer()


class ResumeCorpus():
    def __init__(self, source_dir):
        """
        init with path do the directory with the .txt files
        """
        
        self.source_dir = source_dir
        self.files = self.getFiles(self.source_dir)
        self.sents = self.readFiles(self.files, self.source_dir)
        
    def getFiles(self, source_dir):
        file = [ f for (dirpath, dirnames, filenames) in os.walk(source_dir) for f in filenames if f[6:] == 'labels' ]
        return file
        
    def readFiles(self, file, source_dir):
        resumes = []
        for line in open(file).read():
            filename,tag = line.split('\t')
            resumes.append((filename,tag))
        return resumes
                      
def trainClassifier(training_featureset):
    revSent_classifier = nltk.NaiveBayesClassifier.train(training_featureset)
    #remove stop words
    return revSent_classifier


def feature_consolidation(documents, fd_words, addTrueScore=False):
    if addTrueScore:
        con_features = [ (unigram_features(documents),tag) for (file, tag) in documents]
    else:   
        con_features = unigram_features(documents)
    return con_features

    

def unigram_features(document, lemma_fd_list, features):
    docu=re.sub('[^A-Za-z\' ]+', '', document)
    tokens = nltk.word_tokenize(docu)
    for word in tokens:
        if word in lemma_fd_list:
            features[word] = True
    return features



if __name__ == "__main__":
    traintest_corpus = ResumeCorpus('/Users/divyakarthikeyan/Downloads/data') #4256
    random.shuffle(traintest_corpus)
    train_resumes=traintest_corpus[:2000]
    words = []
    for resume in train_resumes:
        words = words + resume[0].split() 
    fd = FreqDist(words)
    fd_words = [ word for word in fd.keys()[:150] if word not in stopwords ]
    test_resumes=traintest_corpus.resumes[:-500]
    train_featureset  = feature_consolidation(train_resumes, fd_words, True)
    review_classifier = trainClassifier(train_featureset)  
    outputfile = open ('/Users/divyakarthikeyan/Downloads/data','w')
    for document in test_resumes:   
        resume_features = feature_consolidation(document[0], fd_words)
        (fileName,tag) = document
        outputfile.write('%s' %fileName +'\t' + '%s' %str(tag) + '\n')
    
    test_featureset  = feature_consolidation(test_resumes, fd_words, True)
    print nltk.classify.accuracy(review_classifier, test_featureset)
    review_classifier.show_most_informative_features(50)


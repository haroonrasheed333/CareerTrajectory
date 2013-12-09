from __future__ import division
import nltk
import os, sys, re
import random
import string
from collections import Counter
import math
from nltk import FreqDist
from nltk.collocations import  BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import       BigramAssocMeasures,     TrigramAssocMeasures
import itertools
import nltk.classify.svm
import scipy.sparse
import numpy
from nltk.corpus import stopwords as sw
import progressbar
import json

with open('skills_new.json') as data:
    skills_json = json.load(data)

skill_features = []
for skill in skills_json:
    skill_list = skills_json[skill]
    skill_count = Counter(skill_list)
    top_skills = sorted(skill_count, key=skill_count.get, reverse=True)[:40]
    skill_features += top_skills

top_skills = list(set(skill_features))


def pbar(size):
    bar = progressbar.ProgressBar(maxval=size,
                                  widgets=[progressbar.Bar('=', '[', ']'),
                                           ' ', progressbar.Percentage(),
                                           ' ', progressbar.ETA(),
                                           ' ', progressbar.Counter(),
                                           '/%s' % size])
    return bar

punct = string.punctuation

#!/usr/bin/python

#punctuation = re.compile(r'[<>+&.?!\]\[,":;()\/\\|0-9]')
punctuation = re.compile(r'[^A-Za-z0-9\' ]+')
stopwords   = ["a's","accordingly","again","allows","also","amongst","anybody","anyways","appropriate","aside","available","because","before","below","between","by","can't","certain","com","consider","corresponding","definitely","different","don't","each","else","et","everybody","exactly","fifth","follows","four","gets","goes","greetings","has","he","her","herein","him","how","i'm","immediate","indicate","instead","it","itself","know","later","lest","likely","ltd","me","more","must","nd","needs","next","none","nothing","of","okay","ones","others","ourselves","own","placed","probably","rather","regarding","right","saying","seeing","seen","serious","she","so","something","soon","still","t's","th","that","theirs","there","therein","they'd","third","though","thus","toward","try","under","unto","used","value","vs","way","we've","weren't","whence","whereas","whether","who's","why","within","wouldn't","you'll","yourself","able","across","against","almost","although","an","anyhow","anywhere","are","ask","away","become","beforehand","beside","beyond","c'mon","cannot","certainly","come","considering","could","described","do","done","edu","elsewhere","etc","everyone","example","first","for","from","getting","going","had","hasn't","he's","here","hereupon","himself","howbeit","i've","in","indicated","into","it'd","just","known","latter","let","little","mainly","mean","moreover","my","near","neither","nine","noone","novel","off","old","only","otherwise","out","particular","please","provides","rd","regardless","said","says","seem","self","seriously","should","some","sometime","sorry","sub","take","than","that's","them","there's","theres","they'll","this","three","to","towards","trying","unfortunately","up","useful","various","want","we","welcome","what","whenever","whereby","which","whoever","will","without","yes","you're","yourselves","about","actually","ain't","alone","always","and","anyone","apart","aren't","asking","awfully","becomes","behind","besides","both","c's","cant","changes","comes","contain","couldn't","despite","does","down","eg","enough","even","everything","except","five","former","further","given","gone","hadn't","have","hello","here's","hers","his","however","ie","inasmuch","indicates","inward","it'll","keep","knows","latterly","let's","look","many","meanwhile","most","myself","nearly","never","no","nor","now","often","on","onto","ought","outside","particularly","plus","que","re","regards","same","second","seemed","selves","seven","shouldn't","somebody","sometimes","specified","such","taken","thank","thats","themselves","thereafter","thereupon","they're","thorough","through","together","tried","twice","unless","upon","uses","very","wants","we'd","well","what's","where","wherein","while","whole","willing","won't","yet","you've","zero","above","after","all","along","am","another","anything","appear","around","associated","be","becoming","being","best","brief","came","cause","clearly","concerning","containing","course","did","doesn't","downwards","eight","entirely","ever","everywhere","far","followed","formerly","furthermore","gives","got","happens","haven't","help","hereafter","herself","hither","i'd","if","inc","inner","is","it's","keeps","last","least","like","looking","may","merely","mostly","name","necessary","nevertheless","nobody","normally","nowhere","oh","once","or","our","over","per","possible","quite","really","relatively","saw","secondly","seeming","sensible","several","since","somehow","somewhat","specify","sup","tell","thanks","the","then","thereby","these","they've","thoroughly","throughout","too","tries","two","unlikely","us","using","via","was","we'll","went","whatever","where's","whereupon","whither","whom","wish","wonder","you","your","according","afterwards","allow","already","among","any","anyway","appreciate","as","at","became","been","believe","better","but","can","causes","co","consequently","contains","currently","didn't","doing","during","either","especially","every","ex","few","following","forth","get","go","gotten","hardly","having","hence","hereby","hi","hopefully","i'll","ignored","indeed","insofar","isn't","its","kept","lately","less","liked","looks","maybe","might","much","namely","need","new","non","not","obviously","ok","one","other","ours","overall","perhaps","presumably","qv","reasonably","respectively","say","see","seems","sent","shall","six","someone","somewhere","specifying","sure","tends","thanx","their","thence","therefore","they","think","those","thru","took","truly","un","until","use","usually","viz","wasn't","we're","were","when","whereafter","wherever","who","whose","with","would","you'd","yours"]
lemma = nltk.WordNetLemmatizer()
porter = nltk.PorterStemmer()


class ResumeCorpus():
    def __init__(self, source_dir, labels_file=None):
        """
        init with path do the directory with the .txt files
        """
        
        self.source_dir = source_dir
        #self.files = self.getFiles(self.source_dir)
        user_name = os.environ.get('USER')
        if not labels_file:
            self.labels_file = '/Users/' + user_name + '/Documents/Data/labels.txt'
        else:
            self.labels_file = labels_file
        self.resumes = self.readFiles(self.labels_file, self.source_dir)
        
    def getFiles(self, source_dir):
        file = [ f for (dirpath, dirnames, filenames) in os.walk(source_dir) for f in filenames if f[6:] == 'labels' ]
        return file
        
    def readFiles(self, file, source_dir):
        resumes = []
        for line in open(file).readlines():
            try:
                ftag = line.split('\t')
                #filename = ftag[1].rstrip() + '/' + ftag[0]
                filename = ftag[0]
                tag = ftag[1]
                resumes.append((open(source_dir + '/' + filename).read(),tag,filename))
            except:
                pass
        return resumes


def trainClassifier(training_featureset):
    revSent_classifier = nltk.NaiveBayesClassifier.train(training_featureset)
    #remove stop words
    return revSent_classifier

def trainSVMClassifier(training_featureset):
    revSent_classifier = nltk.SvmClassifier.train(training_featureset)
    #remove stop words
    return revSent_classifier


def feature_consolidation(documents, fd_words, addTrueScore=False):
    if addTrueScore:
        con_features = [(unigram_features(file, fd_words),tag) for (file, tag, filename) in documents]
    else:   
        con_features = unigram_features(documents,fd_words)
    return con_features

    

def unigram_features(document, lemma_words_list):
    docu=re.sub('[^A-Za-z\' ]+', '', str(document))
    tokens = nltk.word_tokenize(docu)
    features = {}
    avg_word_len= 0
    count = 0
    for word in tokens:
        word_stem = str(porter.stem(word))
        if word_stem in lemma_words_list:
            features[word_stem] = True
            avg_word_len += len(word_stem)
            count += 1
    features['average_word_length'] = avg_word_len/(count+1)
    features['docu_length'] = len(tokens)
    return features



if __name__ == "__main__":
    user_name = os.environ.get('USER')
    traintest_corpus = ResumeCorpus('/Users/' + user_name + '/Documents/Data/samples_text')
    random.shuffle(traintest_corpus.resumes)
    train_resumes = traintest_corpus.resumes[0:1500]
    test_resumes = traintest_corpus.resumes[1501:]
    words = []
    for resume in train_resumes:
        words = words + resume[0].split()
    fd = FreqDist(words)
    fd_words = [ porter.stem(word) for word in fd.keys()[:200] if word not in stopwords ]
    fd_words = list(set(fd_words + top_skills))
    train_featureset  = feature_consolidation(train_resumes, fd_words, True)
    review_classifier = trainClassifier(train_featureset)
    outputfile = open ('classifier_output.txt','w')
    for document in test_resumes:
        resume_features = feature_consolidation(document[0], fd_words)
        (text,tag,fileName) = document
        classifier_output = review_classifier.classify(resume_features)
        outputfile.write('%s' %fileName + '\t' + '%s' %str(tag) + '%s' %classifier_output + '\n')


    outputfile.close()
    test_featureset  = feature_consolidation(test_resumes,fd_words, True)
    accuracy = nltk.classify.accuracy(review_classifier, test_featureset)
    #review_classifier.show_most_informative_features(50)

    print accuracy

    review_svm_classifier = trainSVMClassifier(train_featureset)
    accuracy_svm = nltk.classify.accuracy(review_svm_classifier, test_featureset)

    print accuracy_svm
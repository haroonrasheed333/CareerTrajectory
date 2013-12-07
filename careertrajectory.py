import nltk
import os, sys, re
import random
import string
from nltk import FreqDist
from nltk.corpus import stopwords as sw
import progressbar


def pbar(size):
    bar = progressbar.ProgressBar(maxval=size,
                                  widgets=[progressbar.Bar('=', '[', ']'),
                                           ' ', progressbar.Percentage(),
                                           ' ', progressbar.ETA(),
                                           ' ', progressbar.Counter(),
                                           '/%s' % size])
    return bar

punct = string.punctuation

#punctuation = re.compile(r'[<>+&.?!\]\[,":;()\/\\|0-9]')
punctuation = re.compile(r'[^A-Za-z0-9\' ]+')
stopwords = sw.words('english')
lemma = nltk.WordNetLemmatizer()


class ResumeCorpus():
    def __init__(self, source_dir):
        """
        init with path do the directory with the .txt files
        """
        
        self.source_dir = source_dir
        #self.files = self.getFiles(self.source_dir)
        self.labels_file = 'labels.txt'
        self.resumes = self.readFiles(self.labels_file, self.source_dir)
        
    def getFiles(self, source_dir):
        file = [ f for (dirpath, dirnames, filenames) in os.walk(source_dir) for f in filenames if f[6:] == 'labels' ]
        return file
        
    def readFiles(self, file, source_dir):
        resumes = []
        for line in open(file).readlines():
            try:
                ftag = line.split('\t')
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


def feature_consolidation(documents, fd_words, addTrueScore=False):
    if addTrueScore:
        con_features = [ (unigram_features(file, fd_words),tag) for (file, tag, filename) in documents]
    else:   
        con_features = unigram_features(documents, fd_words)
    return con_features

    

def unigram_features(document, lemma_fd_list):
    docu=re.sub('[^A-Za-z\' ]+', '', document)
    tokens = nltk.word_tokenize(docu)
    features = {}
    for word in tokens:
        if word in lemma_fd_list:
            features[word] = True
    return features



if __name__ == "__main__":
    user_name = os.environ.get('USER')
    traintest_corpus = ResumeCorpus('/Users/' + user_name + '/Documents/Data/samples_text')

    kfold = 10
    i, bar = 0, pbar(10)
    bar.start()
    partition_size = int(len(traintest_corpus.resumes) / kfold)

    accuracies = []
    while i < kfold:
        train_resumes = traintest_corpus.resumes[0:i*partition_size] + traintest_corpus.resumes[(i+1)*partition_size:]
        test_resumes = traintest_corpus.resumes[i*partition_size:(i+1)*partition_size]

        #random.shuffle(traintest_corpus.resumes)
        #train_resumes=traintest_corpus.resumes[0:1200]

        words = []
        for resume in train_resumes:
            words = words + resume[0].split()
        fd = FreqDist(words)
        fd_words = [word for word in fd.keys()[:200] if word not in stopwords and word not in punct]

        #test_resumes=traintest_corpus.resumes[1201:]
        train_featureset  = feature_consolidation(train_resumes, fd_words, True)
        review_classifier = trainClassifier(train_featureset)

        #outputfile = open ('classifier_output.txt','w')
        #for document in test_resumes:
        #    resume_features = feature_consolidation(document[0], fd_words)
        #    (text,tag,fileName) = document
        #    classifier_output = review_classifier.classify(resume_features)
        #    outputfile.write('%s' %fileName + '\t' + '%s' %str(tag) + '%s' %classifier_output + '\n')


        #outputfile.close()
        test_featureset  = feature_consolidation(test_resumes, fd_words, True)
        accuracies.append(nltk.classify.accuracy(review_classifier, test_featureset))
        #review_classifier.show_most_informative_features(50)
        i += 1
        bar.update(i)

    bar.finish()
    print sum(accuracies) / len(accuracies)


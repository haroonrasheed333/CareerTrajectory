import codecs 
import json 
import string 
import numpy as np 
from collections import defaultdict 
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import precision_score, recall_score, classification_report 
from careertrajectory import ResumeCorpus
  
  
st = PorterStemmer() 
stopwords = stopwords.words('english') 
stopwords = list(set(stopwords)) 
data_dict = defaultdict(list) 
  
  
def PreProcessing(line): 
    unigrams = line.split() 
    word_list = [word.lower() for word in unigrams if word.lower() not in stopwords] 
    st = PorterStemmer() 
    word_list = [st.stem(word) for word in word_list if word] 
    vocab = [word for word in word_list if word not in stopwords] 
    return vocab 
  
  
def prepareData():
    traintest_corpus = ResumeCorpus('/Users/divyakarthikeyan/Desktop/COURSES/Fall 2013/NLP/Project/testing/samples_text')
    for resume in traintest_corpus.resumes:
        try:
            review_text = PreProcessing(resume[0])
            review_text = " ".join(review_text)
            data_dict['data'].append(review_text)
            data_dict['label'].append(int(resume[1]))
        except:
            pass
  
  
def vectorize(count_vect, data): 
    X_counts = count_vect.fit_transform(data) 
    return X_counts 
  
  
def tfidftransform(counts): 
    tfidf_transformer = TfidfTransformer() 
    X_tfidf = tfidf_transformer.fit_transform(counts) 
    return X_tfidf 
  
  
def trainnb(tfidf, train_label): 
    clf = MultinomialNB().fit(tfidf, train_label) 
    return clf 
  
  
if __name__ == '__main__': 
    prepareData() 
    kfold = 10
    i = 0
    partition_size = int(len(data_dict['label']) / kfold) 
    print partition_size 
    accuracy_cv = [] 
    precision_cv = [] 
    recall_cv = [] 
    while i < kfold: 
        print i 
        train_data = data_dict['data'][0:i*partition_size] + data_dict['data'][(i+1)*partition_size:] 
        train_label = data_dict['label'][0:i*partition_size] + data_dict['label'][(i+1)*partition_size:] 
  
        count_vect = CountVectorizer() 
        train_counts = vectorize(count_vect, train_data) 
        tfidf_train = tfidftransform(train_counts) 
        clf = trainnb(tfidf_train, train_label) 
        train_counts = [] 
        tfidf_train = [] 
        train_data = [] 
        train_label = [] 
  
        test_data = data_dict['data'][i*partition_size:(i+1)*partition_size] 
        test_counts = count_vect.transform(test_data) 
        tfidf_test = tfidftransform(test_counts) 
        predicted = clf.predict(tfidf_test) 
        test_counts = [] 
        tfidf_test = [] 
        test_data = [] 
        test_label = data_dict['label'][i*partition_size:(i+1)*partition_size] 
        accuracy_cv.append(np.mean(predicted == test_label)) 
        p = precision_score(test_label, predicted, average='macro') 
        r = recall_score(test_label, predicted, average='macro') 
        precision_cv.append(p) 
        recall_cv.append(r) 
        print classification_report([int(t) for t in test_label], [int(p) for p in predicted])
        test_label = [] 
        predicted = [] 
  
        i += 1
  
    print "======================================"
    print "Cross Validation k=10"
    print accuracy_cv 
    print precision_cv 
    print recall_cv 
    print sum(accuracy_cv) / len(accuracy_cv) 
    print sum(precision_cv) / len(precision_cv) 
    print sum(recall_cv) / len(recall_cv) 
    print "======================================"
    file_handler = open("nbuni_newdata_1.out", 'w') 
    file_handler.write("AccuracyCV: %s" % str(sum(accuracy_cv) / len(accuracy_cv))) 
    file_handler.write(" PrecisionCV: %s" % str(sum(precision_cv) / len(precision_cv))) 
    file_handler.write(" RecallCV: %s" % str(sum(recall_cv) / len(recall_cv))) 
  
    num_models = 10
    test_data = data_dict['data'][0:partition_size] 
    test_label = data_dict['label'][0:partition_size] 
    i = 1
    predictions = [] 
    predictions_T = [] 
    predictions_final = [] 
    clfs = [] 
    count_vects = [] 
    while i < num_models: 
        print i 
        train_data = data_dict['data'][i*partition_size:(i+1)*partition_size] 
        train_label = data_dict['label'][i*partition_size:(i+1)*partition_size] 
  
        count_vects.append(CountVectorizer()) 
        train_counts = vectorize(count_vects[i-1], train_data) 
        tfidf_train = tfidftransform(train_counts) 
        clfs.append(trainnb(tfidf_train, train_label)) 
        train_data = [] 
        train_label = [] 
        i += 1
  
    i = 0
    while i < len(count_vects): 
        test_counts = count_vects[i].transform(test_data) 
        tfidf_test = tfidftransform(test_counts) 
        predicted = clfs[i].predict(tfidf_test) 
        predictions.append(predicted) 
        i += 1
  
    predictions_T = map(list,map(None,*predictions)) 
  
    for pred in predictions_T: 
        predictions_final.append(max(set(pred), key=pred.count)) 
  
    accuracy_b = np.mean(np.array(predictions_final) == np.array(test_label)) 
    precision__b = precision_score(test_label, predictions_final, average='macro') 
    recall__b = recall_score(test_label, predictions_final, average='macro') 
  
    print "======================================"
    print "Boosting Models 10"
    print "----------------------"
    print accuracy_b 
    print precision__b 
    print recall__b 
  
    file_handler.write(" AccuracyB: %s" % str(accuracy_b)) 
    file_handler.write(" PrecisionB: %s" % str(precision__b)) 
    file_handler.write(" RecallB: %s" % str(recall__b)) 
    file_handler.close() 
  
    print classification_report([int(t) for t in test_label], [int(p) for p in predictions_final])
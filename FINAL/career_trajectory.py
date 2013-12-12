from __future__ import division
import os
import re
import nltk
import random
import string
from lxml import etree
from nltk import bigrams
from nltk import FreqDist
from collections import Counter

user_name = os.environ.get('USER')
punct = string.punctuation
punctuation = re.compile(r'[^A-Za-z0-9\' ]+')
stopwords = nltk.line_tokenize(open('stopwords.txt').read())
porter = nltk.PorterStemmer()


def create_skills_json(training_data):
    """
    This function will extract all the skills from the training corpus and create a dictionary with Job Titles as
    keys and list of all the skills for that Job Title as values

    Args:
        training_data -- list of tuples. Eg. [(resume, tag, filename), (resume, tag, filename)...]

    Returns:
        A dictionary with Job Titles as keys and list of all the skills for that Job Title as values
    """

    skills_dict = dict()

    # Get the skills for each resume from its corresponding xml file.
    xml_directory = '/Users/' + user_name + '/Documents/Data/samples'
    for (resume_text, tag_name, filename) in training_data:
        xml_file = filename.split('_')[0] + '.txt'
        xml = etree.parse(xml_directory + '/' + xml_file)
        skill_list = xml.xpath('//skills/text()')

        if skill_list:
            slist = []
            for skill in skill_list:
                try:
                    skill = str(skill).encode('utf-8')
                except:
                    skill = skill.encode('utf-8')
                skill = skill.translate(None, ',:();-')
                skill = skill.replace('/', ' ')
                skill = skill.rstrip('.')
                skill_words = nltk.word_tokenize(skill)

                skill_words_nouns = [porter.stem(w.lower()) for (w, t) in nltk.pos_tag(skill_words) if t == 'NNP']
                skill_words_nouns = list(set(skill_words_nouns))
                slist += skill_words_nouns

            value = skills_dict.get(tag_name, None)
            if value is not None:
                skills_dict[tag_name] = value + slist
            else:
                skills_dict[tag_name] = []
                skills_dict[tag_name] = slist

    return skills_dict


def extract_top_skills(training_data):
    """
    Extract Top Skills for each Job Title from skills_new.json file. This json file with all the skills for each Job
    Title will be will be created during pre processing of the training dataset.

    Args:
        training_data -- list of tuples. Eg. [(resume, tag, filename), (resume, tag, filename)...]

    Returns:
        A consolidated list of top skills for all the Job Titles

    """
    skills_dict = create_skills_json(training_data)

    # Read the top n skills for each Job TiTle
    skill_features = []
    for skill in skills_dict:
        skill_list = skills_dict[skill]
        skill_count = Counter(skill_list)
        top_job_skills = sorted(skill_count, key=skill_count.get, reverse=True)[:20]
        skill_features += top_job_skills

    top_job_skills = list(set(skill_features))
    return top_job_skills


class ResumeCorpus():
    """
    Class to read the source files from source directory and create a list of tuples with resume_text, tag and filename
    for each resume.

    Args:
        source_dir -- string. The path of the source directory.
        labels_file -- string. The path of the labels file (default: None)
    """
    def __init__(self, source_dir, labels_file=None):
        
        self.source_dir = source_dir
        if not labels_file:
            self.labels_file = self.source_dir + '/labels.txt'
        else:
            self.labels_file = labels_file
        self.resumes = self.read_files()
        
    def read_files(self):
        """
        Method to return a list of tuples with resume_text, tag and filename for the training data

        Args:
            No Argument

        Returns:
            resumes -- list of tuples with resume_text, tag and filename for the training data
        """
        resumes = []

        for line in open(self.labels_file).readlines():
            try:
                filename_tag = line.split('\t')
                filename = filename_tag[0]
                resume_tag = filename_tag[1].rstrip()
                resumes.append((open(self.source_dir + '/training/' + filename).read(), resume_tag, filename))
            except:
                pass

        return resumes


def train_classifier(training_featureset):
    """
    Function to train the naive bayes classifier using the training features

    Args:
        training_featureset -- dictionary of training features

    Returns:
        rev_classifier -- NaiveBayes classifier object
    """
    rev_classifier = nltk.NaiveBayesClassifier.train(training_featureset)
    return rev_classifier


def feature_consolidation(documents, top_unigrams, top_bigrams,  add_true_score=False):
    """
    Function to consolidate all the featuresets for the training data

    Args:
        documents -- list of tuples [(resume_text, tag, filename), (resume_text, tag, filename)...]
        top_unigrams -- list of top unigrams from the training dataset
        top_bigrams -- list of top bigrams from the training dataset
        add_true_score -- boolean (default: False)

    Returns:
        consolidated_features -- list of consolidated features
    """
    if add_true_score:
        uni_features = [(unigram_features(resume_text, top_unigrams), tag) for (resume_text, tag, filename) in documents]
        bi_features = [(bigram_features(resume_text, top_bigrams), tag) for (resume_text, tag, filename) in documents]
        consolidated_features = uni_features + bi_features
    else:   
        uni_features = unigram_features(documents, top_unigrams)
        bi_features = bigram_features(documents, top_bigrams)
        consolidated_features = dict(uni_features.items() + bi_features.items())
    return consolidated_features
    #return uni_features


def unigram_features(resume_text, top_unigrams):
    """
    Function to create unigram features from the resume text

    Args:
        resume_text -- content of resume as string
        top_unigrams -- list of top unigrams

    Returns:
        features -- dictionary of unigram features
    """
    resume_text = re.sub('[^A-Za-z\' ]+', '', str(resume_text))
    tokens = nltk.word_tokenize(resume_text)
    features = {}
    avg_word_len= 0
    count = 0
    for token in tokens:
        token_stem = str(porter.stem(token))
        if token_stem in top_unigrams:
            features[token_stem] = True
            avg_word_len += len(token_stem)
            count += 1
    features['average_word_length'] = avg_word_len/(count+1)
    features['docu_length'] = len(tokens)
    return features


def bigram_features(resume_text, top_bigrams):
    """
    Function to create bigram features from the resume text

    Args:
        resume_text -- content of resume as string
        top_bigrams -- list of top bigrams

    Returns:
        features -- dictionary of bigram features
    """
    tokens = nltk.word_tokenize(resume_text)
    bigrs = bigrams(tokens)
    bigram_list = []
    bigram_list += [(bigrm[0], bigrm[1]) for bigrm in bigrs if (bigr[0] not in stopwords or bigr[1] not in stopwords)]
    features = {}
    for bigram in bigram_list:
        if bigram in top_bigrams:
            features[bigram] = True
    return features


if __name__ == "__main__":
    user_name = os.environ.get('USER')
    traintest_corpus = ResumeCorpus('/Users/' + user_name + '/Documents/Data')

    random.shuffle(traintest_corpus.resumes)
    random.shuffle(traintest_corpus.resumes)
    num_resumes = len(traintest_corpus.resumes)
    train_resumes = traintest_corpus.resumes[0:int(num_resumes*0.9)]
    test_resumes = traintest_corpus.resumes[int(num_resumes*0.9) + 1:]

    top_skills = extract_top_skills(train_resumes)
    #print len(top_skills)
    #print top_skills

    words = []
    bigrams_list = []
    for resume in train_resumes:
        unigrams = resume[0].lower().split()
        words = words + unigrams
        bigrms = bigrams(unigrams)
        bigrams_list += [(bigr[0], bigr[1]) for bigr in bigrms if (bigr[0] not in stopwords or bigr[1] not in stopwords)]

    fd = FreqDist(words)
    fd_bi = FreqDist(bigrams_list)

    top_unigrams = [porter.stem(word) for word in fd.keys()[:200] if word not in stopwords]
    top_unigrams += top_skills
    top_bigrams = []
    top_bigrams = fd_bi.keys()[:100]

    train_featureset = feature_consolidation(train_resumes, top_unigrams, top_bigrams, True)
    review_classifier = train_classifier(train_featureset)

    output_file = open('classifier_output.txt','w')
    for document in test_resumes:
        resume_features = feature_consolidation(document[0], top_unigrams, top_bigrams)
        (text, tag, fileName) = document
        classifier_output = review_classifier.classify(resume_features)
        output_file.write('%s' %fileName + '\t' + '%s' %str(tag) + '\t' + '%s' %classifier_output + '\n')

    output_file.close()
    test_featureset = feature_consolidation(test_resumes, top_unigrams, top_bigrams, True)
    accuracy = nltk.classify.accuracy(review_classifier, test_featureset)
    print accuracy

    review_classifier.show_most_informative_features(50)

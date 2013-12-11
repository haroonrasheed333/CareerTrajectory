from __future__ import division
import nltk
import os, re
import random
import string
from lxml import etree
from nltk import bigrams
from nltk import FreqDist
from collections import Counter

user_name = os.environ.get('USER')
punct = string.punctuation
punctuation = re.compile(r'[^A-Za-z0-9\' ]+')
stopwords = ["a's","accordingly","again","allows","also","amongst","anybody","anyways","appropriate","aside","available","because","before","below","between","by","can't","certain","com","consider","corresponding","definitely","different","don't","each","else","et","everybody","exactly","fifth","follows","four","gets","goes","greetings","has","he","her","herein","him","how","i'm","immediate","indicate","instead","it","itself","know","later","lest","likely","ltd","me","more","must","nd","needs","next","none","nothing","of","okay","ones","others","ourselves","own","placed","probably","rather","regarding","right","saying","seeing","seen","serious","she","so","something","soon","still","t's","th","that","theirs","there","therein","they'd","third","though","thus","toward","try","under","unto","used","value","vs","way","we've","weren't","whence","whereas","whether","who's","why","within","wouldn't","you'll","yourself","able","across","against","almost","although","an","anyhow","anywhere","are","ask","away","become","beforehand","beside","beyond","c'mon","cannot","certainly","come","considering","could","described","do","done","edu","elsewhere","etc","everyone","example","first","for","from","getting","going","had","hasn't","he's","here","hereupon","himself","howbeit","i've","in","indicated","into","it'd","just","known","latter","let","little","mainly","mean","moreover","my","near","neither","nine","noone","novel","off","old","only","otherwise","out","particular","please","provides","rd","regardless","said","says","seem","self","seriously","should","some","sometime","sorry","sub","take","than","that's","them","there's","theres","they'll","this","three","to","towards","trying","unfortunately","up","useful","various","want","we","welcome","what","whenever","whereby","which","whoever","will","without","yes","you're","yourselves","about","actually","ain't","alone","always","and","anyone","apart","aren't","asking","awfully","becomes","behind","besides","both","c's","cant","changes","comes","contain","couldn't","despite","does","down","eg","enough","even","everything","except","five","former","further","given","gone","hadn't","have","hello","here's","hers","his","however","ie","inasmuch","indicates","inward","it'll","keep","knows","latterly","let's","look","many","meanwhile","most","myself","nearly","never","no","nor","now","often","on","onto","ought","outside","particularly","plus","que","re","regards","same","second","seemed","selves","seven","shouldn't","somebody","sometimes","specified","such","taken","thank","thats","themselves","thereafter","thereupon","they're","thorough","through","together","tried","twice","unless","upon","uses","very","wants","we'd","well","what's","where","wherein","while","whole","willing","won't","yet","you've","zero","above","after","all","along","am","another","anything","appear","around","associated","be","becoming","being","best","brief","came","cause","clearly","concerning","containing","course","did","doesn't","downwards","eight","entirely","ever","everywhere","far","followed","formerly","furthermore","gives","got","happens","haven't","help","hereafter","herself","hither","i'd","if","inc","inner","is","it's","keeps","last","least","like","looking","may","merely","mostly","name","necessary","nevertheless","nobody","normally","nowhere","oh","once","or","our","over","per","possible","quite","really","relatively","saw","secondly","seeming","sensible","several","since","somehow","somewhat","specify","sup","tell","thanks","the","then","thereby","these","they've","thoroughly","throughout","too","tries","two","unlikely","us","using","via","was","we'll","went","whatever","where's","whereupon","whither","whom","wish","wonder","you","your","according","afterwards","allow","already","among","any","anyway","appreciate","as","at","became","been","believe","better","but","can","causes","co","consequently","contains","currently","didn't","doing","during","either","especially","every","ex","few","following","forth","get","go","gotten","hardly","having","hence","hereby","hi","hopefully","i'll","ignored","indeed","insofar","isn't","its","kept","lately","less","liked","looks","maybe","might","much","namely","need","new","non","not","obviously","ok","one","other","ours","overall","perhaps","presumably","qv","reasonably","respectively","say","see","seems","sent","shall","six","someone","somewhere","specifying","sure","tends","thanx","their","thence","therefore","they","think","those","thru","took","truly","un","until","use","usually","viz","wasn't","we're","were","when","whereafter","wherever","who","whose","with","would","you'd","yours"]
lemma = nltk.WordNetLemmatizer()
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
        top_job_skills = sorted(skill_count, key=skill_count.get, reverse=True)[:30]
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
    if add_true_score:
        uni_features = [(unigram_features(resume_text, top_unigrams), tag) for (resume_text, tag, filename) in documents]
        bi_features = [(bigram_features(resume_text, top_bigrams), tag) for (resume_text, tag, filename) in documents]
        consolidated_features = uni_features + bi_features
    else:   
        uni_features = unigram_features(documents, top_unigrams)
        bi_features = unigram_features(documents, top_bigrams)
        consolidated_features = uni_features.update(bi_features)
    return consolidated_features


def unigram_features(resume_text, top_unigrams):
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
    resume_text = re.sub('[^A-Za-z\' ]+', '', str(resume_text))
    tokens = nltk.word_tokenize(resume_text)
    bigrs = bigrams(tokens)
    bigram_list = []
    bigram_list += ["".join([porter.stem(bigr[0]), porter.stem(bigr[1])]) for bigr in bigrs if (bigr[0] not in stopwords or bigr[1] not in stopwords)]
    features = {}
    for bigram in bigram_list:
        if bigram in top_bigrams:
            features[bigram] = True
    return features


if __name__ == "__main__":
    user_name = os.environ.get('USER')
    traintest_corpus = ResumeCorpus('/Users/' + user_name + '/Documents/Data')

    random.shuffle(traintest_corpus.resumes)
    num_resumes = len(traintest_corpus.resumes)
    train_resumes = traintest_corpus.resumes[0:int(num_resumes*0.9)]
    test_resumes = traintest_corpus.resumes[int(num_resumes*0.9) + 1:]

    top_skills = extract_top_skills(train_resumes)
    print len(top_skills)
    print top_skills

    words = []
    bigrams_list = []
    for resume in train_resumes:
        unigrams = resume[0].split()
        words = words + unigrams
        bigrms = bigrams(unigrams)
        bigrams_list += ["".join([porter.stem(bigr[0]), porter.stem(bigr[1])]) for bigr in bigrms if (bigr[0] not in stopwords or bigr[1] not in stopwords)]

    fd = FreqDist(words)
    fd_bi = FreqDist(bigrams_list)

    top_unigrams = [porter.stem(word) for word in fd.keys()[:200] if word not in stopwords]
    top_bigrams = fd_bi.keys()[:30]

    train_featureset  = feature_consolidation(train_resumes, top_unigrams, top_bigrams, True)
    review_classifier = train_classifier(train_featureset)

    output_file = open('classifier_output.txt','w')
    for document in test_resumes:
        resume_features = feature_consolidation(document[0], top_unigrams, top_bigrams)
        (text, tag, fileName) = document
        classifier_output = review_classifier.classify(resume_features)
        output_file.write('%s' %fileName + '\t' + '%s' %str(tag) + '\t' + '%s' %classifier_output + '\n')

    output_file.close()
    test_featureset  = feature_consolidation(test_resumes, top_unigrams, top_bigrams, True)
    accuracy = nltk.classify.accuracy(review_classifier, test_featureset)
    print accuracy

    review_classifier.show_most_informative_features(50)

from sklearn.feature_extraction.text import TfidfVectorizer
from careertrajectory import ResumeCorpus
from sklearn.cluster import KMeans
import re
import os
from nltk.stem.porter import PorterStemmer
import logging
from optparse import OptionParser
import sys
from time import time
clusters = 5
stopwords   = ["a's","accordingly","again","allows","also","amongst","anybody","anyways","appropriate","aside","available","because","before","below","between","by","can't","certain","com","consider","corresponding","definitely","different","don't","each","else","et","everybody","exactly","fifth","follows","four","gets","goes","greetings","has","he","her","herein","him","how","i'm","immediate","indicate","instead","it","itself","know","later","lest","likely","ltd","me","more","must","nd","needs","next","none","nothing","of","okay","ones","others","ourselves","own","placed","probably","rather","regarding","right","saying","seeing","seen","serious","she","so","something","soon","still","t's","th","that","theirs","there","therein","they'd","third","though","thus","toward","try","under","unto","used","value","vs","way","we've","weren't","whence","whereas","whether","who's","why","within","wouldn't","you'll","yourself","able","across","against","almost","although","an","anyhow","anywhere","are","ask","away","become","beforehand","beside","beyond","c'mon","cannot","certainly","come","considering","could","described","do","done","edu","elsewhere","etc","everyone","example","first","for","from","getting","going","had","hasn't","he's","here","hereupon","himself","howbeit","i've","in","indicated","into","it'd","just","known","latter","let","little","mainly","mean","moreover","my","near","neither","nine","noone","novel","off","old","only","otherwise","out","particular","please","provides","rd","regardless","said","says","seem","self","seriously","should","some","sometime","sorry","sub","take","than","that's","them","there's","theres","they'll","this","three","to","towards","trying","unfortunately","up","useful","various","want","we","welcome","what","whenever","whereby","which","whoever","will","without","yes","you're","yourselves","about","actually","ain't","alone","always","and","anyone","apart","aren't","asking","awfully","becomes","behind","besides","both","c's","cant","changes","comes","contain","couldn't","despite","does","down","eg","enough","even","everything","except","five","former","further","given","gone","hadn't","have","hello","here's","hers","his","however","ie","inasmuch","indicates","inward","it'll","keep","knows","latterly","let's","look","many","meanwhile","most","myself","nearly","never","no","nor","now","often","on","onto","ought","outside","particularly","plus","que","re","regards","same","second","seemed","selves","seven","shouldn't","somebody","sometimes","specified","such","taken","thank","thats","themselves","thereafter","thereupon","they're","thorough","through","together","tried","twice","unless","upon","uses","very","wants","we'd","well","what's","where","wherein","while","whole","willing","won't","yet","you've","zero","above","after","all","along","am","another","anything","appear","around","associated","be","becoming","being","best","brief","came","cause","clearly","concerning","containing","course","did","doesn't","downwards","eight","entirely","ever","everywhere","far","followed","formerly","furthermore","gives","got","happens","haven't","help","hereafter","herself","hither","i'd","if","inc","inner","is","it's","keeps","last","least","like","looking","may","merely","mostly","name","necessary","nevertheless","nobody","normally","nowhere","oh","once","or","our","over","per","possible","quite","really","relatively","saw","secondly","seeming","sensible","several","since","somehow","somewhat","specify","sup","tell","thanks","the","then","thereby","these","they've","thoroughly","throughout","too","tries","two","unlikely","us","using","via","was","we'll","went","whatever","where's","whereupon","whither","whom","wish","wonder","you","your","according","afterwards","allow","already","among","any","anyway","appreciate","as","at","became","been","believe","better","but","can","causes","co","consequently","contains","currently","didn't","doing","during","either","especially","every","ex","few","following","forth","get","go","gotten","hardly","having","hence","hereby","hi","hopefully","i'll","ignored","indeed","insofar","isn't","its","kept","lately","less","liked","looks","maybe","might","much","namely","need","new","non","not","obviously","ok","one","other","ours","overall","perhaps","presumably","qv","reasonably","respectively","say","see","seems","sent","shall","six","someone","somewhere","specifying","sure","tends","thanx","their","thence","therefore","they","think","those","thru","took","truly","un","until","use","usually","viz","wasn't","we're","were","when","whereafter","wherever","who","whose","with","would","you'd","yours"]
user_name = os.environ.get('USER')
traintest_corpus = ResumeCorpus('/Users/' + user_name + '/Documents/Data/samples_text')
docs=[]
filename = {}
i= 0
for resume in traintest_corpus.resumes[:2000]:
    docs.append(resume[0])
    filename[i] = resume[2]
    i += 1

#docs =["space star sky spaceship", "math cosine similarity", "space astronaut meteor", "add subtract math calculate"]

vectorizer = TfidfVectorizer(min_df=1)
tfidf = vectorizer.fit_transform(docs)
km = KMeans(n_clusters=clusters, init='k-means++', max_iter=10, n_init=1)
km.fit(tfidf)
results = []

for i in range (0,len(tfidf.toarray())):
    try:
        results.append([str(filename[i]),int(km.predict(tfidf.toarray()[i]))])
        #i +=1
    except:
        pass

# Generate wordclouds for all clusters

#wordclouds_all = []
for cluster in range (0,clusters-1):
    wordcloudtext =""
    for i in range (0,len(results)-1):
        if results[i][1] == cluster:
            user_name = os.environ.get('USER')
            document = open('/Users/' + user_name + '/Documents/Data/samples_text/'+results[i][0],'r').read()
            docu=re.sub('[^A-Za-z\' ]+', '', str(document).lower())
            unigrams = docu.split()
            word_list = [word.lower() for word in unigrams if word.lower() not in stopwords]
            text = " ".join(word_list)
            wordcloudtext += text
    f=open('/Users/' + user_name + '/Documents/Data/'+str(cluster) +".txt",'w')
    f.write(wordcloudtext)
    f.close()
    #wordclouds_all.append(wordcloudtext)




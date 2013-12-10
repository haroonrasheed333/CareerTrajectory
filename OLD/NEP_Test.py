import nltk

entities = dict()
entities['PERSON'] = []
entities['ORGANIZATION'] = []
entities['GPE'] = []


def extract_entities(tree):

    try:
        if tree.node == 'ORGANIZATION':
            entities['ORGANIZATION'].append(' '.join([child[0] for child in tree]))
        elif tree.node == 'PERSON':
            entities['PERSON'].append(' '.join([child[0] for child in tree]))
        elif tree.node == 'GPE':
            entities['GPE'].append(' '.join([child[0] for child in tree]))
        else:
            for child in tree:
                extract_entities(child)
    except:
        pass

    return entities


def main():
    f = open('DIV.txt', 'r')
    text = f.read()

    sents = nltk.sent_tokenize(text)
    #print sents
    tokenized_sents = [nltk.word_tokenize(sent) for sent in sents]
    #print tokenized_sents
    tagged_sents = [nltk.pos_tag(sentence) for sentence in tokenized_sents]
    #print tagged_sents
    chunked_sents = [nltk.ne_chunk(tagged_sent) for tagged_sent in tagged_sents]
    #print chunked_sents

    for tree in chunked_sents:
        extract_entities(tree)

    print "PERSON: " + str(list(set(entities['PERSON'])))
    print "ORGANIZATION: " + str(list(set(entities['ORGANIZATION'])))
    print "GPE: " + str(list(set(entities['GPE'])))


if __name__ == "__main__":
    main()
import nltk
from nltk.tokenize import word_tokenize
from lxml import etree
import pprint, random, pickle
import itertools
import os, sys, re
import random
from pprint import pprint as pp
from itertools import islice
from nltk import FreqDist
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize



#!/usr/bin/python

class ResumeCorpus():
    def __init__(self, source_dir):
        """
        init with path do the directory with the .txt files
        """
        
        self.source_dir = source_dir
        self.files = self.getFiles(self.source_dir)
        self.readFiles(self.files, self.source_dir)
        
    def getFiles(self, source_dir):
        files = [ f for (dirpath, dirnames, filenames) in os.walk(source_dir) for f in filenames if f[-4:] == '.txt' ]
        return files
   
    def readFiles(self, files, source_dir):
             
        def stripxml(data):
            pattern = re.compile(r'<.*?>')
            return pattern.sub('', data)
    
        labels = open ('labels.txt', 'w')
        for fname in files:
            #data = open(source_dir + '/' + fname).read()
            xml = etree.parse(source_dir + '/' + fname)
            current_employer = xml.xpath('//job[@end = "present"]/employer/text()')
            print current_employer
        
            current_job_title = xml.xpath('//job[@end = "present"]/title/text()')
            print current_job_title
        
            current_job = xml.xpath('//job[@end = "present"]')
            if current_job:
                current_job[0].getparent().remove(current_job[0])
            xml = etree.tostring(xml, pretty_print=True)      
            text_data = stripxml(xml)
            if current_job_title:
                text_data.replace(current_job_title[0], '')
            #print text_data
        
            f = open('samples_text/' + '%s' %fname[:-4] +'_plaintext.txt', 'w')
            f.write(text_data)
            f.close()
            if current_job_title:
                labels.writelines(fname[:-4] +'_plaintext.txt' + "\t" + current_job_title[0] + "\n")
        return


if __name__ == "__main__":
    traintest_corpus = ResumeCorpus('samples') #4256

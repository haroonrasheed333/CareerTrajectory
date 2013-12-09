from lxml import etree
import os, re
import nltk
import progressbar
import json


def pbar(size):
    bar = progressbar.ProgressBar(maxval=size,
                                  widgets=[progressbar.Bar('=', '[', ']'),
                                           ' ', progressbar.Percentage(),
                                           ' ', progressbar.ETA(),
                                           ' ', progressbar.Counter(),
                                           '/%s' % size])
    return bar


class ResumeCorpus():
    def __init__(self, source_dir):
        
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

        user_name = os.environ.get('USER')
        labels = open ('/Users/' + user_name + '/Documents/Data/labels.txt', 'w')
        skill_file = open('skills.txt', 'w')
        #labels_company = open ('labelscompany.txt', 'w')

        names = []
        job_titles = []

        hjobs50 = ['Director', 'Consultant', 'Administrative Assistant', 'Project Manager', 'Manager', 'Owner', 'Vice President', 'Sales Associate', 'Contractor', 'Graphic Designer', 'Customer Service Representative', 'Intern', 'Office Manager', 'Research Assistant', 'Executive Assistant', 'Cashier', 'Volunteer', 'President', 'Software Engineer', 'Business Analyst', 'Senior Software Engineer', 'Account Executive', 'Substitute Teacher', 'Assistant Manager', 'Supervisor', 'Receptionist', 'Program Manager', 'Graduate Assistant', 'Sales Representative', 'Graduate Research Assistant', 'Teaching Assistant', 'Principal', 'Marketing Manager', 'Office Assistant', 'Accountant', 'Account Manager', 'Instructor', 'Web Developer', 'Senior Manager', 'Business Development Manager', 'Associate', 'Medical Assistant', 'Marketing Consultant', 'Computer Technician', 'Senior Consultant', 'Bookkeeper', 'VP', 'Staff Accountant', 'Senior Project Manager', 'Senior Accountant']
        hjobs = ['Director', 'Consultant', 'Project Manager', 'Owner', 'Vice President', 'Sales Associate', 'Graphic Designer', 'Customer Service Representative', 'Software Engineer', 'Business Analyst', 'Senior Software Engineer', 'Account Executive', 'Assistant Manager', 'Supervisor', 'Receptionist', 'Program Manager']
        #hjobs = hjobs50[:20]
        numberjobs = {}
        for i in range(0,len(hjobs)-1):
            numberjobs[hjobs[i]] = i+1
        #print numberjobs

        skills = dict()

        for job in hjobs:
            skills[job] = []

        j, bar = 0, pbar(len(files))
        bar.start()

        for fname in files:
            xml = etree.parse(source_dir + '/' + fname)
            current_employer = xml.xpath('//job[@end = "present"]/employer/text()')
            current_job_title = xml.xpath('//job[@end = "present"]/title/text()')
            current_job = xml.xpath('//job[@end = "present"]')
            contact = xml.xpath('//contact')
            skill_list = xml.xpath('//skills/text()')
            job_title = ''
            try:
                name = xml.xpath('//givenname/text()')[0] + ' ' + xml.xpath('//surname/text()')[0]
                if name not in names:
                    names.append(name)
                    if current_job:
                        xml = etree.tostring(xml, pretty_print=True)
                        text_data = stripxml(xml)
                        i = 0
                        flag = 0
                        if current_job_title:
                            i = 0
                            if len(current_job_title)>1:
                                while (i<len(current_job_title)):
                                    text_data = text_data.replace(current_job_title[i], '')
                                    job_titles.append(current_job_title[i])
                                    i = i+1
                                    if current_job_title[i] in hjobs:
                                        job_title = current_job_title[i]
                                        flag = 1
                            else:
                                text_data = text_data.replace(current_job_title[0], '')
                                job_titles.append(current_job_title[0])
                                if current_job_title[i] in hjobs:
                                    job_title = current_job_title[i]
                                    flag = 1

                    user_name = os.environ.get('USER')

                    if flag == 1:
                        if len(current_job)>1:
                            while (i<len(current_job)):
                                current_job[i].getparent().remove(current_job[i])
                                i = i+1
                        else:
                            current_job[0].getparent().remove(current_job[0])

                        if contact:
                            contact[0].getparent().remove(contact[0])

                        if skill_list:
                            slist = []
                            for skill in skill_list:
                                skill = skill.replace(',', ' ')
                                skill = skill.replace(':', '')
                                skill = skill.replace('(', '')
                                skill = skill.replace(')', '')
                                skill = skill.replace(';', '')
                                skill = skill.replace('/', ' ')
                                words = nltk.word_tokenize(skill)

                                skill_words = [word.lower() for (word, tag) in nltk.pos_tag(words) if tag == 'NNP']
                                skill_words = list(set(skill_words))
                                slist += skill_words

                            skills[job_title].append(slist)

                        number = numberjobs[current_job_title[0]]
                        #directory = '/Users/' + user_name + '/Documents/Data/samples_text_1208/' + str(number)
                        #if not os.path.exists(directory):
                        #    os.makedirs(directory)

                        #f = open(directory + '/' + '%s' %fname[:-4] +'_plaintext.txt', 'w')
                        directory = '/Users/' + user_name + '/Documents/Data/samples_text_1208/'
                        f = open(directory + '%s' %fname[:-4] +'_plaintext.txt', 'w')
                        f.write(text_data)
                        f.close()
                        if current_job_title:
                            number = numberjobs[current_job_title[0]]
                            labels.writelines(fname[:-4] + '_plaintext.txt' + "\t" + str(number) + "\n")
            except:
                pass

            j += 1
            bar.update(j)
        bar.finish()

        skills_json = json.dumps(skills, indent=4)
        filename = 'skills.json'
        sf = open(filename, 'w')
        print >> sf, skills_json
        sf.close()

        print len(job_titles)
        print len(list(set(job_titles)))

        return


if __name__ == "__main__":
    user_name = os.environ.get('USER')
    traintest_corpus = ResumeCorpus('/Users/' + user_name + '/Documents/Data/samples')

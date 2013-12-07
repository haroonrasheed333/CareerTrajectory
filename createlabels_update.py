from lxml import etree
import os, re

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
    
        labels = open ('labels.txt', 'w')
        names = []
        job_titles = []

        hjobs_50 = ['Director', 'Consultant', 'Administrative Assistant', 'Project Manager', 'Manager', 'Owner', 'Vice President', 'Sales Associate', 'Contractor', 'Graphic Designer', 'Customer Service Representative', 'Intern', 'Office Manager', 'Research Assistant', 'Executive Assistant', 'Cashier', 'Volunteer', 'President', 'Software Engineer', 'Business Analyst', 'Senior Software Engineer', 'Account Executive', 'Substitute Teacher', 'Assistant Manager', 'Supervisor', 'Receptionist', 'Program Manager', 'Graduate Assistant', 'Sales Representative', 'Graduate Research Assistant', 'Teaching Assistant', 'Principal', 'Marketing Manager', 'Office Assistant', 'Accountant', 'Account Manager', 'Instructor', 'Web Developer', 'Senior Manager', 'Business Development Manager', 'Associate', 'Medical Assistant', 'Marketing Consultant', 'Computer Technician', 'Senior Consultant', 'Bookkeeper', 'VP', 'Staff Accountant', 'Senior Project Manager', 'Senior Accountant']
        hjobs = hjobs_50[:20]
        for fname in files:
            xml = etree.parse(source_dir + '/' + fname)
            current_employer = xml.xpath('//job[@end = "present"]/employer/text()')
            current_job_title = xml.xpath('//job[@end = "present"]/title/text()')
            current_job = xml.xpath('//job[@end = "present"]')
            contact = xml.xpath('//contact')
            try:
                name = xml.xpath('//givenname/text()')[0] + ' ' + xml.xpath('//surname/text()')[0]
                if name not in names:
                    names.append(name)
                    if current_job:
                        i = 0
                        if len(current_job)>1:
                            while (i<len(current_job)):
                                current_job[i].getparent().remove(current_job[i])
                                i = i+1
                        else:
                            current_job[0].getparent().remove(current_job[0])

                        if contact:
                            contact[0].getparent().remove(contact[0])

                        xml = etree.tostring(xml, pretty_print=True)
                        text_data = stripxml(xml)
                        flag = 0
                        if current_job_title:
                            i = 0
                            if len(current_job_title)>1:
                                while (i<len(current_job_title)):
                                    text_data = text_data.replace(current_job_title[i], '')
                                    job_titles.append(current_job_title[i])
                                    i = i+1
                                    if current_job_title[i] in hjobs:
                                        flag = 1
                            else:
                                text_data = text_data.replace(current_job_title[0], '')
                                job_titles.append(current_job_title[0])
                                if current_job_title[i] in hjobs:
                                    flag = 1

                    if flag == 1:
                        f = open('samples_text/' + '%s' %fname[:-4] +'_plaintext.txt', 'w')
                        f.write(text_data)
                        f.close()
                        if current_job_title:
                            labels.writelines(fname[:-4] +'_plaintext.txt' + "\t" + current_job_title[0] + "\n")

            except:
                pass
        print len(job_titles)
        print len(list(set(job_titles)))

        #import collections
        #counter = [(y,x) for x, y in collections.Counter(job_titles).items()]
        #counter_sort = sorted(counter, reverse=True)
        #print counter_sort[:20]
        #hj = []
        #for j in counter_sort[:20]:
        #    hj.append(j[1])
        #
        #print hj
        return


if __name__ == "__main__":
    traintest_corpus = ResumeCorpus('samples') #4256
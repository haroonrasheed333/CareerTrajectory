import os
import re
import random
import shutil
import progressbar
from lxml import etree

user_name = os.environ.get('USER')


def pbar(size):
    """

    """
    bar = progressbar.ProgressBar(maxval=size,
                                  widgets=[progressbar.Bar('=', '[', ']'),
                                           ' ', progressbar.Percentage(),
                                           ' ', progressbar.ETA(),
                                           ' ', progressbar.Counter(),
                                           '/%s' % size])
    return bar


def split_data(labels_list):
    """
    Function to split the dataset into training and heldout datasets

    Args:
        labels_list -- list of tuples with filename and tag information for each resume
    """
    source_dir = '/Users/' + user_name + '/Documents/Data/samples_text'
    training_dir = '/Users/' + user_name + '/Documents/Data/training'
    heldout_dir = '/Users/' + user_name + '/Documents/Data/heldout'

    random.shuffle(labels_list)
    random.shuffle(labels_list)

    num_files = len(labels_list)

    training_files = labels_list[:int(num_files*0.8)]
    heldout_files = labels_list[int(num_files*0.8) + 1:]

    labels = open('/Users/' + user_name + '/Documents/Data/labels.txt', 'w')
    labels_heldout = open('/Users/' + user_name + '/Documents/Data/labels_heldout.txt', 'w')

    for (filename, tag) in training_files:
        shutil.copy2(source_dir + '/' + filename, training_dir)
        labels.writelines(filename + "\t" + tag + "\n")

    for (filename, tag) in heldout_files:
        shutil.copy2(source_dir + '/' + filename, heldout_dir)
        labels_heldout.writelines(filename + "\t" + tag + "\n")

    labels.close()
    labels_heldout.close()


def stripxml(data):
    """

    Args:

    """
    pattern = re.compile(r'<.*?>')
    return pattern.sub('', data)

   
def prepare_data(source_dir):
    """

    """

    files = [f for (dirpath, dirnames, filenames) in os.walk(source_dir) for f in filenames if f[-4:] == '.txt']

    names = []
    job_titles = []

    hjobs50 = ['Director', 'Consultant', 'Administrative Assistant', 'Project Manager', 'Manager', 'Owner', 'Vice President', 'Sales Associate', 'Contractor', 'Graphic Designer', 'Customer Service Representative', 'Intern', 'Office Manager', 'Research Assistant', 'Executive Assistant', 'Cashier', 'Volunteer', 'President', 'Software Engineer', 'Business Analyst', 'Senior Software Engineer', 'Account Executive', 'Substitute Teacher', 'Assistant Manager', 'Supervisor', 'Receptionist', 'Program Manager', 'Graduate Assistant', 'Sales Representative', 'Graduate Research Assistant', 'Teaching Assistant', 'Principal', 'Marketing Manager', 'Office Assistant', 'Accountant', 'Account Manager', 'Instructor', 'Web Developer', 'Senior Manager', 'Business Development Manager', 'Associate', 'Medical Assistant', 'Marketing Consultant', 'Computer Technician', 'Senior Consultant', 'Bookkeeper', 'VP', 'Staff Accountant', 'Senior Project Manager', 'Senior Accountant']
    hjobs = hjobs50[:20]
    numberjobs = {}
    for i in range(0,len(hjobs)-1):
        numberjobs[hjobs[i]] = i+1

    j, bar = 0, pbar(len(files))
    bar.start()
    labels_list = []

    for fname in files:
        xml = etree.parse(source_dir + '/' + fname)
        current_job_title = xml.xpath('//job[@end = "present"]/title/text()')
        current_job = xml.xpath('//job[@end = "present"]')
        contact = xml.xpath('//contact')
        try:
            name = xml.xpath('//givenname/text()')[0] + ' ' + xml.xpath('//surname/text()')[0]
            if name not in names:
                names.append(name)
                if contact:
                        contact[0].getparent().remove(contact[0])

                if current_job:
                    if len(current_job) > 1:
                        while i < len(current_job):
                            current_job[i].getparent().remove(current_job[i])
                            i += 1
                    else:
                        current_job[0].getparent().remove(current_job[0])

                    xml = etree.tostring(xml, pretty_print=True)
                    text_data = stripxml(xml)
                    i = 0
                    flag = 0
                    if current_job_title:
                        i = 0
                        if len(current_job_title) > 1:
                            while i < len(current_job_title):
                                text_data = text_data.replace(current_job_title[i], '')
                                job_titles.append(current_job_title[i])
                                i += 1
                                if current_job_title[i] in hjobs:
                                    job_title = current_job_title[i]
                                    flag = 1
                        else:
                            text_data = text_data.replace(current_job_title[0], '')
                            job_titles.append(current_job_title[0])
                            if current_job_title[i] in hjobs:
                                job_title = current_job_title[i]
                                flag = 1

                if flag == 1:

                    number = numberjobs[current_job_title[0]]
                    directory = '/Users/' + user_name + '/Documents/Data/samples_text/' + str(number)

                    if current_job_title:
                        directory = '/Users/' + user_name + '/Documents/Data/samples_text/'
                        f = open(directory + '%s' %fname[:-4] +'_plaintext.txt', 'w')
                        f.write(text_data)
                        f.close()

                        labels_list.append((fname[:-4] + '_plaintext.txt', current_job_title[0].replace('\n', '')))
        except:
            pass

        j += 1
        bar.update(j)
    bar.finish()

    split_data(labels_list)

    return

if __name__ == "__main__":
    prepare_data('/Users/' + user_name + '/Documents/Data/samples')


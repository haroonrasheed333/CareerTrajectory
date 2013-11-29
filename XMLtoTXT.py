import re
from lxml import etree
import os


def stripxml(data):
    pattern = re.compile(r'<.*?>')
    return pattern.sub('', data)


def main():
    data = open('Resume1.txt').read()
    #print data

    text_data = stripxml(data)
    #print text_data

    source_dir = 'samples'

    files = [ f for (dirpath, dirnames, filenames) in os.walk(source_dir) for f in filenames]
    print files
    print len(files)

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

        f = open('samples_text/' + '%s' %fname +'_plaintext.txt', 'w')
        f.write(text_data)
        f.close()


if __name__ == '__main__':
    main()
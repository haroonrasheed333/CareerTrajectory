from lxml import etree
import re

xml = etree.parse("Resume1.txt")
current_employer = xml.xpath('//job[@end = "present"]/employer/text()')
print current_employer

current_job_title = xml.xpath('//job[@end = "present"]/title/text()')
print current_job_title


def stripxml(data):
    pattern = re.compile(r'<.*?>')
    return pattern.sub('', data)


def main():
    data = open('Resume1.txt').read()
    #print data

    text_data = stripxml(data)
    print text_data

    f = open('Resume_plaintext.txt', 'w')
    f.write(text_data)
    f.close()


if __name__ == '__main__':
    main()
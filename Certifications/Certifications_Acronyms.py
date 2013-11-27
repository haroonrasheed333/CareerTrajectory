import json
from bs4 import BeautifulSoup
import urllib2
import string

baseurl = 'http://www.careeronestop.org/educationtraining/find/certification-finder.aspx?keyword='
with open('certifications1.json') as data:
    json_data = json.load(data)

n = 0
while n < len(json_data['certifications']):
    i = 0
    while i < len(json_data['certifications'][n]['certification_list']):
        keyword = json_data['certifications'][n]['certification_list'][i]['Certification Name']
        keyword = string.replace(keyword, " ", "%20")
        certid_base = json_data['certifications'][n]['certification_list'][i]['Certification ID']

        for x in ['-A', '-B', '-C']:
            certid = certid_base + x
            url = baseurl + keyword + '&direct=0&certid=' + certid
            flag = 0
            try:
                response = urllib2.urlopen(url)
                html = response.read()
                soup = BeautifulSoup(html)
                table = soup.findAll('table')[0]
                cert_name = table.findAll('th')[0].text
                start_pos = cert_name.find('(')
                stop_pos = cert_name.find(')')
                if start_pos > 0:
                    acr = cert_name[start_pos+1:stop_pos]
                else:
                    acr = ''
                json_data['certifications'][n]['certification_list'][i]['Acronym'] = ''
                json_data['certifications'][n]['certification_list'][i]['Acronym'] = acr
                if flag == 0:
                    flag = 1
                break
            except:
                pass

            if flag == 0:
                json_data['certifications'][n]['certification_list'][i]['Acronym'] = ''

        i += 1

    temp_cert = json_data['certifications'][n]
    j = json.dumps(temp_cert, indent=4)
    filename = 'certifications_acr/certifications_' + str(n) + '.json'
    f = open(filename, 'w')
    print >> f, j
    f.close()
    n += 1

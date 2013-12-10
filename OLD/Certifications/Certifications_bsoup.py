from bs4 import BeautifulSoup
import urllib2
import json

onets = []
with open("Occupation Data.txt") as f:
    for line in f:
        onets.append(line.split()[0].replace('-', '').replace('.', ''))

onets = onets[1:]

certifications = dict()
certifications['certifications'] = []

index = 0
for onet in onets:
    url = 'http://www.careeronestop.org/WebService/demo.aspx?webService=CertList&onetcode=' + onet
    response = urllib2.urlopen(url)
    html = response.read()

    soup = BeautifulSoup(html)
    table_div = soup.findAll('div', {'id': 'ctl00_ContentPlaceHolder1_results'})[0]

    if table_div.findAll('table'):
        rows = table_div.findAll('tr')

        data = []
        for tr in rows:
            ths = tr.findAll('th')
            tds = tr.findAll('td')

            if ths:
                thead = [th.text for th in ths]

            if tds:
                data.append([td.text for td in tds])

        certs_onet = dict()
        certs_onet['onet'] = onet
        certs_onet['certification_list'] = []

        for row in data:
            cert_dict = dict()
            for i in range(len(thead)):
                cert_dict[thead[i]] = row[i]
            certs_onet['certification_list'].append(cert_dict)

        certifications['certifications'].append(certs_onet)
        print index
        index += 1



print certifications

j = json.dumps(certifications, indent=4)
f = open('certifications1.json', 'w')
print >> f, j
f.close()
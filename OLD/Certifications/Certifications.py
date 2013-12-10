from selenium import webdriver
import json

browser = webdriver.Firefox()
onet = '11101100'
url = 'http://www.careeronestop.org/WebService/demo.aspx?webService=CertList&onetcode=' + onet
browser.get(url)

data=[]
for tr in browser.find_elements_by_xpath('//div[@id="ctl00_ContentPlaceHolder1_results"]/div/table//tr'):
    ths = tr.find_elements_by_tag_name('th')
    tds = tr.find_elements_by_tag_name('td')

    if ths:
        thead = [th.text for th in ths]

    if tds:
        data.append([td.text for td in tds])

certifications = dict()
certifications['certifications'] = []
certifications['certifications'].append({})
certifications['certifications'][0]['onet'] = onet
certifications['certifications'][0]['cert_dict_list'] = []

for row in data:
    cert_dict = dict()
    for i in range(len(thead)):
        cert_dict[thead[i]] = row[i]
    certifications['certifications'][0]['cert_dict_list'].append(cert_dict)


print certifications

j = json.dumps(certifications, indent=4)
f = open('certifications.json', 'w')
print >> f, j
f.close()
browser.quit()
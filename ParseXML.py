from lxml import etree

xml = etree.parse("Resume1.txt")
current_employer = xml.xpath('//job[@end = "present"]/employer/text()')
print current_employer

current_job_title = xml.xpath('//job[@end = "present"]/title/text()')
print current_job_title
import os
import json

source_dir = 'certifications_acr'

files = [ f for (dirpath, dirnames, filenames) in os.walk(source_dir) for f in filenames]
print len(files)

certifications_json = dict()
certifications_json['certifications'] = []

for filename in files:
    jsonc = open(source_dir + '/' + filename)
    jsonc = json.load(jsonc)
    certifications_json['certifications'].append(jsonc)


j = json.dumps(certifications_json, indent=4)
f = open('certifications_final.json', 'w')
print >> f, j
f.close()
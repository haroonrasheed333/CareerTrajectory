import json
from collections import Counter

with open('skills.json') as data:
    skills_json = json.load(data)

skills_json_flat = dict()
skill_features = []
for skill in skills_json:
    skill_list = [item for sl in skills_json[skill] for item in sl]
    skill_count = Counter(skill_list)
    top_40_skills = sorted(skill_count, key=skill_count.get, reverse=True)[:40]
    skills_json_flat[skill] = []
    skills_json_flat[skill] = skill_list
    #skills_json_flat[skill] = top_40_skills
    skill_features += top_40_skills

skills_json_flat = json.dumps(skills_json_flat, indent=4)
filename = 'skills_new.json'
sf = open(filename, 'w')
print >> sf, skills_json_flat
sf.close()

print list(set(skill_features))
print len(list(set(skill_features)))

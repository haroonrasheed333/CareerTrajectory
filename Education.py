import json
import urllib

#api_key = "AIzaSyDJHubLPJMd40pGnXLigHM8tvI4hAs9ZoY"
#service_url = 'https://www.googleapis.com/freebase/v1/mqlread'
#query = [{'id': None, 'name': None, 'type': '/education/educational_institution'}]
#params = {
#        'query': json.dumps(query),
#        'key': api_key
#}
#url = service_url + '?' + urllib.urlencode(params)
#response = json.loads(urllib.urlopen(url).read())
#
#j = json.dumps(response, indent=4)
#f = open('institution.json', 'w')
#print >> f, j
#f.close()

from apiclient import discovery
from apiclient import model
import json
import itertools

DEVELOPER_KEY = 'AIzaSyDJHubLPJMd40pGnXLigHM8tvI4hAs9ZoY'

#model.JsonModel.alt_param = ""
#freebase = discovery.build('freebase', 'v1', developerKey=DEVELOPER_KEY)
query = [{'id': None, 'name': None, 'type': '/education/educational_institution'}]
service_url = 'https://www.googleapis.com/freebase/v1/mqlread'

output = []

index = 0
def do_query(cursor=""):

    #response = json.loads(freebase.mqlread(query=json.dumps(query), cursor=cursor).execute())
    params = {
        'query': json.dumps(query),
        'key': DEVELOPER_KEY,
        'cursor': cursor
    }
    url = service_url + '?' + urllib.urlencode(params)
    try:
        response = json.loads(urllib.urlopen(url).read())
        output.append(response['result'])

        return response.get("cursor")
    except:
        return cursor

cursor = do_query()
while(cursor):
    print index
    index += 1
    cursor = do_query(cursor)

merged = list(itertools.chain.from_iterable(output))

j = json.dumps(merged, indent=4)
f = open('institution.json', 'w')
print >> f, j
f.close()
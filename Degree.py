import urllib
import json
import itertools

DEVELOPER_KEY = 'AIzaSyDJHubLPJMd40pGnXLigHM8tvI4hAs9ZoY'

query = [{'id': None, 'name': None, 'type': '/education/educational_degree'}]
service_url = 'https://www.googleapis.com/freebase/v1/mqlread'

output = []

index = 0
def do_query(cursor=""):

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
f = open('degree.json', 'w')
print >> f, j
f.close()
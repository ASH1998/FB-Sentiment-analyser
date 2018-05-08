import urllib
a = urllib.request.urlopen('http://api.worldbank.org/v2/countries/all/indicators/SP.POP.TOTL?format=json')

import json

b = json.loads(a)
print(b)
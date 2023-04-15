import json
import random

data = ""
with open("UltraSet-L-RAW.json", 'r', encoding='utf-8') as file:
        data = json.load(file)

initdata = len(data)
print(initdata)

import os
import json
import time
from tqdm import tqdm

def check_data(data):
    for d in data:
        try:
            t = (d['input'], d['output'])
        except KeyError:
            print(f"Warning: object {d} does not have 'input' and 'output' keys")

def dedupe(lst):
    seen = set()
    result = []
    ked = 0
    for d in lst:
        t = (d['input'], d['output'])
        if t not in seen:
            seen.add(t)
            result.append(d)
        if(ked%5000):
            print(ked)
        ked+=1
    return result

check_data(data)
# Remove similar objects from the list
data = dedupe(data)





random.shuffle(data)





# Write the shuffled data back to the file
with open('UltraSet-L.json', 'w') as file:
    json.dump(data, file, indent=4)



print(f'Removed {initdata-len(data)} objects.')



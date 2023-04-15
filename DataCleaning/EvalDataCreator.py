import json
import random

import json

with open('UltraSet-Eval.json', 'r') as f:
    data = json.load(f)

with open('UltraSet-Eval.json', 'w') as f:
    json.dump(data, f, indent=4)
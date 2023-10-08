import os
import json
import jsonlines
import random
import numpy as np

incorrect_instances = []
with open("test_outputs.json") as ip_file:
    json_data = json.load(ip_file)
    for json_line in json_data:
        if json_line["label"] != json_line["predicted"]:
            incorrect_instances.append(json_line)

random_samples = []
for i in range(10):
    random_idx = random.randint(0, len(incorrect_instances))
    random_samples.append(incorrect_instances[random_idx])


filename = 'selected_outputs.txt' # give a name
with jsonlines.open(filename, mode='w') as writer:
    for sample in random_samples:
        writer.write(sample)

        
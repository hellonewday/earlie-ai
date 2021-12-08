import json
import pandas as pd
import os

"""
    Since data is take too long to download, maybe a week...I am going to cut down the data with a proper rate, based on nums or urls, nums of data.
"""

train_file = open("data/asl/MSASL_train.json")
train_json = json.load(train_file)

val_file = open("data/asl/MSASL_val.json")
val_json = json.load(val_file)

test_file = open("data/asl/MSASL_test.json")
test_json = json.load(test_file)

classes_file = open('data/asl/MSASL_classes.json')
classes = json.load(classes_file)
train_tbl = pd.json_normalize(train_json)
label_sum = []
for i in range(1, 1000):
    label_sum.append({'label': i, 'urls': len(set(train_tbl[train_tbl['label'] == i]['url'])),
                      'sum': len(train_tbl[train_tbl['label'] == i]), 'class': classes[i - 1]})

newlist = sorted(label_sum, key=lambda d: d['sum'], reverse=True)
sumOfUrlsPerLabel = sum(label['urls'] for label in newlist[:50])
sumOfDataPerLabel = sum(label['sum'] for label in newlist[:50])

# print(newlist[:50])
final_data = newlist[:50]
labels = [d['label'] for d in final_data]
classes = [d['class'] for d in final_data]

new_train_json = [d for d in train_json if d['label'] in labels]
new_test_json = [d for d in test_json if d['label'] in labels]
new_val_json = [d for d in val_json if d['label'] in labels]
print(len(train_json))
print(len(new_train_json))

print(len(test_json))
print(len(new_test_json))

print(len(val_json))
print(len(new_val_json))

with open("data/asl/MSALN_classes.json", "w") as newclasses:
    newclasses.write(json.dumps(classes))

with open("data/asl/MSALN_train.json", "w") as newtrain:
    newtrain.write(json.dumps(new_train_json))

with open("data/asl/MSALN_test.json", "w") as newtest:
    newtest.write(json.dumps(new_test_json))

with open("data/asl/MSALN_val.json", "w") as newval:
    newval.write(json.dumps(new_val_json))
# print(labels)
# print(classes)
# print("Sum of data to process: ", sumOfDataPerLabel)
# print("Sum of url to download", sumOfUrlsPerLabel)
# print()

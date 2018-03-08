import sys
from collections import defaultdict
import IPython as ipy

with open(sys.argv[1],'r') as f:
    predictions = f.readlines()
with open(sys.argv[2],'r') as f:
    gold = f.readlines()

if len(predictions) != len(gold):
    print('help')
    ipy.embed()

label_inventory = set()

for idx in range(len(predictions)):
    pline = predictions[idx].split()
    gline = gold[idx].split()
    for apred in range(14, len(gline)):
        if gline[apred] != '_':
            label_inventory.add(gline[apred])
        if pline[apred] != '_':
            label_inventory.add(pline[apred])
label_inventory.add('_')
label_inventory = sorted(list(label_inventory))

confusionmat = defaultdict(lambda: defaultdict(list))
c = 0.0
g = 0.0
p = 0.0

for idx in range(len(predictions)):
    pline = predictions[idx].split()
    gline = gold[idx].split()
    for apred in range(14, len(gline)):
        gold_label = gline[apred]
        p_label = pline[apred]
        if gold_label != '_' or p_label != '_':
            confusionmat[gold_label][p_label].append((idx, pline[1], pline[4]))
        if gold_label != '_':
            g += 1
        if p_label != '_':            
            p += 1
        if gold_label != '_' and p_label != '_':
            c += 1

print("Unlabeled arg precision: {}/{}={}".format(c,p,c/p))
print("Unlabeled arg recall: {}/{}={}".format(c,g,c/g))
print("Unlabeled arg F1: {}".format(2/(g/c + p/c)))

print('__', end='\t|')
for label in label_inventory:
    print(label, end='\t|')
print('')
for lab1 in label_inventory:
    print(lab1, end='\t|')
    for lab2 in label_inventory:
        print(str(len(confusionmat[lab1][lab2])), end='\t|')
    print('')

ipy.embed()
              

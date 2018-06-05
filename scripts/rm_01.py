import sys

filename = sys.argv[1]

for line in open(filename, 'r'):
    ls = line.split('\t')
    if len(ls) > 13 and ls[12] =='Y' and len(ls[13]) > 3 and ls[13][-3:] == '.01' and '.01' not in ls[1]:
        ls[13] = ls[13][:-3]
        print('\t'.join(ls), end='')
    else:
        print(line, end='')

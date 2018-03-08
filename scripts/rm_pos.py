import sys

filename = sys.argv[1]

for line in open(filename, 'r'):
    ls = line.split('\t')
    if len(ls) > 5:
        ls[4], ls[5] = '_', '_'
        print('\t'.join(ls), end='')
    else:
        print(line, end='')

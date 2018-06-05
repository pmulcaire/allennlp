import sys

for line in open(sys.argv[1]):
    ls = line.strip().split('\t')
    for idx,label in enumerate(ls[14:]):
        if label in ['arg0-agt']:
            ls[14+idx] = 'agent'
        elif label in ['arg1-pat']:
            ls[14+idx] = 'patient'
        elif label in ['arg2-loc', 'arg3-loc']:
            ls[14+idx] = 'place'
        else:
            ls[14+idx] = '_'
        
    print('\t'.join(ls))
        

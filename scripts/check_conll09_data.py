import sys
import IPython as ipy

def get_sentences(filename):
    sentence = {'text':'', 'words':[], 'predicate_indices':[], 'predicate_senses':[], 'args':None}
    lines = open(filename,'r').readlines()
    for line in lines:
        if line.strip() == '':
            #print(sentence['text'])
            yield sentence
            sentence = {'text':'', 'words':[], 'predicate_indices':[], 'predicate_senses':[], 'args':None}
            continue
        fields = line.split()
        idx = int(fields[0])-1
        sentence['words'].append(fields[2]) # gold lemma
        if fields[12] == 'Y':
            sentence['predicate_indices'].append(idx)
            sentence['predicate_senses'].append(fields[13])
        if sentence['args'] is None:
            sentence['args'] = [[] for i in fields[14:]]
        for i,col in enumerate(fields[14:]):
            sentence['args'][i].append(col)
        sentence['text'] += line
        

def main(filename):
    gen = get_sentences(filename)
    for sent in gen:
        text = ' '.join([tok.split(':')[-1] for tok in sent['words']])
        for pnum,pidx in enumerate(sent['predicate_indices']):
            try:
                arg_label = sent['args'][pnum][pidx]
                if arg_label != '_' and arg_label != 'A0':
                    pred_sense = sent['predicate_senses'][pnum]
                    print('Index {} of sentence "{}": self-loop for predicate {} ({} is {} to itself)'.format(pidx, text, pnum, pred_sense, arg_label))
            except:
                print("Error in checking sentence")
                ipy.embed()

if __name__=="__main__":
    filename = sys.argv[1]
    main(filename)

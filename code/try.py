D = []

fname1 = '../glove.6B.300d.txt'
fname2 = '../glove.6B.300d.vocab.txt'

with open(fname1, 'r') as f:
    for line in f:
        line = line.strip().split(' ')
        D.append(line[0])

with open(fname2, 'w') as f:
    for d in D:
        f.write(d+'\n')

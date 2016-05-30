import math

def _get_lyrics(fname):
    lyrics = []
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            lyrics.extend(line)
    lyrics = lyrics[:-9] # getting rid of warning at end of each lyrics sample
    lyrics = [''.join(e.lower() for e in x if e.isalpha()) for x in lyrics] # remove everything that is not a letter and convert all letters to lowercase
    lyrics = [x for x in lyrics if x not in ['']] # remove all invalid words
    return lyrics

def get_wv(self, fname):
    with open(fname, 'r') as f:
        wv = np.zeros((self.total_words, len(f.readline().strip().split(' '))-1))

    c = 0
    d = 0
    with open(fname, 'r') as f:
        for line in f:
            d += 1
            line = line.strip().split(' ')
            try:
                _ = self.word_to_index[line[0]]
            except KeyError:
                continue
            wv[self.word_to_index[line[0]],:] = [float(x) for x in line[1:]]
            c += 1     
            if c==self.total_words:
                break
    print 'Found ', c, 'word vectors in the first', d, 'words.'
    return wv

def _get_word_counts(X):
    word_counts = {}
    for x in X:
        for xx in x:
            for xxx in xx:
                if xxx in word_counts.keys():
                    word_counts[xxx] += 1
                else:
                    word_counts[xxx] = 1
    return word_counts

def check_word(x, t_down, t_up, word_counts, pre_vocab):
    if not t_down <= word_counts[x] <= t_up:
        return False
    try:
        _ = pre_vocab[x]
    except KeyError:
        return False
    return True

def _count_threshold_filter(X, threshold_down, threshold_up, word_counts, pre_vocab):
    for i in range(len(X)):
        X[i][0] = [x for x in X[i][0] if check_word(x, threshold_down, threshold_up, word_counts, pre_vocab)]
        X[i][1] = [x for x in X[i][1] if check_word(x, threshold_down, threshold_up, word_counts, pre_vocab)]
    for key in word_counts.keys():
        if not check_word(key, threshold_down, threshold_up, word_counts, pre_vocab):
            del word_counts[key]
    return X, word_counts

def _add_words(word_counts, x):
    for y in x:
        try:
             _ = word_counts[y]
        except KeyError:
            word_counts[y] = 1
            continue
        word_counts[y] += 1
    return word_counts

def get_data(pair_fname, lyrics_path, wv_fname, num_examples=float('Inf'), threshold_down=-1, threshold_up=float('inf'), score_boundary=0.5, npos=float('inf'), nneg=float('inf')):
    X = []
    y = []
    seq_len1 = []
    seq_len2 = [] 
    c = 0
    cpos = 0
    cneg = 0
    word_counts = {}

    pre_vocab = {}
    with open(wv_fname[:-3]+'vocab.txt', 'r') as f:
        for line in f:
            line = line.strip()
            pre_vocab[line] = 1
    with open(pair_fname, 'r') as f:
        for line in f:
            c += 1
            #if c%1000==0:
            #    print c, cpos, cneg
            line = line.strip().split(',')
            if float(line[2])>score_boundary:
                if cpos==npos:
                    if cneg==nneg:
                        break
                    continue
                y.append(1)
                cpos += 1
            else:
                if cneg==nneg:
                    if cpos==npos:
                        break
                    continue
                y.append(0)
                cneg += 1
            x = [[],[]]
            x[0] = _get_lyrics(lyrics_path+line[0]+'.txt')
            x[1] = _get_lyrics(lyrics_path+line[1]+'.txt')
            word_counts = _add_words(word_counts, x[0])
            word_counts = _add_words(word_counts, x[1])
            X.append(x)
            seq_len1.append(len(x[0]))
            seq_len2.append(len(x[1]))
    #word_counts = _get_word_counts(X)
    X, word_counts = _count_threshold_filter(X, threshold_down, threshold_up, word_counts, pre_vocab)
    max_steps = max([max(len(x[0]), len(x[1])) for x in X])
    return X, y, word_counts, seq_len1, seq_len2, max_steps

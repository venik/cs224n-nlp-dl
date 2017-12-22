#!/usr/bin/env python

# just some fun with info from the lecture 2 of the cs224n
# https://www.youtube.com/watch?v=ASn7ExxLZws&t=3834s&index=3&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6

import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

corpus = []
corpus.append('I like deep learning')
corpus.append('I like NLP')
corpus.append('I enjoy flying')

word_index = {}
uniq_words = []
idx = 0

# build matrix of corpus words
for txt in corpus:
    words = txt.split()
    for i in xrange(0, words.__len__()):
        if words[i] not in word_index:
            print('word: ' + words[i] + ' new index: ' + str(idx))
            word_index[words[i]] = idx
            uniq_words.append(words[i])
            idx += 1

print('================')
occurances = np.zeros(idx * idx).reshape(idx, idx)
# build matrix of occurances - window 1
for txt in corpus:
    words = txt.split()
    print("=> " + txt)
    for i in xrange(0, words.__len__()):
        print('word: ' + words[i] + ' index: ' + str(word_index[words[i]]))
        cur_word_idx = word_index[words[i]]
        if i > 0:
            left_word_idx = word_index[words[i - 1]]
            occurances[cur_word_idx][left_word_idx] += 1

        if i < words.__len__() - 1:
            right_word_idx = word_index[words[i + 1]]
            occurances[cur_word_idx][right_word_idx] += 1
        
        # print('' + str(occurances))

print ('' + str(occurances))

# factorization
U, s, Vh = linalg.svd(occurances, full_matrices=False)
for i in xrange(len(uniq_words)):
    plt.text(U[i, 0], U[i, 1], uniq_words[i])

# plt.axis([-10, -10, 10, 10])
plt.grid()
plt.ylim([-0.05, 0.95])
plt.xlim([-1, 0.2])
plt.show()
from collections import Counter, OrderedDict
from itertools import product
import matplotlib.pyplot as plt
from random import choices

import numpy as np
import string
import sys
import re
import math

np.set_printoptions(suppress = True)

# in this piece of code, I leave out a bunch of thing for you to fill up modify.
# The current code may run into a ZeroDivisionError. Thus, you need to add Laplace first.
'''
Todo: 
1. Laplace smoothing
2. Naive Bayes prediction
3. All the output.

'''
def ceiling(number, decimal):
    if number >= 0.1 ** decimal:
        return math.ceil(number * (10 ** decimal)) / (10 ** decimal)
    else: return round(0.1 ** decimal,decimal)

with open('Logan.txt', encoding='utf-8') as f:
    data = f.read()
# len(data)

data = data.lower()
data = data.translate(str.maketrans('', '', string.punctuation))
data = re.sub('[^a-z]+', ' ', data)
data = ' '.join(data.split(' '))

# f = open("output.txt", "w+")
# f.write(str(data))
# f.close()

allchar = ' ' + string.ascii_lowercase
# print(len(data))

unigram = Counter(data)
# print(unigram["c"])
unigram_prob = {ch: round((unigram[ch]) / (len(data)), 4) for ch in allchar}

uni_list = [unigram_prob[c] for c in allchar]

f2 = open("p3q2.txt", "w+")
f2.write(str(uni_list[0]))
for i in range(1, len(uni_list)):
    f2.write(",")
    f2.write(str(uni_list[i]))
f2.close()

def ngram(n):
    # all possible n-grams
    d = dict.fromkeys([''.join(i) for i in product(allchar, repeat=n)],0)
    # update counts
    d.update(Counter([''.join(j) for j in zip(*[data[i:] for i in range(n)])]))
    return d

bigram = ngram(2)  # c(ab)

bigram_prob = {c: bigram[c] / unigram[c[0]] for c in bigram}  # p(b|a)
bigram_prob_smooth = {c: (bigram[c]+1) / (unigram[c[0]]+27) for c in bigram}

bi_list = [round(bigram_prob[c],4) for c in bigram_prob]
bi_list_smooth = [ceiling(bigram_prob_smooth[c],4) for c in bigram_prob_smooth]

bi_matrix = np.asarray(bi_list).reshape(27, 27)
bi_matrix_smooth = np.asarray(bi_list_smooth).reshape(27,27)
# for c in bi_matrix_smooth:
#     print(sum(c))

f3 = open("p3q3.txt", "w+")
f4 = open("p3q4.txt", "w+")

for i in range(0,27):
    f3.write(str(bi_matrix[i][0]))
    f4.write(str(bi_matrix_smooth[i][0]))
    for j in range(1,27):
        f3.write(",")
        f3.write(str(bi_matrix[i][j]))
        f4.write(",")
        f4.write(str(bi_matrix_smooth[i][j]))
    if i != 26:
        f3.write("\n")
        f4.write("\n")
f3.close()
f4.close()

trigram = ngram(3)
trigram_prob_smooth = {c: (trigram[c]+1) / (bigram[c[:2]]+27) for c in trigram}


def gen_bi(c):
    w = [bigram_prob_smooth[c + i] for i in allchar]
    return choices(allchar, weights=w)[0]
    

def gen_tri(ab):
    w_tri = [trigram_prob_smooth[ab + i] for i in allchar]
    return choices(allchar, weights=w_tri)[0]   


def gen_sen(c, num):
    res = c + gen_bi(c)
    for i in range(num - 2):
        if bigram[res[-2:]] == 0:
            t = gen_bi(res[-1])
        else:
            t = gen_tri(res[-2:])
        res += t
    return res


example_sentence = gen_sen('h', 100)
print(example_sentence)
# print(allchar)
f5 = open("p3q5.txt", "w+")
for i in allchar:
    if i != " ":
        f5.write(gen_sen(i, 1000))
    if i != "z" and i != " ":
        f5.write("\n")
f5.close()

with open('script.txt', encoding='utf-8') as f:
    young = f.read() 

dict2 = Counter(young)
likeli = [dict2[c] / len(young) for c in allchar]
f7 = open("p3q7.txt", "w+")
f7.write(str(round(likeli[0], 4)))
for i in range(1, len(uni_list)):
    f7.write(",")
    f7.write(str(round(likeli[i], 4)))
f7.close()

post_young = [round(likeli[i] / (likeli[i] + uni_list[i]), 4) for i in range(27)]

post_hugh = [1 - post_young[i] for i in range(27)]

f8 = open("p3q8.txt", "w+")
f8.write(str(round(post_young[0], 4)))
for i in range(1, len(uni_list)):
    f8.write(",")
    f8.write(str(round(post_young[i], 4)))
f8.close()


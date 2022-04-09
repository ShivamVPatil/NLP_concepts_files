# -*- coding: utf-8 -*-

!pip install gensim

import nltk 
from gensim.models import Word2Vec
from nltk.corpus import stopwords

nltk.download('punkt')

nltk.download('stopwords')

import re

para = """ It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout
. The point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters, as opposed to using 'Content here,
 content here', making it look like readable English. Many desktop publishing packages and web page editors now use Lorem Ipsum as 
 their default model text, and a search for 'lorem ipsum' will uncover many web sites still in their infancy. Various versions have
  evolved over the years, sometimes by accident, sometimes on purpose (injected humour and the like)."""

test = re.sub(r'\[[0-9]*\]',' ',para)

test = re.sub(r'\s+',' ',test)
test = test.lower()
test = re.sub(r'\d',' ',test)
test = re.sub(r'\s+',' ',test)

sentence = nltk.sent_tokenize(para)

sentence = [nltk.word_tokenize(sentence) for sentence in sentence]

for i in range(len(sentence)):
  sentence[i] = [word for word in sentence[i] if word not in stopwords.words('english')]

model = Word2Vec(sentence,min_count = 1 )

words = model.wv.vocab

print(words)

vectore = model.wv.most_similar('reader')

print(vectore)


import nltk

para = """ It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout
. The point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters, as opposed to using 'Content here,
 content here', making it look like readable English. Many desktop publishing packages and web page editors now use Lorem Ipsum as 
 their default model text, and a search for 'lorem ipsum' will uncover many web sites still in their infancy. Various versions have
  evolved over the years, sometimes by accident, sometimes on purpose (injected humour and the like)."""

#Cleaning text or preprocessing

import re #Regular Expression Library
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
lem = WordNetLemmatizer()
sentence = nltk.sent_tokenize(para)
corpus = []
lemma = []

 # Using Stemmer
for i in range(len(sentence)):
    review = re.sub('[^a-zA-Z]', ' ', sentence[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

#Using Lemmatizer
for i in range(len(sentence)):
    review1 = re.sub('[^a-zA-Z]', ' ', sentence[i])
    review1 = review1.lower()
    review1 = review1.split()
    review1 = [lem.lemmatize(word) for word in review1 if word not in set(stopwords.words('english'))]
    review1 = ' '.join(review1)
    lemma.append(review1)


from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
a = cv.fit_transform(corpus).toarray()
b = cv.fit_transform(lemma).toarray()
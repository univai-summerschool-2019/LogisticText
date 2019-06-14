# modified from https://github.com/amueller/scipy-2018-sklearn/blob/master/notebooks/15.Pipelining_Estimators.ipynb

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

lz = lambda x, y: list(zip(x,y))

data = Path('./data')
with open(data/"SMSSpamCollection.txt") as f:
    lines = [line.strip().split("\t") for line in f.readlines()]
text = [x[1] for x in lines]
y = [x[0] == "ham" for x in lines]
text_train, text_test, y_train, y_test = train_test_split(text, y, random_state=2017)

# This illustrates a common mistake. Don't use this code!

vectorizer = CountVectorizer()
vectorizer.fit(text_train)
vocab = vectorizer.vocabulary_
ivocab = dict(map(reversed, vocab.items()))
X_train = vectorizer.transform(text_train)
X_test = vectorizer.transform(text_test)
for text, label in lz(text_test[:2], y_test[:2]):
    print("<<",text,">>", label)
c = 0
for stuff in X_test[:2]: 
    for csr in stuff[0,:]:
        for tup in lz([str(e)+" "+ivocab[e] for e in csr.indices], csr.data):
            print(c, tup[0], tup[1])
    c+=1

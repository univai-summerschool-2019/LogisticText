# modified from https://github.com/amueller/scipy-2018-sklearn/blob/master/notebooks/15.Pipelining_Estimators.ipynb

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression

lz = lambda x, y: list(zip(x,y))

data = Path('./data')
with open(data/"SMSSpamCollection.txt") as f:
    lines = [line.strip().split("\t") for line in f.readlines()]
text = [x[1] for x in lines]
y = [x[0] == "ham" for x in lines]
text_train, text_test, y_train, y_test = train_test_split(text, y, random_state=2017)


from sklearn.model_selection import GridSearchCV

pipeline = make_pipeline(TfidfVectorizer(), 
                         LogisticRegression())

grid = GridSearchCV(pipeline,
                    param_grid={'logisticregression__C': [.1, 1, 10, 100]}, cv=5)

grid.fit(text_train, y_train)
print("Score",grid.score(text_test, y_test))
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import pickle
from tqdm import tqdm
print("import complete...")

def create_word_features(words):
    useful = [word for word in words if word not in stopwords.words('english')]
    use_dict = dict([(word, True) for word in useful])
    return use_dict

negative_reviews = []
for fileid in tqdm(movie_reviews.fileids('neg')):
    words = movie_reviews.words(fileid)
    negative_reviews.append((create_word_features(words), "negative"))

print("compiled negative reviews", len(negative_reviews))

positive_reviews = []
for fileid in tqdm(movie_reviews.fileids('pos')):
    words = movie_reviews.words(fileid)
    negative_reviews.append((create_word_features(words), "positive"))

print("compiled positive reviews", len(positive_reviews))

train_set = negative_reviews[:750] + positive_reviews[:750]
test_set = negative_reviews[750:] + positive_reviews[750:]

classifier = NaiveBayesClassifier.train(train_set)

accuracy = nltk.classify.util.accuracy(classifier, test_set)
print(accuracy)

with open('classifier_nltk2.pkl', 'wb') as fopen:
    pickle.dump(classifier, fopen)

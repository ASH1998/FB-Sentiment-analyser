import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

with open('classifier_nltk2.pkl' , 'rb') as fopen:
    clf = pickle.load(fopen)

review1 = ''' its the best one'''


def create_word_features(words):
    useful_words = [word for word in words if word not in stopwords.words("english")]
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict
word1 = word_tokenize(review1)
words = create_word_features(word1)
print(words)
print(clf.classify(words))

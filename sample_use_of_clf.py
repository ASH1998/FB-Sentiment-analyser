import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

with open('classifier_nltk2.pkl' , 'rb') as fopen:
    clf = pickle.load(fopen)

review1 = ''' its the best one'''
review2 = "its the not best one"
review3 = "it may be good or bad, i dont know"
review4 = "i dont know"

def create_word_features(words):
    useful_words = [word for word in words if word not in stopwords.words("english")]
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict


word1 = word_tokenize(review1)
word2 = word_tokenize(review2)
word3 = word_tokenize(review3)
word4 = word_tokenize(review4)

all_words = [word1, word2, word3, word4]
created_dict = []
for i in range(len(all_words)):
    words = create_word_features(all_words[i])
    print(clf.classify(words))

print(" " )
words = create_word_features(word1)
print(clf.classify(words))

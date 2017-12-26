import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
 
def word_feats(words):
    return dict([(word, True) for word in words])
 
negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')
 
negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]
 
negcutoff = (len(negfeats)*3)/4
poscutoff = (len(posfeats)*3)/4
 
trainfeats = negfeats[:750] + posfeats[:750]
testfeats = negfeats[750:] + posfeats[750:]
 
classifier = NaiveBayesClassifier.train(trainfeats)
print ('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
classifier.show_most_informative_features()

r = "this movie was best"
w = word_tokenize(r)
wo = word_feats(w)
print(classifier.classify(wo))
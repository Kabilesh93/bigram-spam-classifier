# bigram-spam-classifier <br />
A bigram approach for classifying Spam and Ham messages

#install with pip <br />
pip install bigram-spam-classifier

#import in your python file <br />
from bigram_spam_classifier import spamclassifier

#create an object of the classifier and pass your message as the parameter <br />
classifier = spamclassifier.classifier("Customer service annoncement. You have a New Years delivery waiting for you. Please call 07046744435 now to arrange delivery")

#classify the message<br />
cls = classifier.classify()

print(cls)

#find the unigrams and bigrams in the message <br />
unigrams = classifier.inputUnigrams

print(unigrams)

bigrams = classifier.inputBigrams

print(bigrams)

#find the bigram probabilities of Spam and Ham <br />
spam_probability = classifier.bigramPSpam

print(spam_probability)

ham_probability = classifier.bigramPHam

print(ham_probability)

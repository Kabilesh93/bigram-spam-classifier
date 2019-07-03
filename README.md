**# bigram-spam-classifier <br />**
**# A bigram approach for classifying Spam and Ham messages**<br />

**# install with pip <br />**
pip install bigram-spam-classifier<br />

**# import in your python file <br />**
from bigram_spam_classifier import spamclassifier<br />

**# create an object of the classifier and pass your message as the parameter <br />**
classifier = spamclassifier.classifier("Customer service annoncement. You have a New Years delivery waiting for you. Please call 07046744435 now to arrange delivery")<br />

**# classify the message<br />**
 
cls = classifier.classify()<br />
print(cls)<br />

**# If returns 1 message is a Spam, if returns 0 message is a Ham <br />**

**# find the unigrams and bigrams in the message <br />**
unigrams = classifier.inputUnigrams<br />
print(unigrams)<br />
bigrams = classifier.inputBigrams<br />
print(bigrams)<br />

**# find the bigram probabilities of Spam and Ham <br />**
spam_probability = classifier.bigramPSpam<br />
print(spam_probability)<br />
ham_probability = classifier.bigramPHam<br />
print(ham_probability)<br />

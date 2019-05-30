import string
import ast
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk import sent_tokenize
from nltk import ngrams

from calculate_prob import *

stop = stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()


# sentence tokenizing and removing punctuation
def tokenizeSentences(fullCorpus):
    punctuations = string.punctuation
    punctuations = punctuations.replace(',','')
    sent_tokenized = fullCorpus['body_text'].apply(sent_tokenize)
    f = lambda sent: ''.join(ch for w in sent for ch in w
                                                  if ch not in string.punctuation)

    sent_tokenized = sent_tokenized.apply(lambda row: list(map(f, row)))
    return sent_tokenized


# Converting to lowercase
def toLowercase(fullCorpus):
   lowerCased = fullCorpus['sentTokenized'].astype(str).str.lower().transform(ast.literal_eval)
   return lowerCased


# Removing stopwords
def removeStop(fullCorpus):
    stopwordsRemoved = fullCorpus['lowerCased'].astype(str).apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])).transform(ast.literal_eval)
    return stopwordsRemoved


# Tokenizing sentences into words
def tokenize(fullCorpus):
    tokenize = nltk.word_tokenize
    tokenized = fullCorpus['stopwordsRemoved'].apply(lambda row:list(map(tokenize, row)))
    return tokenized


# Lemmatizing words
def lemmatize(fullCorpus):
    lemmatized = fullCorpus['tokenized'].apply(
            lambda row: list(list(map(lemmatizer.lemmatize,y)) for y in row))
    return lemmatized


# Creating bigrams from words
def toBigram(fullCorpus):
    bigram = fullCorpus['lemmatized'].apply(
        lambda row: list(map(lambda x: list(ngrams(x, 2)), row)))
    return bigram


# Joining the lists that contain unigrams
def toFlatListUnigram(fullCorpus):
    flatListUnigram = fullCorpus['lemmatized'].apply(
        lambda row: [item for sublist in row for item in sublist])
    return flatListUnigram


# Joining the lists that contain bigrams
def toFlatListBigram(fullCorpus):
    flatListBigram = fullCorpus['bigrams'].apply(
        lambda row: [item for sublist in row for item in sublist])
    return flatListBigram


# Preprocessing the input text
inputbigrams = []
inputunigrams = []


def processInputText(inputText):
    sentences = sent_tokenize(inputText)
    for x in sentences:
        punctRemoved = nltk.re.sub(r'[^\w\s]', '', x)

        sentencesLower = punctRemoved.lower()

        sentencesTokenized = nltk.word_tokenize(sentencesLower)
        sentencesNonStop = [x for x in sentencesTokenized if x != []]
        LemmatizedWords = []
        for x in sentencesNonStop:
            LemmatizedWords.append(lemmatizer.lemmatize(x))

        unigram = LemmatizedWords
        bigram = list(ngrams(LemmatizedWords, 2))
        inputunigrams.append(unigram)
        inputbigrams.append(bigram)


def main():
    fullCorpus = pd.read_csv("spam_collection.csv", sep="\t", header=None)
    fullCorpus.columns = ["lable", "body_text"]

    fullCorpus['sentTokenized'] = tokenizeSentences(fullCorpus)
    fullCorpus['lowerCased'] = toLowercase(fullCorpus)
    fullCorpus['stopwordsRemoved'] = removeStop(fullCorpus)
    fullCorpus['tokenized'] = tokenize(fullCorpus)
    fullCorpus['lemmatized'] = lemmatize(fullCorpus)
    fullCorpus['bigrams'] = toBigram(fullCorpus)
    fullCorpus['unigrams_flattern'] = toFlatListUnigram(fullCorpus)
    fullCorpus['bigrams_flattern'] = toFlatListBigram(fullCorpus)

    unigramCorpus = fullCorpus.groupby('lable').agg({'unigrams_flattern': 'sum'})
    bigramCorpus = fullCorpus.groupby('lable').agg({'bigrams_flattern': 'sum'})

    inputText = input("Please enter something: ")
    processInputText(inputText)

    inputUnigrams = [item for sublist in inputunigrams for item in sublist]
    inputBigrams = [item for sublist in inputbigrams for item in sublist]

    print("Unigrams from your input\n", inputUnigrams)
    print("\n")
    print("Bigrams from your input\n", inputBigrams)
    print("\n")

    # Call calculateBigramProbability method in CalculateProb file
    bigramPSpam, bigramPHam = calculateBigramProbability(unigramCorpus, bigramCorpus, inputUnigrams, inputBigrams)

    print("bigram probability for Spam \n", bigramPSpam)
    print("\n")
    print("bigram probability for Ham \n", bigramPHam)
    print("\n")

    if(bigramPSpam > bigramPHam):
        print("Messeage You entered is a Spam!!!")
    else:
        print("Messeage You entered is a Ham :)")


if __name__ == '__main__':
    main()

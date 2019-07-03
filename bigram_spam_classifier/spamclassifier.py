import string
import ast
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk import sent_tokenize
from nltk import ngrams
import os

stop = stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()


class classifier:

    def __init__(self, message):
        self.message = message
        self.inputUnigrams = ""
        self.inputBigrams = ""
        self.bigramPSpam = 1
        self.bigramPHam = 1

    def classify(self):

        this_dir, this_filename = os.path.split(__file__)
        DATA_PATH = os.path.join(this_dir, "data", "spam_collection.csv")

        fullCorpus = pd.read_csv(DATA_PATH, sep="\t", header=None)
        fullCorpus.columns = ["lable", "body_text"]

        fullCorpus['sentTokenized'] = self.__tokenizeSentences(fullCorpus)
        fullCorpus['lowerCased'] = self.__toLowercase(fullCorpus)
        fullCorpus['stopwordsRemoved'] = self.__removeStop(fullCorpus)
        fullCorpus['tokenized'] = self.__tokenize(fullCorpus)
        fullCorpus['lemmatized'] = self.__lemmatize(fullCorpus)
        fullCorpus['bigrams'] = self.__toBigram(fullCorpus)
        fullCorpus['unigrams_flattern'] = self.__toFlatListUnigram(fullCorpus)
        fullCorpus['bigrams_flattern'] = self.__toFlatListBigram(fullCorpus)

        unigramCorpus = fullCorpus.groupby('lable').agg({'unigrams_flattern': 'sum'})
        bigramCorpus = fullCorpus.groupby('lable').agg({'bigrams_flattern': 'sum'})

        self.__processInputText(self.message)

        self.inputUnigrams = [item for sublist in self.__inputunigrams for item in sublist]
        self.inputBigrams = [item for sublist in self.__inputbigrams for item in sublist]

        # Call calculateBigramProbability method in CalculateProb file
        self.bigramPSpam, self.bigramPHam = self.__calculateBigramProbability(unigramCorpus, bigramCorpus, self.inputUnigrams, self.inputBigrams)

        if self.bigramPSpam > self.bigramPHam:
            return 1
        else:
            return 0

    # sentence tokenizing and removing punctuation
    def __tokenizeSentences(self, fullCorpus):
        punctuations = string.punctuation
        punctuations = punctuations.replace(',','')
        sent_tokenized = fullCorpus['body_text'].apply(sent_tokenize)
        f = lambda sent: ''.join(ch for w in sent for ch in w
                                                      if ch not in string.punctuation)

        sent_tokenized = sent_tokenized.apply(lambda row: list(map(f, row)))
        return sent_tokenized

    # Converting to lowercase
    def __toLowercase(self, fullCorpus):
       lowerCased = fullCorpus['sentTokenized'].astype(str).str.lower().transform(ast.literal_eval)
       return lowerCased

    # Removing stopwords
    def __removeStop(self, fullCorpus):
        stopwordsRemoved = fullCorpus['lowerCased'].astype(str).apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])).transform(ast.literal_eval)
        return stopwordsRemoved

    # Tokenizing sentences into words
    def __tokenize(self, fullCorpus):
        tokenize = nltk.word_tokenize
        tokenized = fullCorpus['stopwordsRemoved'].apply(lambda row:list(map(tokenize, row)))
        return tokenized

    # Lemmatizing words
    def __lemmatize(self, fullCorpus):
        lemmatized = fullCorpus['tokenized'].apply(
                lambda row: list(list(map(lemmatizer.lemmatize,y)) for y in row))
        return lemmatized

    # Creating bigrams from words
    def __toBigram(self, fullCorpus):
        bigram = fullCorpus['lemmatized'].apply(
            lambda row: list(map(lambda x: list(ngrams(x, 2)), row)))
        return bigram

    # Joining the lists that contain unigrams
    def __toFlatListUnigram(self, fullCorpus):
        flatListUnigram = fullCorpus['lemmatized'].apply(
            lambda row: [item for sublist in row for item in sublist])
        return flatListUnigram

    # Joining the lists that contain bigrams
    def __toFlatListBigram(self, fullCorpus):
        flatListBigram = fullCorpus['bigrams'].apply(
            lambda row: [item for sublist in row for item in sublist])
        return flatListBigram

    __inputbigrams = []
    __inputunigrams = []

    def __processInputText(self, inputText):
        sentences = sent_tokenize(inputText)
        # Preprocessing the input text

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
            self.__inputunigrams.append(unigram)
            self.__inputbigrams.append(bigram)

    def __calculateBigramProbability(self, unigramCorpus, bigramCorpus, inputUnigrams, inputBigrams):

        unigramCount = unigramCorpus.assign(count=unigramCorpus.unigrams_flattern.apply(lambda x: len(set(x))))
        V_Spam = unigramCount.at['spam', 'count']
        V_Ham = unigramCount.at['ham', 'count']

        bigramPSpam = 1
        bigramPHam = 1

        # Calculating bigram probability using Spam forpus
        for x in range(len(inputBigrams)-1):
            bigramPSpam *= (((bigramCorpus.loc["spam", "bigrams_flattern"].count(inputBigrams[x])) + 1) / (
                (unigramCorpus.loc["spam", "unigrams_flattern"].count(inputUnigrams[x]) + V_Spam)))

        # Calculating bigram probability using Ham forpus
        for x in range(len(inputBigrams)-1):
            bigramPHam *= (((bigramCorpus.loc["ham", "bigrams_flattern"].count(x)) + 1) / (
                (unigramCorpus.loc["ham", "unigrams_flattern"].count(inputUnigrams[x]) + V_Ham)))

        return (bigramPSpam, bigramPHam)

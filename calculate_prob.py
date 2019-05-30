
def calculateBigramProbability(unigramCorpus, bigramCorpus, inputUnigrams, inputBigrams):

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

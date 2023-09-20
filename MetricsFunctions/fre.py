
import pandas as pd
import nltk
from nltk.corpus import cmudict
from nltk.tokenize import word_tokenize
import string
from nltk.tokenize import sent_tokenize
import syllables as sylCount
import numpy  as np
from tqdm.auto import tqdm


# Flesch Reading Ease score
#requirement: “text”
def prepare_download():
    nltk.download('punkt')
    nltk.download('cmudict')
    d = cmudict.dict()


# the fre_score dictionary corresponding to the input
# RETURN the list of values
def FREReadabilityScore(text_list):
    '''
    Params:
        text_list: the list of text for calculation
    Return:
        lists of value: freReadabilityScore
    Usage:
        perplexity = FREReadabilityScore(text_list)
    '''
    prepare_download()
    fre_result = {}
    for i in range(len(text_list)):
        text = text_list[i]
        # conut sentences
        sentences = sent_tokenize(text)
        sentence_count = len(sentences)

        # count words without punctuation marks
        text = text.translate(str.maketrans("", "", string.punctuation))
        words = word_tokenize(text)
        word_count = len(words)

        # count syllables without punctuation marks
        syl_count = 0
        for word in words:
            try:
                syllables = [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]]
                syl_count += max(syllables)
            except KeyError:
                syllables = sylCount.estimate(word)
                syl_count += syllables
            except Exception:
                syl_count += 1

        # calculate score
        score = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syl_count / word_count)

        # return (word_count, sentence_count, syl_count, max(0, min(score, 100)))
        fre_result[i] = max(0, min(score, 100))
    
    return list(fre_result.values())







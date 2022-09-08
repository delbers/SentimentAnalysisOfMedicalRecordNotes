# -*- coding: utf-8 -*-
"""
Purpose of this script is to establish the frequency of words in the medical 
dictionary that are not present in the word score list and find words that are 
scored 'wrongly' by finding their surrounding sentiment score. 
"""

from nltk import FreqDist
import numpy as np
import pandas as pd
import re
import string


def clean_note(text, stopwords):
    
    words = re.split(r'\W+', text)
    table = str.maketrans('', '', string.punctuation)
    words = [w.translate(table) for w in words]  #clean out punctutation
    words = [word.lower() for word in words if word.isalpha()]  #keep only words, not numbers & send to lower case
    words = [word for word in words if word not in stopwords] #validate for stop words

    return words

def filterTheDict(dictObj, callback):
    '''
    function adapted from https://thispointer.com/python-filter-a-dictionary-by-conditions-on-keys-or-values/

    '''
    newDict = dict()
    # Iterate over all the items in dictionary
    for (key, value) in dictObj.items():
        # Check if item satisfies the given condition then add to new dict
        if callback((key, value)):
            newDict[key] = value
    return newDict

def select_words(freq, n):
    '''returns n highest frequency words'''
    
    l = list(freq.values())
    l.sort(reverse=True)
    value = l[n]
    select_dict = filterTheDict(freq, lambda elem: elem[1] > value)
    return select_dict

def find_surrounding_words(window, ii, text):
    
    words = []
    for elem in ii:

        words += list(text[elem-window:elem])
        words += list(text[elem:elem+window])

    return words

def calculate_score(words, score_list):
    
    score = []
    count = 0
    freq = FreqDist(words)
    
    for w in set(words):
        try:
            w_score = float(score_list[score_list['Word'] == w]['Happiness Score'])
            score += [w_score] * freq[w]
            count += freq[w]
        except:
            continue
    return score, count
    
def surrounding_sentiment(word_list, window, text, score_list, alt_score_list):
    '''
    Parameters
    ----------
    word_list : words to score surrounding sentiment for
    window : sentiment window, symmetrical
    text : notes
    score_list : all words
    alt_score_list : words to use for sentiment scoring

    Returns
    -------
    ss_score : DataFrame

    '''
    
    ss_score = score_list[score_list['Word'].isin(word_list)].reset_index(drop=True)
    ss_score['SS_Mean'] = 0
    ss_score['SS_Std'] = 0
    ss_score['N_SS_Words'] = 0
    text = np.array(text)
    elapsed = 0
    for w in list(ss_score['Word']):
        ii = list(np.where(text == w))[0]
        
        surrounding_w = find_surrounding_words(window, ii, text)   
        surrounding_scores, n_words_with_score = calculate_score(surrounding_w, alt_score_list)
        if len(surrounding_scores) > 0:
           
            ss_score.loc[ss_score['Word'] == w, 'SS_Mean'] = np.mean(surrounding_scores)
            ss_score.loc[ss_score['Word'] == w, 'SS_Std'] = np.std(surrounding_scores)
            ss_score.loc[ss_score['Word'] == w, 'N_SS_Words'] = n_words_with_score
        elapsed += 1
        print(elapsed/len(list(ss_score['Word'])))  #keep an eye on progress  
    return ss_score

def lens(df, upper_bound, lower_bound):
    '''returns dataframe without the lens''' 
    
    high_df = df[df['Happiness Score'] >= upper_bound]
    low_df = df[df['Happiness Score'] <= lower_bound]
    keep_df = pd.concat([high_df, low_df], axis = 0)
    return keep_df

stopwords = pd.read_csv('stopwords.csv')
hedonometer_words = pd.read_csv('Hedonometer.csv')
hedonometer_words_list = list(hedonometer_words['Word'])
data = pd.read_csv('Notes.csv').dropna() 
h_words_lens_list = []

#list of bland words to leave out of analysis
lower_bound = 4
upper_bound = 6
h_words_lens = lens(hedonometer_words, upper_bound, lower_bound)
h_words_lens_list = list(h_words_lens['Word'])

#transform notes into one big text
text = ''.join(list(data['ReportText']))
text = clean_note(text, list(stopwords['0']))

#find the >n occuring words
n = 500
freq = FreqDist(text)
words_high_freq = select_words(freq, n)
list_words_high_freq = list(words_high_freq.keys())

#find high freq med words missing in hedonometer corpus
med_words_missing = set(list_words_high_freq) - set(hedonometer_words_list)
med_words_missing_dict = filterTheDict(words_high_freq, lambda elem: elem[0] in med_words_missing)
med_words_missing_df = pd.DataFrame.from_dict(med_words_missing_dict, orient = 'index', columns = ['freq'])\
    .sort_values(by = ['freq'], ascending = False)
med_words_missing_df.to_csv('missing_words_frequency.csv')

#find surrounding sentiment for high freq med words not in med_words_missing
med_words_eval = set(words_high_freq) - set(med_words_missing) 
med_words_eval_dict = filterTheDict(words_high_freq, lambda elem: elem[0] in med_words_eval)

window = 5
sentiment = surrounding_sentiment(med_words_eval, window, text
                                  , score_list = hedonometer_words[['Word', 'Happiness Score']]
                                  , alt_score_list = h_words_lens[['Word', 'Happiness Score']])

sentiment.to_csv('sentiment_lens4-6.csv')

#word coverage hedonometer
n_words = len(text)
text_scored = [x for x in text if x in hedonometer_words_list]
word_coverage_hedonometer = np.round(len(text_scored)/n_words, 2)
print('word coverage for hedonemeter words in notes: ' +str(word_coverage_hedonometer))

#creating clean dataset for plotting etc.
words_high_freq_df = pd.DataFrame.from_dict(words_high_freq, orient = 'index', columns = ['freq'])\
    .sort_values(by = ['freq'], ascending = False)
words_high_freq_df['coverage'] = words_high_freq_df['freq'] / n_words
words_high_freq_df['missing'] = 0
for w in med_words_missing:
    words_high_freq_df.loc[w, 'missing'] = 1
words_high_freq_df.to_csv('word_coverage.csv')




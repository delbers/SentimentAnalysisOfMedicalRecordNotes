import pyodbc
import pandas as pd
import re
import string
from nltk import FreqDist


def connect_cdw():
    """Runs a query against the VA's CDW and returns the 
       result set as a list of of tuples."""
    # This was the connection string for the VA CDW, update to use your own 
    connection = pyodbc.connect('Driver=driver; Server=server; Database=database; Trusted_Connection=tc;')    

    cursor = connection.cursor()

    return cursor

def clean_note(text, stopwords):
    '''cleans up notes for analysis'''
    
    words = re.split(r'\W+', text)
    table = str.maketrans('', '', string.punctuation)
    words = [w.translate(table) for w in words]  #clean out punctutation
    words = [word.lower() for word in words if word.isalpha()]  #keep only words, not numbers & send to lower
    words = [word for word in words if word not in stopwords] #validate for stop words

    return words

def lens(df, upper_bound, lower_bound):
    '''returns dataframe without words in the lens''' 
    
    high_df = df[df['Happiness Score'] >= upper_bound]
    low_df = df[df['Happiness Score'] <= lower_bound]
    keep_df = pd.concat([high_df, low_df], axis = 0)
    return keep_df

def score(text, words):
    ''' calculates text score based on given words with happiness score'''

    freq_dist = FreqDist(text)
    text_df = pd.DataFrame.from_dict(freq_dist, orient = 'index', columns = ['Count'])
    text_df.reset_index(inplace = True)
    text_df = text_df.rename(columns = {'index':'Word'})
    all_df = text_df.merge(words[['Word', 'Happiness Score']], on = 'Word', how = 'left').dropna()

    if len(all_df) > 0 :
        score  = sum(all_df['Happiness Score']*all_df['Count'])/sum(all_df['Count'])
        return score
    else:
        return None

#example query, should be tailored to your database / tables
input_query = "SELECT TOP 1 ReportText \
      ,TIUDocumentSID \
      ,TIUStandardTitle \
	  ,EntryDateTime \
      FROM Table \
      WHERE S.TIUStandardTitle IS NULL"

#load data
stopwords = pd.read_csv('stopwords.csv')
onc_hedonometer_words = pd.read_csv('Oncology_Hedonometer.csv')
hedonometer_words = pd.read_csv('Hedonometer.csv')
onc_only_hedonometer_words = pd.read_csv('Oncology_Only_Hedonometer.csv')

#create lens for everything
lower_bound = 4
upper_bound = 6
all_words_lens = lens(onc_hedonometer_words, upper_bound, lower_bound)

#create lens for original words, but not oncology words
h_words_lens = lens(hedonometer_words, upper_bound, lower_bound)
h_words_lens.drop(['Rank', 'Word in English'], inplace = True, axis = 1)
orig_words_only_lens = h_words_lens.append(onc_only_hedonometer_words)
orig_words_only_lens.drop_duplicates(subset = 'Word', keep = 'last', inplace = True)

cursor = connect_cdw()
cursor.execute(input_query)
result = cursor.fetchall()

while bool(result):

    #unpacking values from query
    for row in result:
        data = [r for r in row]

    text = data[0]
    tiudocumentsid = data[1]
    doctitle = data[2]
    datetime = data[3]

    #clean text
    if bool(text):
        text = clean_note(text, list(stopwords['0']))

        #calculate note score
        all_score = score(text, all_words_lens)
        orig_score = score(text, orig_words_only_lens)
        print(all_score, orig_score)
    else:
        all_score = None
        orig_score = None

#example query, should be tailored to your database / tables
    insert_query = "UPDATE TABLE \
    SET OncologyHedonometerScore = ? \
    , OncologyHedonometerScore_LensAll = ? \
    , TIUStandardTitle = ?\
    , EntryDateTime = ? \
    WHERE TIUDocumentSID = ?"

    cursor.execute(insert_query, orig_score, all_score, doctitle, datetime, tiudocumentsid)
    cursor.commit()

    cursor.execute(input_query)
    result = cursor.fetchall()









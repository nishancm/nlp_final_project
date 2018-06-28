import pandas as pd
import gensim
import itertools
import keras.backend as K
import pickle
import re
import sys

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from textblob import TextBlob


batch_size = 1


def exponent_neg_manhattan_distance(left, right):
    """
    Manhattn distance calculated for output of 
    LSTM for question1 and question2
    """
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

# clean text
def clean_question(q):
    """
    Clean the text in a given question to provide
    a list of relevent words
    """
    q = str(q)
    q = q.lower() # lowercase
    q = re.sub(r"what's", "what is ", q)
    q = re.sub(r"\'ve", " have ", q)
    q = re.sub(r"can't", "cannot ", q)
    q = re.sub(r"n't", " not ", q)
    q = re.sub(r"i'm", "i am ", q)
    q = re.sub(r"\'re", " are ", q)
    q = re.sub(r"\'d", " would ", q)
    q = re.sub(r"\'ll", " will ", q)
    q = re.sub(r"[^A-Za-z0-9]", " ", q)
    q = TextBlob(q).tokens #tokenize
    return q
if __name__ == "__main__":
    
    # load the model
    lstm = load_model('lstm_model', 
                  custom_objects={
                      'exponent_neg_manhattan_distance':\
                      exponent_neg_manhattan_distance})
    # get stop words
    stop_words = stopwords.words('english')
    
    # google word vectors
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(
        "GoogleNews-vectors-negative300.bin.gz", binary=True)
    
    # load vocabulary of the words from traning
    with open('vocabulary', 'rb') as f:
        vocabulary = pickle.load(f)
        
    # load maximum length of sentence used in training
    with open('max_seq_length', 'rb') as f:
        max_seq_length = pickle.load(f)
    
    print(34)
    while True:
        # get user input
        question1 = input("Enter first question: ")
        question2 = input("Enter second question: ")
        
        quora = pd.DataFrame({"question1":[question1], 
                              "question2":[question2]})
        
        # cleans the the text in question1 and question2, 
        # and them turn them into integers
        questions_cols = ['question1', 'question2']
        dataset = quora.copy()

        # Iterate over the questions only of both training 
        # and test datasets
        for index, row in dataset.iterrows():

            # Iterate through the text of both questions of the row
            for question in questions_cols:

                q2n = []  # q2n -> question numbers representation
                for word in clean_question(row[question]):

                    # Check for unwanted words
                    if word in stop_words and word not in word2vec.vocab:
                        continue
                    else:
                        try:
                            q2n.append(vocabulary[word])
                        except KeyError:
                            q2n.append(0) # unknown word

                # Replace questions as word to question as 
                # number representation
                dataset.set_value(index, question, q2n)

        dataset = {'left': dataset.question1, 
                   'right': dataset.question2}
        
        # padding make sure everything is of same size
        for dataset, side in itertools.product([dataset], 
                                               ['left', 'right']):
            dataset[side] = pad_sequences(dataset[side], 
                                          maxlen=max_seq_length)
            
        pred_v = lstm.predict([dataset['left'], 
                               dataset['right']], batch_size=batch_size)
        
        if pred_v>0.5:
            print("Questions are duplicated")
        else:
            print("Questions are not duplicated")
   
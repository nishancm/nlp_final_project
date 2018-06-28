# Project Goal
In this project I am trying to indentify whether two questions are the same or not. 

# Dataset
I am using the Quora question pairs dataset available from https://www.kaggle.com/c/quora-question-pairs for this purpose

# Overview of the project
First I am trying to see how Word Mover Distance helps in identifying similar questions. Then I move on to a MaLSTM deeplearning architecture as described in https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07.

# Folder structure
- Are two questions the same.pdf -> Poster presented to the class
- LSTM.ipynb -> Implementation of MaLSTM model
- WMD.ipynb -> Identify similar question pairs using Word Mover Distance
- display_confusion_matrix.py -> Used to plot the confusion matrix (By Brian Spiering)
- find_duplicate_questions.py -> commnad line program to identify duplicated question pairs
- lstm_model.gz -> LSTM model object from Keras
- max_seq_length -> Maximum number of words found in a question. Stored as pickle file
- model_diagram.pdf -> Model architecture diagram initially suggested
- vocabulary -> Vocabulary of all the words found from `train.csv` avaiable in the Quora dataset

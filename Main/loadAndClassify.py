import os
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, f1_score, auc
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
import itertools
import matplotlib.pyplot as plt
import random
import tensorflow as tf

model = keras.models.load_model('../input/notebook5267d82d06/RNN1.h5')
print("Model Loaded")
tokenizer = open('../input/notebook5267d82d06/tokenizer.pkl', 'rb')
print("Tokenizer loaded")
tok = pickle.load(tokenizer)
df = pd.read_csv("../input/clean2d14k/CleanData2D.csv", names=["text", "target"],
                 encoding="ISO-8859-1", skiprows=1, low_memory=False, index_col=False)
print("Data Loaded")
df=df.sample(frac=1)
print("Data Shuffled")
df=df.astype(str)
print("Data type-cast to String")
df1k=df.head(1000)
print("Df sliced over 1k rows")
le = LabelEncoder()
print("LabelEncoder Loaded")
targets = df.target
targets = le.fit_transform(targets)
print("LabelEncoder Fitted")
df1k.head(10)

def predictClass(text):
    text_pad = sequence.pad_sequences(tok.texts_to_sequences([text]), maxlen=300)
    score = le.inverse_transform(model.predict_classes([text_pad]))
    return score[0]

def dfPredicter(df):
    pred = []
    for i in df['text']:
        score = predictClass(i)
        pred.append(score)
        print(i, score)

    Table = {'input': df['text'],
             'target': df['target'],
             'pred': pred}
    dfPred = pd.DataFrame(Table)
    dfPred.to_csv('testPredict2.csv')

def csvPredicter(csv):
    df = pd.read_csv(csv, names=['input', 'target'], encoding="ISO-8859-1", skiprows=1,
                     low_memory=False, index_col=False)
    pred = []
    for i in df['input']:
        score = predictClass(i)
        pred.append(score)
        print(i, score)

    Table = {'input': df['input'],
             'target': df['target'],
             'pred': pred}
    dfPred = pd.DataFrame(Table)
    dfPred.to_csv('testPredict2.csv')

# dfPredicter(df1k)

# csvPredicter("../input/clean2d14k/CleanData2D.csv")

targets = dfPred




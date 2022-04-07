import os
import pandas as pd
import pickle
import keras
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import sequence
from itertools import repeat
import numpy as np

threshold = 0.75

CSV = r'C:\Users\mail\Downloads\data\bisgaard NOOS data.csv'
df = pd.read_csv(CSV, error_bad_lines=False, engine='c', encoding='ISO-8859-1', low_memory=False)
df = df.astype(str)
df = df.sample(1000, random_state=7)
df = df.reset_index(drop=True)

model = keras.models.load_model(r'C:\Users\mail\PycharmProjects\MLDM\Miscellaneous Code\Models\organized\6_class_MVP.h5')
tokenizer = pickle.load(open(r'C:\Users\mail\PycharmProjects\MLDM\Miscellaneous Code\Models\organized\6_class_MVP.pkl', 'rb'))
le = LabelEncoder()
print('Model, Tokenizer and LabelEncoder loaded.')

def predictClass(text, tok, model):
    text_pad = sequence.pad_sequences(tok.texts_to_sequences([text]), maxlen=300)
    score = le.inverse_transform(model.predict_classes([text_pad]))
    return score[0]

col_dict = {df.columns.get_loc(c): c for idx, c in enumerate(df.columns)}
manual_map = {0: 'Amount', 1: 'Unknown', 2: 'Name', 3: 'EAN', 4: 'EAN',
              5: 'Amount', 6: 'Price', 7: 'Price', 8: 'Price', 9: 'Price'}

map_list = []
correct = 0

lists = list(map(list, col_dict.items()))

for n in (range(len(col_dict))):
    df_select = df[col_dict[n]]
    df_select = df_select.astype(str)
    class_map = []
    for m in (range(len(df_select))):
        class_predictions = []
        targets = ['Name', 'Description', 'Category', 'Price', 'Amount', 'EAN']
        targets = le.fit_transform(targets)
        text = df_select[m]
        score = predictClass(text, tokenizer, model)
        lists[n].append(score)
        print('data:', text, ', prediction:', score)
    x = pd.Series(lists[n])
    y = pd.Series(x.value_counts())
    print('_____________________________')
    if y.iloc[0] > (len(df_select)) * threshold:
        print(y.index[0], 'Is the Majority Predicted Class.')
        for i in range(len(y) - 2):
            print('Majority Predicted classes:', y.index[i], ', predictions:', y.iloc[i], '/', len(df_select))
            class_predictions.append('Pred: {class_pred}: {num_pred} / {class_len}. Original: {origin}'
                                     .format(class_pred=y.index[i], num_pred=y.iloc[i], class_len=len(df_select),
                                             origin=col_dict[n], actual=manual_map[n]))
            if y.index[i] == manual_map[n]:
                correct += 1

    else:
        print('There is no Majority Predicted Class above the threshold.')
        for i in range(len(y) - 2):
            print('Predicted classes:', y.index[i], ', predictions:', y.iloc[i], '/', len(df_select))
            class_predictions.append(
                'Pred {num}: {class_pred}: {num_pred} / {class_len}. Original: {origin}, Actual: {actual}'
                .format(num=(i + 1), class_pred=y.index[i], num_pred=y.iloc[i], class_len=len(df_select),
                        origin=col_dict[n], actual=manual_map[n]))
    print('Original Class:', col_dict[n])
    print('__________________________________________')
    map_list.append(class_predictions)

print('Correct Maps: {correct}/{total}'.format(correct = correct, total = len(col_dict)))
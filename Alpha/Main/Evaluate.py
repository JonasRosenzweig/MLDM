
import re
import os
import pickle
import keras
import statistics

import numpy as np
import pandas as pd

from os import listdir
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import sequence

from stdnum import ean

def read_csv(df):
    return pd.read_csv(df, error_bad_lines=False, engine='c', encoding='ISO-8859-14', low_memory=False, dtype=str)

# EAN Validator
def validate_EAN(string):
    try:
        ean.validate(string)
        return True
    except:
        return False

# removes duplicates items from a list
def remove_duplicates(l):
    return list(dict.fromkeys(l))

Model_path = r'C:\Users\mail\PycharmProjects\MLDM\Alpha\Main\Trained Models\OTHER_NAME_PRICE_SIZE_COLOR_OVER_WORDEMB.h5'
Tokenizer_path = r'C:\Users\mail\PycharmProjects\MLDM\Alpha\Main\Trained Models\OTHER_NAME_PRICE_SIZE_COLOR_OVER_WORDEMB.pkl'

Mapped_Headers_path = r'C:\Users\mail\PycharmProjects\MLDM\Alpha\Organized Data\Manual maps\Headers.csv'
Mapped_Targets_path = r'C:\Users\mail\PycharmProjects\MLDM\Alpha\Organized Data\Manual maps\Targets.csv'

Mapped_Headers_df = read_csv(Mapped_Headers_path)
Mapped_Targets_df = read_csv(Mapped_Targets_path)

Mapped_Targets_list = Mapped_Targets_df.values.tolist()
Mapped_Headers_list = Mapped_Headers_df.values.tolist()

Mapped_Targets_list = [[x for x in y if str(x) != 'nan'] for y in Mapped_Targets_list]
Mapped_Headers_list = [[x for x in y if str(x) != 'nan'] for y in Mapped_Headers_list]

EVAL_path = r'C:\Users\mail\PycharmProjects\MLDM\Alpha\Organized Data\Product Data feeds'
list_files = listdir(EVAL_path)

TARGETS = ['OTHER', 'NAME', 'PRICE', 'SIZE', 'COLOR']
OTHER_list, EAN_list, PRICE_list, NAME_list, COLOR_list, SIZE_list = [], [], [], [], [],[]
Features_lists = [OTHER_list, EAN_list, PRICE_list, NAME_list, COLOR_list, SIZE_list]

for i in range(len(Mapped_Targets_list)):
    for j in range(len(Mapped_Targets_list[i])):
        if (Mapped_Targets_list[i][j]) == 'OTHER':
            OTHER_list.append(Mapped_Headers_list[i][j])
        if (Mapped_Targets_list[i][j]) == 'EAN':
            EAN_list.append(Mapped_Headers_list[i][j])
        if (Mapped_Targets_list[i][j]) == 'PRICE':
            PRICE_list.append(Mapped_Headers_list[i][j])
        if (Mapped_Targets_list[i][j]) == 'NAME':
            NAME_list.append(Mapped_Headers_list[i][j])
        if (Mapped_Targets_list[i][j]) == 'COLOR':
            COLOR_list.append(Mapped_Headers_list[i][j])
        if (Mapped_Targets_list[i][j]) == 'SIZE':
            SIZE_list.append(Mapped_Headers_list[i][j])

for i in range(len(Features_lists)):
    Features_lists[i] = remove_duplicates(Features_lists[i])
    #print(Features_lists[i])


#print(list(Mapped_Headers_df.iloc[1]))
#print(list(Mapped_Targets_df.iloc[1]))

#print(Mapped_Targets_df.iloc[1][1])

#print(len(Mapped_Headers_df.index))
#print(len(Mapped_Targets_df.index))

model = keras.models.load_model(Model_path)
tokenizer = pickle.load(open(Tokenizer_path, 'rb'))
le = LabelEncoder()
print('Model, Tokenizer and LabelEncoder loaded.')



# class prediction function using trained keras model and tokenizer
# tokenizes text, performs prediction on it, trains label encoder, returns class prediction
def predictClass(text, tok, model):
    text_pad = sequence.pad_sequences(tok.texts_to_sequences([text]), maxlen=300)
    predict_x = model.predict(text_pad)
    predict_class = np.argmax(predict_x, axis=1)
    score = le.inverse_transform(predict_class)
    prediction = score[0]
    return prediction

Predictions_list = []
# evaluate method - takes a path of .csv product feed data
# returns evaluation metrics
def evaluate(path):
    listfiles = listdir(path)
    THRESHOLD = 0.51
    SAMPLE_AMOUNT = 2
    RANDOM_STATE = 7
    k = 0


    # for file in path
    for i in range(len(listfiles)):
        Predictions = []
        correct = 0
        dataset_filename = os.listdir(path)[i]
        print(dataset_filename)
        mapped_filename = 'mapped_' + dataset_filename
        dataset_path = os.path.join('../..', path, dataset_filename)
        df = pd.read_csv(dataset_path, error_bad_lines=False, engine='c', encoding='UTF-8', low_memory=False, dtype=str)
        if len(df.index) >= SAMPLE_AMOUNT:
            try:
                df = df.sample(SAMPLE_AMOUNT, random_state=RANDOM_STATE)
            except ValueError:
                pass
        df = df.reset_index(drop=True)
        column_headers = list(df.columns)
        predictions_map = list(map(lambda item: [item], column_headers))
        predictions_only = list(map(lambda item: [item], column_headers))
        print(predictions_map)

        # for column in file
        for j in range(len(column_headers)):
            df_select = df[column_headers[j]]
            df_select = df_select.astype(str)
            print('Original Class:', column_headers[j])

            # for data in column
            for k in range(len(df_select)):
                try:
                    print('Target:', Mapped_Targets_list[i][j+1])
                except IndexError:
                    pass
                targets = TARGETS
                targets = le.fit_transform(targets)
                data = df_select[k]
                if validate_EAN(data):
                    predicted_class = 'EAN'
                elif len(data) > 200:
                    predicted_class = 'OTHER'
                elif data == 'nan':
                    predicted_class = 'NAN'
                else:
                    predicted_class = predictClass(data, tokenizer, model)
                predictions_map[j].append(predicted_class)
                predictions_only[j].append(predicted_class)
                print('prediction:', predicted_class)
                print('_________')

            predictions_only[j].pop(0)
            column_predictions = (predictions_only[j])
            column_predictions_series = pd.Series(column_predictions)
            column_predictions_count = column_predictions_series.value_counts()

            if column_predictions_count.iloc[0] > len(df_select) * THRESHOLD:
                Predictions.append(column_predictions_count.index[0])
            else:
                Predictions.append(column_predictions_count.index[0])
        Predictions_list.append(Predictions)
    os.chdir(r'C:\Users\mail\PycharmProjects\MLDM\Alpha\Organized Data\Output')
    with open('predictions.txt', 'w') as output:
        output.write(str(Predictions_list))

evaluate(EVAL_path)







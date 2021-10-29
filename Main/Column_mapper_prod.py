import os
import pandas as pd
import pickle
import keras
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import sequence
import numpy as np
import json

threshold = 0.75
sample_amount = 25

#CSV = r'/var/www/html/ADM_Data/Joha produkter.csv'
CSV = r'C:\Users\mail\Downloads\data\Joha produkter.csv'
df = pd.read_csv(CSV, error_bad_lines=False, engine='c', encoding='ISO-8859-1', low_memory=False)
df = df.astype(str)
print(len(df.index))
print(len(df.index) > sample_amount)
if len(df.index > sample_amount):
    try:
        df = df.sample(sample_amount, random_state=7)
    except ValueError:
        pass
df = df.reset_index(drop=True)
#model = keras.models.load_model(r'/var/www/html/ADM_Data/6_class_MVP.h5')
#tokenizer = pickle.load(open(r'/var/www/html/ADM_Data/6_class_MVP.pkl','rb'))
model = keras.models.load_model(r'C:\Users\mail\PycharmProjects\MLDM\Main\Models\organized\6_class_MVP.h5')
tokenizer = pickle.load(open(r'C:\Users\mail\PycharmProjects\MLDM\Main\Models\organized\6_class_MVP.pkl', 'rb'))
le = LabelEncoder()
print('Model, Tokenizer and LabelEncoder loaded.')


def predictClass(text, tok, model):
    text_pad = sequence.pad_sequences(tok.texts_to_sequences([text]), maxlen=300)
    predict_x = model.predict(text_pad)
    predict_class = np.argmax(predict_x, axis=1)
    score = le.inverse_transform(predict_class)
    return score[0]


col_dict = {df.columns.get_loc(c): c for idx, c in enumerate(df.columns)}
map_list = []
maps_object = []

lists = list(map(list, col_dict.items()))

for n in (range(len(col_dict))):
    print('----------Mapping column {num} of {len}----------'.format(num=n, len=len(col_dict)))
    df_select = df[col_dict[n]]
    df_select = df_select.astype(str)
    class_map = []
    multiple_pred = []
    for m in (range(len(df_select))):
        class_predictions = []
        targets = ['Name', 'Description', 'Category', 'Price', 'Amount', 'EAN']
        targets = le.fit_transform(targets)
        text = df_select[m]
        predicted_class = predictClass(text, tokenizer, model)
        lists[n].append(predicted_class)
        print('data:', text, ', prediction:', predicted_class)
    predicted_classes = pd.Series(lists[n])
    predicted_count = pd.Series(predicted_classes.value_counts())
    print('_____________________________')
    if predicted_count.iloc[0] > (len(df_select)) * threshold:
        print(predicted_count.index[0], 'is the Majority Predicted Class.')
        print('Majority Predicted classes:', predicted_count.index[0], ', predictions:', predicted_count.iloc[0],
              '/', len(df_select))
        class_predictions.append('Pred: {class_pred}: {num_pred} / {class_len}. Original: {origin}'
                                 .format(class_pred=predicted_count.index[0], num_pred=predicted_count.iloc[0],
                                         class_len=len(df_select),
                                         origin=col_dict[n]))
        class_map.append([col_dict[n], predicted_count.index[0]])


    else:
        print('There is no Majority Predicted Class above the threshold.')
        for i in range(len(predicted_count) - 2):
            print('Predicted classes:', predicted_count.index[i], ', predictions:', predicted_count.iloc[i], '/',
                  len(df_select))
            class_predictions.append(
                'Pred {num}: {class_pred}: {num_pred} / {class_len}. Original: {origin}'
                    .format(num=(i + 1), class_pred=predicted_count.index[i], num_pred=predicted_count.iloc[i],
                            class_len=len(df_select),
                            origin=col_dict[n]))
            multiple_pred.append([col_dict[n], predicted_count.index[i]])
        class_map.append(multiple_pred)
    print('Original Class:', col_dict[n])
    print('__________________________________________')
    map_list.append(class_predictions)
    maps_object.append(class_map)

maps_object_json = json.dumps(maps_object)
#print('map_list:', map_list)
#print('maps_object:', maps_object)
jsondf = pd.DataFrame(maps_object)
json_maps = jsondf.to_json()
#print(json_maps)
#os.chdir(r'/var/www/html/ADM_Data/json_maps')
os.chdir(r'C:\Users\mail\PycharmProjects\MLDM\Data\Maps')
with open('map.json', 'w') as file:
    file.write(json_maps)


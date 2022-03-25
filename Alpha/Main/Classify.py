import os
import pickle
import keras
import json
import glob
from stdnum import ean

import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import sequence

# Constants
THRESHOLD = 0.51
SAMPLE_AMOUNT = 10
RANDOM_STATE = 7
TARGETS = ['OTHER', 'NAME', 'UNIT_PRICE', 'SIZE', 'COLOR']
uploads_path = r'C:\Users\mail\PycharmProjects\MLDM\Alpha\Organized Data\Product Data feeds\*.csv'

# lists for custom rules classification
NAME_Headers = ['Style Name', 'Article Name', 'ïṠṡNavn']
OTHER_Headers = ['Washing','Ironing','Drying','Drycleaning','Bleaching','materiale','Materiale','%', 'Color Number',
                 'Type','type','Kolli','pct','Antal','Quantity','Qty','quantity','qty','Fit','lukning',
                 'Category','category','Kategori','kategori','Weight','weight','Qty.','Customer','Units','Age','age',
                 'delivery','date','Date','Delivery','Delivery','Season','Køn','Gender',
                 'Account','No.','Stock','stock','pct.','month','Composition','composition','available','Finish',
                 'Order Closing','Washing', 'In Stock', 'Available','Date', 'Per Display', 'Weight',
                 'Division', 'Delivery', 'Brand', 'Drop', 'No', 'Quality', 'Weigth', 'length', 'height', 'Pieces',
                 'Style no', 'Colour no', 'Color no', 'Item no.', 'Article Number', 'Total number of pairs',
                 'Farve kode', 'interval', 'toldnummer', 'LÃḊngde', 'Tarif', 'Item number', 'SKU']
PRICE_Headers = ['price','Price','Cost','VAT','Retail','Subtotal','Sub total','total','sub total','subtotal','RRP',
                 'Wholesale','wholesale','udsalgspris','pris','Pris','Udsalgspris','Katalogpris','katalogpris',
                 'Vejl.uds', 'M.S.R.P', 'R.R.P. DKK']
COLOR_Headers = ['Colour name', 'Color name', 'Color Name', 'Colour Name', 'color name', 'Color', 'Farve beskrivelse']
SIZE_Headers = ['MÅL', 'Size', 'size', 'Størrelse', 'størrelse', 'mål', 'Mål','StÃẁrrelse']
EAN_Headers = ['Barcode']
Exact_OTHER_Headers = ['brand', 'width_mm', 'hs_code', 'country_of_origin', 'description', 'Style nr.',
                       'Marketing Text (ENU)', 'Marketing Text (DAN)', 'Dess.', 'Collection', 'Country of Ori.',
                       'Compos.', 'Brand', 'Per Case', 'Dessin', 'Currency', 'Del. week', 'Material', 'Origin',
                       'Kundens referencetekst', 'Composition ', 'Varenummer', 'Style #', 'Brugstarif', 'ArticleNumber',
                       'Country', 'Oprindelsesland', 'Style Number', 'Længde', 'Department', 'Launch Month', 'id',
                       'Order no.', 'Discount', 'Art. Nr', 'Country of orgin', 'Top categories', 'HS Code',
                       'Article ID', 'LP', 'LOT', 'SalesCurr', 'Toldnummer', 'Style number']
Exact_NAME_Headers = ['Name', 'title', 'name', 'Style name', 'Style name ', 'Varenavn', 'Product name', 'Navn']
Exact_SIZE_Headers = ['Dimensions']
Exact_COLOR_Headers = ['Col. Descrip.', 'Colour', 'Col.text', 'Farve_1']
Exact_PRICE_Headers = ['Rek SKR Ex Moms', 'Veil NKR', 'M.S.R.P (EUR)', 'WHS DKK', 'WHS EUR', 'WHS NOK', 'WHS SEK',
                       'WHS USD', 'WHSP', 'Vejl DKK', 'Vejl NOK', 'Vejl SEK', 'MSRP EUR', 'R.R.P. DKK']
output = []

def print_and_save(s):
    print(s)
    output.append(s)

replace_strings = ['DKK', 'SEK', 'NOK', 'USD', 'EUR', 'GBP', ',', 'ïṠṡ']
# list files in directory
list_files = glob.glob(uploads_path)

# select most recently uploaded file in directory
recent_upload = max(list_files, key=os.path.getctime)

# read csv params
def read_csv(path):
    return pd.read_csv(path, error_bad_lines=False, engine='c', encoding='UTF-8',
                       low_memory=False, dtype=str)

# EAN validator
def validate_ean(string):
    try:
        ean.validate(string)
        return True
    except:
        return False

# model and tokenizer paths
Model_path = r'C:\Users\mail\PycharmProjects\MLDM\Alpha\Main\Trained Models\OTHER_NAME_PRICE_SIZE_COLOR_OVER_WORDEMB.h5'
Tokenizer_path = r'C:\Users\mail\PycharmProjects\MLDM\Alpha\Main\Trained Models\OTHER_NAME_PRICE_SIZE_COLOR_OVER_WORDEMB.pkl'

model = keras.models.load_model(Model_path)
tokenizer = pickle.load(open(Tokenizer_path, 'rb'))
le = LabelEncoder()

# prediction function using trained keras model and tokenzer
# tokenizes text, performs predictions on it, trains labelencoder, returns class prediction
def predictClass(text, tok, model):
    text_pad = sequence.pad_sequences(tok.texts_to_sequences([text]), maxlen=300)
    predict_x = model.predict(text_pad)
    predict_class = np.argmax(predict_x, axis=1)
    score = le.inverse_transform(predict_class)
    prediction = score[0]
    return prediction

# check if substring in list
def Substring_match(match_list, string):
    for item in match_list:
        if item in string:
            return True

# check if String in list
def String_match(match_list, string):
    for item in match_list:
        if item == string:
            return True

def remove_punct(string):
    string = string.replace(",", "")
    return string

def fix_encoding(df):
    df.replace('Ãẁ', 'ø', inplace=True, regex=True)
    df.replace('ÃḊ', 'æ', inplace=True, regex=True)
    df.replace('Ãċ', 'å', inplace=True, regex=True)
    df.replace('Ãƒáº', 'ø', inplace=True, regex=True)
    df.replace('ÃƒÄ‹', 'æ', inplace=True, regex=True)
    df.replace('ÃƒÄ‹', 'å', inplace=True, regex=True)

# load csv into df
df_recent = read_csv(recent_upload)

def classify(df, save_name, json_name):
    global retail_prices, cost_prices
    map_list = []
    maps_object = []
    Predictions = []
    Certainty = []
    Maj_Pred = []
    multiple_predictions = []
    multiple_certainties = []
    class_predictions = []

    # sample SAMPLE_AMOUNT of rows from df with equal probability
    if len(df.index >= SAMPLE_AMOUNT):
        try:
            df_sampled = df.sample(SAMPLE_AMOUNT,  random_state=RANDOM_STATE)
        except ValueError:
            pass
    df_sampled = df_sampled.reset_index(drop=True)

    # column headers list
    column_headers = list(df_sampled.columns)

    # predictions map - predictions will be appended, matching the column header
    predictions_map = list(map(lambda item: [item], column_headers))

    # predictions only - map a list of size of column headers - headers will be popped
    # only includes predictions
    predictions_only = list(map(lambda item: [item], column_headers))


    for n in range(len(column_headers)):
        print_and_save('-----------Mapping column {num} of {len}-----------'
              .format(num=n+1, len=len(column_headers)))
        df_select = df_sampled[column_headers[n]]
        df_select = df_select.astype(str)
        fix_encoding(df_select)

        for item in replace_strings:
            df_select = df_select.replace(item, '')

        for m in range(len(df_select)):
            targets = TARGETS
            target = le.fit_transform(targets)
            data = df_select[m]
            if Substring_match(OTHER_Headers, column_headers[n] or len(data) > 200)\
                    or String_match(Exact_OTHER_Headers, column_headers[n]) or data == '0':
                predicted_class = 'OTHER'
                print('Custom rule applied - OTHER')
            elif Substring_match(COLOR_Headers, column_headers[n]) \
                    or String_match(Exact_COLOR_Headers, column_headers[n]):
                predicted_class = 'COLOR'
                print('Custom rule applied - COLOR')
            elif Substring_match(PRICE_Headers, column_headers[n]) \
                    or String_match(Exact_PRICE_Headers, column_headers[n]):
                predicted_class = 'UNIT_PRICE'
                print('Custom rule applied - PRICE')
            elif Substring_match(SIZE_Headers, column_headers[n]) \
                    or String_match(Exact_SIZE_Headers, column_headers[n]):
                predicted_class = 'SIZE'
                print('Custom rule applied - SIZE')
            elif Substring_match(EAN_Headers, column_headers[n]):
                predicted_class = 'PROD_BARCODE_NUMBER'
                print('Custom rule applied - EAN')
            elif Substring_match(NAME_Headers, column_headers[n]) \
                    or String_match(Exact_NAME_Headers, column_headers[n]):
                predicted_class = 'PROD_NAME'
                print('Custom rule applied - NAME')
            elif validate_ean(data):
                predicted_class = 'PROD_BARCODE_NUMBER'
                print('EAN Validator applied')
            elif data == 'nan' or data == '0' or data == 0:
                predicted_class = 'NAN'
            elif predictClass(data, tokenizer, model) == 'NAME':
                predicted_class = 'PROD_NAME'
                print_and_save('Model Prediction: {f}'.format(f=predicted_class))
            else:
                predicted_class = predictClass(data, tokenizer, model)
                print_and_save('Model Prediction: {f}'.format(f=predicted_class))
            predictions_map[n].append(predicted_class)
            predictions_only[n].append(predicted_class)
            print('Data: {data}, OC: {OC}, Prediction: {pred}'
                  .format(data=data, OC=column_headers[n], pred=predicted_class))
        predictions_only[n].pop(0)
        column_predictions = (predictions_only[n])
        column_predictions_series = pd.Series(column_predictions)
        column_predictions_count = column_predictions_series.value_counts()
        print_and_save('predictions only: {f}'.format(f=predictions_only))
        print_and_save('column predictions: {f}'.format(f=column_predictions))
        print_and_save('column predictons series: {f}'.format(f=column_predictions_series))
        print_and_save('column predictions count: {f}'.format(f=column_predictions_count))
        print_and_save('Majority class: {f}'.format(f=column_predictions_count.index[0]))
        print_and_save('Number of predictions: {f}'.format(f=column_predictions_count.iloc[0]))
        print_and_save('____________________________')

        if column_predictions_count.iloc[0] > len(df_select) * THRESHOLD:
            print_and_save('{f} is the majority prediction.'.format(f=column_predictions_count.index[0]))
            print_and_save('The majority class is: {pred} with {num_pred} of {len} predictions.'
                  '\n Original Class: {original}'
                  .format(pred=column_predictions_count.index[0],
                          num_pred=column_predictions_count.iloc[0],
                          len=len(df_select),
                          original=column_headers[n]))
            Predictions.append(column_predictions_count.index[0])
            Maj_Pred.append(column_predictions_count.index[0])
            Certainty.append(100)

        else:
            print_and_save('There is no majority class'
                  '\n Original class: {original}'
                  .format(original=column_headers[n]))
            for i in range(len(column_predictions_count)):
                print_and_save('Predicted class {i}: {pred} with {num_pred} of {len} predictions'
                      .format(i=i+1,
                              pred=column_predictions_count.index[i],
                              num_pred=column_predictions_count.iloc[i],
                              len=len(df_select)))
                multiple_predictions.append(column_predictions_count.index[i])
                multiple_certainties.append(column_predictions_count.iloc[i] / len(df_select) * 100)
            Predictions.append(multiple_predictions)
            Maj_Pred.append(column_predictions_count.index[0])
            Certainty.append(multiple_certainties)
    # fix so we remove duplicates inside list of maps
    predictions_map_no_duplicates = []
    [predictions_map_no_duplicates.append(x) for x in predictions_map if x not in predictions_map_no_duplicates]
    print_and_save('map list: {f}'.format(f=predictions_map_no_duplicates))
    json_map = {"Columns": []}
    json_pred_cert = {}

    for i in range(len(column_headers)):
        json_map["Columns"].append({"Original Class {i}".format(i=i+1): column_headers[i],
                                    "Model Prediction(s)": []})
        if isinstance(Predictions[i], list):
            json_map["Columns"][i]["Model Prediction(s)"] \
                .append({"Prediction": (Predictions[i]), "Certainty": (Certainty[i])})
        else:
            json_map["Columns"][i]["Model Prediction(s)"]\
                .append({"Prediction": [(Predictions[i])], "Certainty": [(Certainty[i])]})

    json_map = json.dumps(json_map)


    os.chdir(r'C:\Users\mail\PycharmProjects\MLDM\Data\Maps')
    #filename = Path(os.path.basename(recent_upload))
    mapped_filename = 'mapped_'+ os.path.basename(filename)
    #json_filename = filename.with_suffix('.json')
    with open(json_filename, 'w') as file:
        file.write(json_map)

    print_and_save('----------Dataset Mapping Summary----------')
    print_and_save('map list: {f}'.format(f=predictions_map))
    #print('predictions only:', predictions_only)
    print_and_save('Predictions: {f}'.format(f=Predictions))
    print_and_save('Percentages: {f}'.format(f=Certainty))
    print(df_sampled.head())
    df_renamed = df.copy()
    print(len(Maj_Pred))
    print(Maj_Pred)
    df_renamed.columns = Maj_Pred

    df_prices = df_renamed.filter(like='UNIT_PRICE')
    df_colors = df_renamed.filter(like='COLOR')
    df_colors = df_colors.replace('\d+', '', regex=True)
    df_colors = df_colors.replace('/', '', regex=True)
    df_colors = df_colors.replace(r'\'', '', regex=True)
    for column in df_colors:
        try:
            if pd.to_numeric(df_colors[column], errors='coerce').notnull().all():
                del df_colors[column]
        except TypeError:
            pass
    try:
        df_prices = df_prices.astype(float)
    except ValueError:
        pass
    prices = []
    for i in range(len(df_prices)):
        try:
            col = df_prices.iloc[:, i]
            prices.append(list(col))
        except IndexError:
            pass
    for i in range(len(df_prices)):
        try:
            if prices[0][0] < prices[i+1][0]:
                cost_prices = prices[0]
                retail_prices = prices[i+1]
            elif prices[i+1][0] < prices[0][0]:
                cost_prices = prices[i+1]
                retail_prices = prices[0]
        except:
            pass
    try:
        cost_dict = {'COST_PRICE': cost_prices}
        retail_dict = {'UNIT_PRICE': retail_prices}
        df_cost_prices = pd.DataFrame(cost_dict)
        df_retail_prices = pd.DataFrame(retail_dict)
    except UnboundLocalError:
        pass
    try:
        del df_renamed['UNIT_PRICE']
    except KeyError:
        pass
    try:
        df_renamed = pd.concat([df_renamed, df_cost_prices, df_retail_prices], axis=1, join='inner')
    except UnboundLocalError:
        pass
    try:
        del df_renamed['OTHER']
        del df_renamed['NAN']
    except KeyError:
        pass
    df_renamed = df_renamed.loc[:,~df_renamed.columns.duplicated()]
    try:
        df_renamed['PROD_NAME'] = df_renamed.PROD_NAME.str.cat(df_renamed.COLOR, sep=', Color: ')
        df_renamed['PROD_NAME'] = df_renamed.PROD_NAME.str.cat(df_renamed.SIZE, sep=', Size ')
    except AttributeError:
        pass
    try:
        del df_renamed['COLOR']
        del df_renamed['SIZE']
    except KeyError:
        pass

    print(df_renamed.head())
    Multi_Header1 = ['PRODUCTS']
    for y in range(len(df_renamed.columns)-1):
        Multi_Header1.append('')
    df_renamed.columns = pd.MultiIndex.from_arrays([Multi_Header1, df_renamed.columns])
    print(df_renamed.head())
    os.chdir(r'C:\Users\mail\PycharmProjects\MLDM\Demo_Output')
    df_renamed.to_csv(mapped_filename, index=False)

for i in range(len(list_files)):
    print_and_save('Classifyng File: {f}'.format(f=list_files[i]))
    df_select = read_csv(list_files[i])
    filename = Path(os.path.basename(list_files[i]))
    savename = 'mapped_' + os.path.basename(list_files[i])
    json_filename = filename.with_suffix('.json')
    classify(df_select, savename, json_filename)

os.chdir(r'C:\Users\mail\PycharmProjects\MLDM\Demo_Text_output')
with open('output.txt', 'w', encoding='utf-8') as f:
    for line in output:
        f.write(line)
        f.write('\n')


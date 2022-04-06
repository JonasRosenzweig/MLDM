### Old data transformation script for early DanDomain classifier MVP ###
# Concatenates all .csv files from dir into one .csv
import os

import pandas as pd

from os import listdir

DATAPATH = r'C:\Users\mail\PycharmProjects\MLDM\Data\2D Data'
SAVEPATH = r'C:\Users\mail\PycharmProjects\MLDM\Data\Concatenated Data'

files = listdir(DATAPATH)
df_list = []
i = 0

for j in range(len(files)):
    print(i + len(files),  'of', (len(files)), "files remaining.")
    dataset_filename = os.listdir(DATAPATH)[j]
    dataset_path = os.path.join("../..", DATAPATH, dataset_filename)
    data = pd.read_csv(dataset_path)
    i -= 1
    df_list.append(data)

print("0 of", len(files), "files remaining.")
os.chdir(SAVEPATH)
df = pd.concat(df_list)
print('Done Concatening files.')

# remove rows with included headers
unused_headers = ['PROD_PDF_URL_TEXT_3', 'PROD_PDF_URL_TEXT_2', 'PROD_PDF_URL_TEXT', 'PROD_HIDDEN_PERIOD_ID',
                  'PROD_FILE_URL', 'FIELD_8', 'PROD_NOTES', 'PROD_FRONT_PAGE_PERIOD_ID', 'FIELD_6',
                  'PROD_PDF_URL_2', 'PROD_PDF_URL', 'FIELD_3', 'PROD_UNIQUE_URL_NAME', 'FIELD_2', 'PROD_NEW_PERIOD_ID',
                  'FIELD_5', 'PROD_DELIVER_NOT_IN_STOCK', 'PROD_PICTURE_ALT_TEXT', 'PROD_GOOGLE_FEED_CATEGORY',
                  'PROD_LOCATION_NUMBER', 'DESC_LONG_2', 'DESC_SHORT', 'PROD_SEARCHWORD', 'VENDOR_NUM', 'INTERNAL_ID',
                  'PROD_TYPE_ID', 'MANUFAC_ID', 'PROD_CAT_ID']
removed_headers = ['PROD_HIDDEN', 'PROD_VAR_MASTER', 'PROD_NEW', 'PROD_FRONT_PAGE', 'TOPLIST_HIDDEN', 'OMIT_FROM_FREE_SHIPPING_LIMIT',
                   'PROD_CREATED', 'PROD_EDITED', 'PROD_RETAIl_PRICE', 'PROD_COST',  'UNIT_PRICE', 'PROD_COST_PRICE',
                   'LANGUAGE_ID', 'PROD_WEIGHT', 'PROD_PHOTO_URL', 'CURRENCY_CODE', 'DIRECT_LINK',
                   'PROD_SHOW_ON_GOOGLE_FEED', 'PROD_BARCODE_NUMBER', 'PROD_SHOW_ON_FACEBOOK_FEED', 'PROD_UNIQUE_URL_NAME',
                   'PROD_SHOW_ON_PRICERUNNER_FEED', 'PROD_SHOW_ON_KELKOO_FEED', 'PROD_RETAIL_PRICE', 'PROD_PDF_URL', 'PROD_PDF_URL_2',
                   'PROD_PDF_URL_3', 'PROD_DELIVERY', 'PROD_DELIVERY_NOT_IN_STOCK', 'FIELD_1', 'FIELD_2', 'FIELD_3', 'FIELD_4', 'FIELD_5',
                   'FIELD_6', 'FIELD_7', 'FIELD_8', 'FIELD_9', 'FIELD_10', 'PROD_FILE_URL', 'ïï\x06Óµ\x07L¸Úb.j"ÿ"%\x165´', 'EDBPriser_NUM',
                   'PROD_SORT', 'PROD_SITE_SPECIFIC_SORTING', 'PROD_UNIT_ID']
kept_headers = ['PROD_NAME', 'TITLE', 'PROD_CREATED_BY', 'PROD_EDITED_BY', 'DESC_LONG', 'META_DESCRIPTION', 'MANUFAC_ID',
                'PROD_CAT_ID', 'PROD_NUM']
count = ['PROD_SALES_COUNT', 'STOCK_COUNT', 'STOCK_LIMIT', 'PROD_VIEWED', 'PROD_MIN_BUY', 'PROD_MAX_BUY', 'PROD_MIN_BUY_B2B']
description = ['DESC_LONG', 'META_DESCRIPTION']
author = ['PROD_CREATED_BY', 'PROD_EDITED_BY']
#kept_headers = []
# BOOL: PROD_SHOW_ON_GOOGLE_FEED]
print('Nunique headers:', df.Header.nunique())
print('unique headers:', df.Header.unique())

df = df[df.Header.str.contains('|'.join(kept_headers))]
df = df[~df.Header.str.contains('DESC_LONG_2')]
#df = df[~df.Header.str.contains('|'.join(count))]
#df = df[~df.Header.str.contains('|'.join(description))]
#df = df[~df.Header.str.contains('|'.join(removed_headers))]
#df = df[~df.Header.str.contains('|'.join(unused_headers))]
df = df.sample(frac=1)
for n in range(len(author)):
    df = df.replace(author[n], 'AUTHOR')
# for i in range(len(removed_headers)):
#     df = df.replace(removed_headers[i], 'UNKNOWN')
# for j in range(len(unused_headers)):
#     df = df.replace(unused_headers[j], 'UNKNOWN')
# for k in range(len(count)):
#     df = df.replace(count[k], 'COUNT')
df = df.head(100)
print('Removed unwanted headers.')
df.to_csv('8_class_PNUM_PNAME_AUT_TIT_DESCL_DESCM_MANID_PCATID_100.csv', index=False)
print('Done saving transformed file.')
print('Nunique headers:', df.Header.nunique())
print('unique headers:', df.Header.unique())
 #['DESC_LONG' 'MANUFAC_ID' 'PROD_NAME' 'PROD_NUM' 'TITLE' 'META_DESCRIPTION'] # -> last one

 #['MANUFAC_ID', 'INTERNAL_ID', 'COUNT', 'PROD_TYPE_ID', 'COUNT'] # -> new one
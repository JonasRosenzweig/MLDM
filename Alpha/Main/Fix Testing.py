import pandas as pd

# fix DKK
# if DKK or SEK or NOK or USD or EUR or GBP is in price, remove
DKK = ['DKK 100.00', 'DKK 350.00', 'DKK 346.50']
SEK = ['SEK 100.00', 'SEK 350.00', 'SEK 346.50']
NOK = ['NOK 100.00', 'NOK 350.00', 'NOK 346.50']
USD = ['USD 100.00', 'USD 350.00', 'USD 346.50']
EUR = ['EUR 100.00', 'EUR 350.00', 'EUR 346.50']
GBP = ['GBP 100.00', 'GBP 350.00', 'GBP 346.50']
prices = {'DKK': DKK, 'SEK': SEK, 'NOK': NOK, 'USD': USD, 'EUR': EUR, 'GBP': GBP }
df_prices = pd.DataFrame(data=prices)
replace_strings = ['DKK', 'SEK', 'NOK', 'USD', 'EUR', 'GBP']

for column in df_prices:
    for item in replace_strings:
        df_prices[column] = df_prices[column].str.replace(item, '')
print('-------test 1-------:')
print(df_prices.head())


# fix , in 1,000
# change 1,000 to 1000
commas1 = ['1,000', '2,000', '1,500', '12,500']
commas2 = ['3,000', '7,000', '4,500', '19,500']
commas_dict = {'Commas1': commas1, 'Commas2': commas2}
df_commas = pd.DataFrame(data=commas_dict)
for column in df_commas:
    df_commas[column] = df_commas[column].str.replace(',', '')
print('-------test 2-------:')
print(df_commas.head())

# change column order
# save each column as its own df then recombine
columns1 = ['COLOR', 'COST_PRICE', 'RETAIL_PRICE', 'SIZE', 'PROD_NAME', 'PROD_BARCODE_NUMBER']
columns2 = ['COST_PRICE', 'RETAIL_PRICE', 'SIZE', 'PROD_NAME', 'COLOR', 'PROD_BARCODE_NUMBER']
desired = ['PROD_NAME', 'COST_PRICE', 'RETAIL_PRICE', 'PROD_BARCODE_NUM', 'SIZE', 'COLOR']


# remove numbers from colors
# 1345 Blue f.x should just be Blue
colors = ['1532 Blue/green', '420 Red', '720 Black/Silver']
colors_dict = {'Colors': colors}
df_colors = pd.DataFrame(data=colors_dict)
for column in df_colors:
    df_colors[column] = df_colors[column].str.replace('\d+', '')
    df_colors[column] = df_colors[column].str.replace('/', ' ')
    df_colors[column] = df_colors[column].str.replace(r'\'', ' ')

print('-------test 3-------:')
print(df_colors.head())

# fix wrong unicode characters
test_string = ''
def fix_encoding(df):
    df.replace('Ãẁ', 'ø', inplace=True, regex=True)
    df.replace('ÃḊ', 'æ', inplace=True, regex=True)
    df.replace('Ãċ', 'å', inplace=True, regex=True)
    df.replace('Ãƒáº', 'ø', inplace=True, regex=True)
    df.replace('ÃƒÄ‹', 'æ', inplace=True, regex=True)
    df.replace('ÃƒÄ‹', 'å', inplace=True, regex=True)

# add CR for non-substring matches (full matches)
# check if full string matches not just substring
match_list = ['test', 'test1']
def String_match(l, string):
    for item in l:
        if item == string:
            return True


print('-------test 4.1-------:')
if String_match(match_list, 'test'):
    print('True')
else:
    print('False')
print('-------test 4.2-------:')
if String_match(match_list, 'test1'):
    print('True')
else:
    print('False')
print('-------test 4.3-------:')
if String_match(match_list, 'test2'):
    print('True')
else:
    print('False')

# when color is only number, remove whole column
# if column name = COLOR and value is numeric del column
colors_num = ['1532', '420', '720']
colors_num_dict = {'Colors1': colors_num, 'Colors2': colors}
df_colors_num = pd.DataFrame(data=colors_num_dict)
for column in df_colors_num:
    if pd.to_numeric(df_colors_num[column], errors='coerce').notnull().all():
        del df_colors_num[column]
print('-------test 5-------:')
print(df_colors_num.head())


# fix size/price misclassifications with CR

# print all console output to text file
print('-------test 6-------:')
numbers1 = ['100', '200', '300']
numbers2 = ['50', '100', '200']
numbers3 = ['25', '50', '100']
numbers_dict = {'1': numbers1, '2': numbers2, '3': numbers3}
df_numbers = pd.DataFrame(data=numbers_dict)
for column in df_numbers:
    df_select = df_numbers[column]
    df_select = df_select.astype(float)
    print(column, sum(df_select))

def find_largest_sum(df):
    df = df.astype(float)
    sum_list = []
    for col in df:
        sum_list.append(sum(df[col]))



# compare sum of two columns of numbers
# def df_test:
#   df.astype(float)

# implement changes in Classify.py
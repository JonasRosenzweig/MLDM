### testing Regex for various matches ###
import re

pycharm_format = '03-05-2021 13:07:56'
excel_format = '03/05/2021 13.07.56'
format3 = '03/05/2021 13:07:56'


DATE_TIME_REGEX = '([0-9]|0[0-9]|1[0-9])-([0-9][0-9]|[0-9])-[0-9]{4} ([0-9]|0[0-9]|1[0-9])(.)[0-9]{2}:[0-9]{2}'

match = re.search(DATE_TIME_REGEX, pycharm_format)
match2 = re.search(DATE_TIME_REGEX, excel_format)
match3 = re.search(DATE_TIME_REGEX, format3)

boolean = bool(match)
boolean2 = bool(match2)
boolean3 = bool(match3)

print(boolean, boolean2, boolean3)


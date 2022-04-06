### Quick testing of np.argmax output for improved modelling predictions and certainty ###
import re
#print(np.argmax([[3.6676682e-04 1.1747205e-02 1.1753693e-03 8.1953275e-01 5.5644330e-02 1.1153350e-01]), axis=1)
REGEX = '([A-Z0-9]{2})$'
print(bool(re.search(REGEX, 'AF')))
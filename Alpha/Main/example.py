### Example of using Classify.py ###
from pathlib import Path
import os
import glob
from Alpha.Main import Classify

# directory of data feeds to classify
directory = r'C:\Users\mail\PycharmProjects\MLDM\Alpha\Organized Data\Product Data feeds\*.csv'
# list files in directory
list_files = glob.glob(directory)
# directory for .json map output - change if needed
Classify.maps_path = Classify.maps_path
# directory for .csv output - change if needed
Classify.demo_output = Classify.demo_output
# directory for .txt output - change if needed
Classify.demo_text_output = Classify.demo_text_output


for n in range(len(list_files)):
    filename = Path(os.path.basename(list_files[n]))
    savename = 'mapped_' + os.path.basename(list_files[n])
    json_filename = filename.with_suffix('.json')
    Classify.classify(Classify.read_csv(list_files[n]), savename, json_filename)
    # Classify takes a DataFrame, a savename for the output .csv and a savename for the output.json




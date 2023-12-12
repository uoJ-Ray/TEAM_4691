## Environments
- Ubuntu 20.04.4 LTS
- Linux 5.4.0-156-generic
- Nvidia RTX A6000 GPU * 2

## Packages
- python 3.11.5
- pip 23.3
Other required packages are listed in the ``requirements.txt``. Run 
``python -m pip install requirements.txt`` to install the packages.

<!-- ## Script
Run
- ``bash code/script.sh {TRAIN_FILE} {TEST_FILE} {TEST_DIR} {PATH1} {PATH2} {OUT_FILE1} {OUT_FILE2} {OUT_FILE3} {OUT_FILE4} {OUT_FILE5}``

A valid example is 
``bash code/script.sh ./train.tsv ./opendid_test.tsv ./opendid_test ./model.pt ./model_date.pt ./model.txt ./model_date.txt ./extract.txt ./date.txt ./final.txt``

After running this script, you will obtain our final submitted answer ``final.txt``. You can also try to run each command by following the instructions below. -->

## Run Commands One by One
To obtain our results, please follow the commands below:


### Preprocessing
Run
- `python code/preprocess.py -data_dir {TRAIN_DIR} -output_file {TRAIN_FILE}` 

Please replace the {TRAIN_DIR}  to the directory which you put all of the training file (including the first phase and second phase dataset e.g. 1001.txt). {TRAIN_DIR} should contain 1120+614=1814 .txt files.

Please replace {TRAIN_FILE} to the path which you want to put the training data.
A valid command example is ``python code/preprocess.py -data_dir ./ -output_file ./train.tsv``

### Training
Run these two commands for the training
- ``python code/main.py -model_path {PATH1} -dataset {TRAIN_FILE}``
- ``python code/main.py -model_path {PATH2} -dataset {TRAIN_FILE} -date``

Please replace {PATH1} and {PATH2} to the paths which you want to save your model. Make sure {PATH1} $\neq$ {PATH2}. 
Please replace {TRAIN_FILE} to the path which you specify in the preprocessing stage.
A valid example for each command is
- ``python code/main.py -model_path ./model.pt -dataset ./train.tsv``
- ``python code/main.py -model_path ./model_date.pt -dataset ./train.tsv -date``

### Testing
Run the two commands for the testing
- ``python code/main.py -model_path {PATH1} -dataset {TEST_FILE} -output_file {OUT_FILE1} -test``
- ``python code/main.py -model_path {PATH2} -dataset {TEST_FILE} -output_file {OUT_FILE2} -test``
- ``python extract.py -dataset {TEST_FILE} -infile {OUT_FILE1} -infile_date {OUT_FILE2} -output_file {OUT_FILE3}``
- ``python code/chatgpt.py -api_key {API_KEY} -data_dir {TEST_DIR}``
- ``python date.py -data_dir {TEST_DIR} -dataset {TEST_FILE} -output_file {OUT_FILE4}``
- ``cat {OUT_FILE3} {OUT_FILE4} > {OUT_FILE5}``

Please replace {PATH1} and {PATH2} to the paths which you have saved your model in the training stage.
Please replace {TEST_FILE} to the path of the "opendid_test.tsv".
Please replace {OUT_FILE1}, {OUT_FILE2}, {OUT_FILE3}, {OUT_FILE4} and {OUT_FILE5} to the paths which you want to save your results. Make sure {OUT_FILE1} $\neq$ {OUT_FILE2} $\neq$ {OUT_FILE3} $\neq$ {OUT_FILE4} $\neq$ {OUT_FILE5}.
Please replace {TEST_DIR} to the directory which you put all of the testing file (e.g. 5389.txt)
Please replace {API_KEY} with your OpenAI API_KEY
A valid example for each command is
- ``python code/main.py -model_path ./model.pt -dataset ./opendid_test.tsv -output_file ./model.txt -test``
- ``python code/main.py -model_path ./model_date.pt -dataset ./opendid_test.tsv -output_file ./model_date.txt -test``
- ``python code/extract.py -dataset ./opendid_test -infile ./model.txt -infile_date ./model_date.txt -output_file extract.txt``
- ``python code/chatgpt.py -api_key sk-xxx -data_dir ./opendid_test``
- ``python code/date.py -data_dir ./opendid_test -dataset ./opendid_test.tsv -output_file date.txt``
- ``cat extract.txt date.txt > final.txt``

Final.txt is our final submitted answer.

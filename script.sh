#!/bin/bash
## $1: TRAIN_FILE
## $2: TEST_FILE
## $3: TEST_DIR
## $4: PATH1
## $5: PATH2
## $6: OUT_FILE1
## $7: OUT_FILE2
## $8: OUT_FILE3
## $9: OUT_FILE4
## $10: OUT_FILE5
## python code/preprocess.py -output_file {TRAIN_FILE}
## python code/main.py -model_path {PATH1} -dataset {TRAIN_FILE}
## python code/main.py -model_path {PATH2} -dataset {TRAIN_FILE} -date
## python code/main.py -model_path {PATH1} -dataset {TEST_FILE} -output_file {OUT_FILE1} -test
## python code/main.py -model_path {PATH2} -dataset {TEST_FILE} -output_file {OUT_FILE2} -test
## python extract.py -dataset {TEST_FILE} -infile {OUT_FILE1} -infile_date {OUT_FILE2} -output_file {OUT_FILE3}
## python date.py -data_dir {TEST_DIR} -dataset {TEST_FILE} -output_file {OUT_FILE4}
## cat {OUT_FILE3} {OUT_FILE4} > {OUT_FILE5}

python code/preprocess.py -output_file $1
python code/main.py -model_path $4 -dataset $1
python code/main.py -model_path $5 -dataset $1 -date
python code/main.py -model_path $4 -dataset $2 -output_file $6 -test
python code/main.py -model_path $5 -dataset $2 -output_file $7 -test
python extract.py -dataset $2 -infile $6 -infile_date $7 -output_file $8
python date.py -data_dir $3 -dataset $2 -output_file $9
cat $8 $9 > $10
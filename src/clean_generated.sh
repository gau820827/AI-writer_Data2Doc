#! /bin/bash 
# python data_utils.py -mode prep_gen_data -gen_fi $1 -dict_pfx "roto-ie" -output_fi $2 -input_path "../boxscore-data/rotowire"
python data_utils.py -test -mode prep_gen_data -gen_fi $1 -dict_pfx "roto-ie" -output_fi $2 -input_path "../boxscore-data/rotowire"

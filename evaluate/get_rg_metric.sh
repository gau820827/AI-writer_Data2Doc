#! /bin/bash
th extractor.lua -gpuid 1 -datafile roto-ie.h5 -preddata $1 -dict_pfx "roto-ie" -just_eval

# AI writter on Rotowire dataset
This project can be divided into two parts:
* data2text generation
* extract evaluation of the generated text

## data2text generation
The data2text generation part of this project are fully self-implemented.
The files are for this part include:
* dataprepare.py -- word2index map class, storing the vocabulary and relations
* model.py -- The file that contains the implementation of several encoder, decoder and embedding model class
* preprocessing.py -- mainly for read of parse the data
* train.py -- the file that defines the training processes of the all models
* util.py -- utility functions for time, showing etc.
* setting.py -- store the hyper-parameter, file location etc.
 
## extract evaluation
The extract evaluation used in this system is mainly based on the work of Wiseman, Sam, Stuart M. Shieber, and Alexander M. Rush. "Challenges in data-to-document generation." arXiv preprint arXiv:1707.08052 (2017).
The files from their repo contains:
* data_utils.py -- for the data parsing and cleaning
* extractor.lua -- the main script for relation classifier
* non_rg_metrics.py -- CS and RO computation

There are a also a CNN relation extractor in the /not_used directory implemented by us.
The three files:
* cnn_model.py -- definition of CNN relation classifier
* data_prepare.py -- data cleaning and preparaion
* train_cnn.py -- training process of CNN relation classifier

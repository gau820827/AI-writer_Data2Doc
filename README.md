# AI writter - Data2Doc
This is the project for automatically generating summarizatons given NBA game box-score.

## Requirements
1. Python 3.6
2. PyTorch 0.2

## Data set
We use Rotowire [dataset](https://github.com/harvardnlp/boxscore-data) for training as in [Challenges in Data-to-Document Generation](https://arxiv.org/abs/1707.08052) (Wiseman, Shieber, Rush; EMNLP 2017). This dataset consists of (human-written) NBA basketball game summaries aligned with their corresponding box- and line-scores.

## Basic Usage

1. First extract the dataset, using `tar -jxvf boxscore-data/rotowire.tar.bz2`.
2. Go to directory train/, using `cd train`
3. (Optional) Train the model, using `python3 train.py`.
4. Generate some text, using `python3 small_evaluate.py`

Some pre-trained model files could be found [here](https://drive.google.com/open?id=1tfVJin5_RYSW4b8DJE2YQTzoezAsNXyV).
Extract them under model/ directory.

* Sample Generate Summary using BiLSTM encoder
```
The Chicago Bulls defeated the Orlando Magic , 102 - 92 , at
United Center on Saturday . The Bulls ( 10 - 2 ) were expected
to win this game rather easily and they did n’t disappoint in the
fourth quarter . In fact , they outscored the Magic by 12 points in
the fourth quarter to pull out the win . Defense was key for the
Bulls , as they held the Magic to 37 percent from the field and 25
percent from three - point range . The Magic also dominated the
rebounding , winning that battle , 44 - 32 . The Magic ( 8 - 18 )
had to play this game without Joel Embiid and Nikola Vucevic ,
as they simply did n’t have enough to get out of hand . Jimmy
Butler was the player of the game , as he tallied 23 points , five
rebounds and four assists . Aaron Gordon was the only other
starter with more than 10 points , as he totaled 15 points , five
rebounds and four assists. . . ...
```

## Settings

In train/settings.py you can find the configurations.

### Parameter for model
* `LAYER_DEPTH = 2`, the depth of recurrent unit, default = 2.
* `EMBEDDING_SIZE = 600`, the hidden size for embedding, default = 600.

### Parameter for training
* `MAX_SENTENCES = None`, limit the maximum length for training. Set lower (e.g 5) for faster training speed. Set None to train for entire corpus.
* `LR = 0.01`, learning rate.
* `ITER_TIME = 10000`
* `BATCH_SIZE = 8`
* `USE_MODEL = None`, load trained model before training.

### Parameter for display
* `GET_LOSS = 1`, print out average loss every `GET_LOSS` iterations.
* `SAVE_MODEL = 5000`, save the model every `SAVE_MODEL` iterations.
* `OUTPUT_FILE = 'default'`, starting name for saving the model. During training, encoder and decoder would be saved as `[OUTPUT_FILE]_[encoder|decoder]_[iter_time]` every `SAVE_MODEL` iteratons.
* `ENCODER_STYLE = 'BiLSTM'`, choose encoder model, currently I have 3 styles -- `'RNN'`, `'LIN'` and `'BiLSTM'`. All these archotectures are within seq2seq with attention framework.

## More Details

### Train
In train/ directory is the part of data2text generation.
The files are for this part include:
* dataprepare.py -- word2index map class, storing the vocabulary and relations
* model.py -- The file that contains the implementation of several encoder, decoder and embedding model class
* preprocessing.py -- mainly for read of parse the data
* train.py -- the file that defines the training processes
* util.py -- utility functions for time, showing etc.
* setting.py -- store the hyper-parameter, file location etc.
 
### Evaulate
In evaluate/ directory is the extraction evaluation system, based on [Challenges in Data-to-Document Generation](https://arxiv.org/abs/1707.08052) (Wiseman, Shieber, Rush; EMNLP 2017) and part of [their codes](https://github.com/harvardnlp/data2text).
The files from their repo contains:
* data_utils.py -- for the data parsing and cleaning
* extractor.lua -- the main script for relation classifier
* non_rg_metrics.py -- CS and RO computation

## Thanks to the dataset and code from Wiseman et. al.

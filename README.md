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

## Arguments

The `train/train.py` accepts the following arguments.

(The default configurations for each argument can be found in `train/settings.py`.)

```
  # Parameters for model
  -embed EMBEDDING_SIZE,    the hidden size for embedding,, default = 600
  -lr LR,                   initial learning rate, default = 0.01
  -batch BATCH_SIZE,        batch size, default = 2
  
  # Parameters for model
  -encoder ENCODER_STYLE,   type of encoder NN (LIN, BiLSTM, RNN, BiLSTMMaxPool, HierarchicalRNN,
                            HierarchicalBiLSTMMaxPool, HierarchicalLIN)
  -decoder DECODER_STYLE,   type of decoder NN (RNN, HierarchcialRNN)
  -copy,                    if apply pointer-generator network(True, False), default = False
 
  # Parameters for training
  -gradclip GRAD_CLIP,      gradient clipping, default = 2
  -pretrain PRETRAIN,       file name of pretrained model (must assign with iternum)
  -iternum ITER_NUM,        file name of pretraiend model (must assign with pretrain)
  -layer LAYER_DEPTH,       the depth of recurrent units, default = 2; no depth for linear units
  -copyplayer COPY_PLAYER,  if include player's information in data, default = False 
  -epoch EPOCH_TIME,        maximum epoch time for training
  -maxlength MAX_LENGTH,    maximum words for each sentence
  -maxsentece MAX_SENTECE,  limit the maximum length for training. Set lower (e.g 5) for faster training speed. If not specify, program will train entire corpus.

  # Parameters for display
  -getloss GET_LOSS,        print out average loss every `GET_LOSS` iterations.
  -epochsave SAVE_MODEL,    save the model every `SAVE_MODEL` epochs.
  -outputfile OUTPUT_FILE,  starting name for saving the model. During training, encoder and decoder would be saved as `[OUTPUT_FILE]_[encoder|decoder]_[iter_time]` every `SAVE_MODEL` epochs.

```


With above arguments, a variety of configurations could be trained:
```python
python train.py # This will train using default settings in `train/settings.py`

python train.py -embed 300 -lr 0.01 -batch 3 -getloss 20 -encoder HierarchicalRNN 
                -decoder HierarchicalRNN -epochsave 12 -copy True -copyplayer False 
                -gradclip 2 -layer 2 -epoch 3 -outputfile pretrain_copy 
                -pretrain hbilstm -iternum 200
                # -pretrain and -iternum must be specified together
                # the corresponding pretrained model name will be in the format:
                #    [pretrain]_[encoder|decoder|optim]_[iter_num]

python train.py -embed 720 -lr 0.02 -batch 3 -getloss 10 -encoder HierarchicalLIN 
                -decoder HierarchicalRNN -epochsave 5 -copy True -copyplayer True 
                -gradclip 3 -maxsentence 800  -epoch 3

python train.py -embed 512 -lr 0.03 -batch 3 -getloss 10 -encoder BiLSTM 
                -decoder RNN -epochsave 12 -copy True -copyplayer True 
                -gradclip 3 -maxsentence 230 -layer 2 -epoch 3

python train.py -embed 600 -lr 0.02 -batch 3 -getloss 10 -encoder LIN 
                -decoder RNN -epochsave 12 -copy True -copyplayer False 
                -gradclip 5 -maxsentence 800 -layer 2 -epoch 3
```

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

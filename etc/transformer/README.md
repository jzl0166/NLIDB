
# transformer model

## minor revision

- register ```tensor2tensor.data_generators.NLIDB_wiki``` in data_generators/all_problems.py

- add NLIDB_wiki.py in data_generators folder

- replace utils/modality.py

- replace layers/modalities.py

## how to run

```
PROBLEM=nlidb_wiki
MODEL=transformer
HPARAMS=transformer_base_single_gpu
DATA_DIR=$HOME/t2t_data
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS
mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR
BEAM_SIZE=4
ALPHA=0.6
```
```
# Generate data
t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM
```

```
# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR 
```

```
# Decode
BEAM_SIZE=4
ALPHA=0.6

DECODE_FILE=/home/gongzhitaao/ww/wiki/test.qu
OUT_FILE=/home/gongzhitaao/ww/wiki/test_inf
t2t-decoder \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=$DECODE_FILE \
  --decode_to_file=$OUT_FILE

DECODE_FILE=/home/gongzhitaao/ww/wiki/dev.qu
OUT_FILE=/home/gongzhitaao/ww/wiki/dev_inf
t2t-decoder \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=$DECODE_FILE \
  --decode_to_file=$OUT_FILE
```





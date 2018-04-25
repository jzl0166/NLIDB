# NLIDB
a NLIDB seq2seq model that achieves high transfer-ability. 

## Dependencies
- TF 1.4
- python 2.7
- python3 (for annotating WikiSQL)
- tqdm
- editdistance

## How to use

Since data has been preprocessed and stored in numpy files. You can either omit first two steps go straight to train/load model or rebuild dataset.
- Set PATH (optional)
  
      export WIKI_PATH={PATH to WikiSQL}
      export GLOVE_PATH={PATH to GloVe}
  Store 'glove.840B.300d.txt' in GLOVE_PATH.

- Prepare WikiSQL for training and evaluation (optional)

  1. Annotate WikiSQL
  
     Annotated data has been saved in data/DATA/overnight_source.
     
         python3 utils/annotation/annotate.py
    
  2. Annotate Overnight
  
     Annotated data has been saved in data/DATA/wiki.
  
         python utils/annotate_overnight_vocab.py
         python utils/denotate_overnight.py
         
  3. Prepare Glove
      
     Edit path for 'glove.840B.300d.txt', set rebuild = True.
     
         python utils/glove.py
      
  4. Build data
      
     Data has been stored in data folder.
      
         python utils/both.py
      
- Train or load model 
    
   pretrained model https://drive.google.com/open?id=13cFjU9tEZ1hx-exBuq6qpYgSrlzKyGXl
   
   please put all files in model/ folder
   
      python nlidb.py --mode train
      python nlidb.py --mode load_pretrained
      
- Transfer-ability
      
      python nlidb.py --mode transfer

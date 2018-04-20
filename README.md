# NLIDB
a NLIDB seq2seq model that achieves high transfer-ability. 

## Config
- TF 1.4
- python 2.7
- python3 (for annotating WikiSQL)

## How to use
- set path

- prepare WikiSQL for training and evaluation

  1. Annotate WikiSQL
  
         python3 utils/annotation/annotate.py
    
  2. Annotate Overnight
  
         python utils/denotate_overnight.py
         
  3. prepare Glove
      
     edit path for 'glove.840B.300d.txt', set rebuild = True.
     
         python utils/glove.py
      
  4. Load data
      
         python utils/both.py
      
- Train or load model 
    
      python nlidb.py --mode train
      python nlidb.py --mode load_pretrained
      
- Transfer-ability
      
      python nlidb.py --mode transfer

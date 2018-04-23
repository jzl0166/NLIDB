# NLIDB
a NLIDB seq2seq model that achieves high transfer-ability. 

## Dependencies
- TF 1.4
- python 2.7
- python3 (for annotating WikiSQL)
- tqdm

## How to use
- Set PATH (optional)
  
      export WIKI_PATH = '{PATH to WikiSQL}'
      export GLOVE_PATH = '{PATH to GloVe}'

- Prepare WikiSQL for training and evaluation

  1. Annotate WikiSQL (optional)
  
     Annotated data has been saved in data/DATA/overnight_source
     
         python3 utils/annotation/annotate.py
    
  2. Annotate Overnight (optional)
  
     Annotated data has been saved in data/DATA/wiki
  
         python utils/denotate_overnight.py
         
  3. Prepare Glove
      
     edit path for 'glove.840B.300d.txt', set rebuild = True.
     
         python utils/glove.py
      
  4. Load data
      
         python utils/both.py
      
- Train or load model 
    
      python nlidb.py --mode train
      python nlidb.py --mode load_pretrained
      
- Transfer-ability
      
      python nlidb.py --mode transfer

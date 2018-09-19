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
  Store 'glove.840B.300d.txt' from https://nlp.stanford.edu/projects/glove/ in GLOVE_PATH.
  
  Store raw dataset from https://github.com/salesforce/WikiSQL in WIKI_PATH.
  
- Prepare WikiSQL for training and evaluation (optional)

  1. Annotate WikiSQL
  
     Annotated data has been saved in data/DATA/wiki.
     
         python3 utils/annotation/annotate.py
    
  2. Annotate Overnight
  
     Annotated data has been saved in data/DATA/overnight_source.
  
         python utils/annotate_overnight_vocab.py
         python utils/denotate_overnight.py
         
  3. Prepare Glove
      
     Edit path for 'glove.840B.300d.txt', set rebuild = True.
     
         python utils/glove.py
      
  4. Build data
      
     Data has been stored in data folder.
      
         python utils/both.py
      
- Train or load model 
    
   pretrained model https://drive.google.com/open?id=1nugvgpLwuc9o2uRuSU5cLM1LHu4MrrqJ
   
   please put all files in model/ folder
   
      python nlidb.py --mode train
      python nlidb.py --mode load_pretrained
      
- Transfer-ability
      
  The annotated dataset used for transfer-ability evaluation is adopted from previous work https://github.com/alantian/nlidb . We have extracted related code into our etc/overnight-tagger folder.
      
      python nlidb.py --mode transfer

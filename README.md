# NLIDB
a NLIDB seq2seq model that achieves high transfer-ability. 

- prepare WikiSQL for training and evaluation

  1. Annotate WikiSQL
  
         python utils/annotation/annotate.py
    
  2. Annotate Overnight
  
         python utils/denotate_overnight.py
      
  3. Load data
      
         python utils/both.py
      
- Train or load model 
    
      python run_model.py --mode train
      python run_model.py --mode load_pretrained
      
- Transfer-ability
      
      python run_model.py --mode transfer

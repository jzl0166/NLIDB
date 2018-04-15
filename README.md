# NLIDB
a NLIDB seq2seq model that achieves high transfer-ability. 

- prepare WikiSQL for training and evaluation

  1. Annotate WikiSQL
  
      annotation/annotate.py
    
  2. Annotate Overnight
  
      utils/denotate_overnight.py
      
  3. Load data
      
      utils/both.py
      
- Train or load model 
    
      run_model.py --train
      
      run_model.py --load_pretrained
      
- Transfer-ability
      
      run_model.py --transfer

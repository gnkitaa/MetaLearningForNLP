# MetaLearningForNLP
This code uses BERT model to obtain textual representations and uses prototypical networks with triplet loss to learn a metric space. 

Command to run : python main.py

1. main.py : Calls the train function to train prototypical networks. Specify following arguments before running.
    a. dataset_root : folder containing train, val and test files
    b. experiment_root : folder where the models should be saved
    c. pretrained_model_path : path to a pre-tarined model if exists. This model will be loaded and the training will start from this point.
    
2. loss.py : Implements prototypical network loss modified to incorporate triplet loss. 
3. model.py : Implements BERT pre-trained model from hugging face. Embedding of <CLS> token from last layer is used as sentence embedding. 
              Specify whether to keept the BERT layers frozen or not in the arguments section of main.py
4. data.py : file to process data processing (BERT Tokenizer and padding)


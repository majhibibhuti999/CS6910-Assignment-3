# CS6910-Assignment-3
Assignment 3 Submission for CS6910 Fundamentals of Deep Learning

`Bibhuti Majhi CS22M031`

## Question 1

The ipynb file reads the data from akshantar_sampled folder uploaded into kaggle.
I chose the Hindi Language dataset for the task of Transliteration.
Using pandas read all three train,valid and test data.
Created Functions for Encoder,Decoder and Seq2Seq.

## Question 2

Created Functions for Encoder,Decoder,Valevaluate(evaluates the validation data),withoutattention function(used in sweeps) and sweep configuration

The sweep configuration is :
```python

sweep_configuration = {
    'method' : 'bayes',
    'metric' : { 'goal' : 'maximize',
    'name' : 'validation_accuracy'},
    'parameters':{
        'batchsize' : {'values' : [64,128,256,512,1024]},
        'input_embedding_size' : {'values' : [128,256,512,1024]},
        'no_of_layers' : {'values' : [1,2,3,4,5,6,7,8]},
        'hidden_size' : {'values' : [128,256,512,1024]},
        'cell_type' : {'values' : ['RNN','GRU','LSTM']},
        'bidirectional' : {'values' : ['Yes']},
        'dropout' : {'values' : [0.1,0.2,0.3,0.4,0.5]},
        'epochs' : {'values' : [10,20,30]}
    }
}

```

### Steps to build the Seq2Seq Network.

Inside the .ipynb file have created Functions for encoder,decoder and train that takes instances of encoder and decoder with specified parameters

An Instance of Encoder is as follows:

```python
 encoder = Encoder(char_embed_size,hidden_size,no_of_layers,dropout,rnn)
```

It can be implemented using the following parameters:

- char_embed_size = Embedding size required to get a representation of a character.

- hidden_size = Size of cell state of RNN,LSTM,GRU

- no_of_layers = no of stacks of RNN,LSTM,GRU one upon another.

- dropout = ranges between 0-1. denotes the probability to dropout.

- rnn = LSTM or GRU or RNN

An Instance of a Decoder is as follows:

``` python
 decoder = Decoder(char_embed_size,hidden_size,no_of_layers,dropout,rnn).to(device)
```
Parameters are same as Encoder. Only difference is Encoder is bidirectional.
  
### Training the Seq2Seq Network


The model can be trained using the `Train` method.

- for an instance of Seq2Seq network given earlier we can train it by calling the train function as follows:

```python
    Encoder1,Decoder1 = train(batchsize,hidden_size,char_embed_size,no_of_layers,dropout,epochs,rnn)
```

## Question 4

Evaluated Accuracy for Test data after getting the best configuration from sweeps in Q2.ipynb

Best configuration is :
 ``` python
    batchsize = 128
    hidden_size = 1024
    char_embed_size = 128
    no_of_layers = 2
    dropout = 0.5
    epochs = 20
    rnn = 'LSTM'
 ```
 - Stored the predictions using the Evaluate function :
 
 ``` python
    test_loss,test_accuracy,predictions = Evaluate(False,test_eng_word,test_hin_word,Encoder1,Decoder1,batchsize,hidden_size,char_embed_size,no_of_layers)
 ```
 - Using this predictions which is a dataframe having columns as original and predicted hindi words.
 
 - Store this using to_excel method of pandas into a folder Prediction_vanilla

## Question 5:

Evaluated Accuracy for Test data after getting the best configuration from sweeps for Attention model.

Best Configuration is :

``` python
batchsize = 512
hidden_size = 1024
char_embed_size = 1024
no_of_layers = 1
dropout = 0.4
epochs = 30
rnn = 'LSTM'
```

- Got the test_loss,Test_accuracy and predictions using the Evaluate function:

``` python
test_loss,test_accuracy,predictions = Evaluate(True,test_eng_word,test_hin_word,Encoder1,Decoder1,batchsize,hidden_size,char_embed_size,no_of_layers)
```

- Using this predictions which is a dataframe having columns as original and predicted hindi words .

- Store this using to_excel method of pandas into a folder Prediction_vanilla.
  
## Instructions about Train.py

- For train.py you need to login into your wandb account from the terminal using the api key.

- In the arguments into the terminal please give valid credentials like project_name and entity_name.

- List of argument supported by train.py is :

``` python
    -wp,--wandb_project = wandb project name
    -we,--wandb_entity = wandb entity name
    -e,--epochs = no of epochs
    -b,--batchsize = size of batch
    -hidden,--hidden_size = size of cell state
    -embed, -- embedding_size = embedding size
    -cell,--cell_type = cell type choices = ['LSTM','GRU','RNN']
    -drop,--dropout = probablity of dropout
    -attn,--attentionRequired = choices = [True,False]
    -layer,--no_of_layers = no of stack of RNN or LSTM or GRU
    
```

- For calling train.py with appropriate argument please follow the given example:

  `python train.py -wp projectname -we entity name -e 10 -b 128 -hidden 1024 -embed 1024 -cell 'LSTM' -drop 0.3 -attn True -layer 2 
  




  

  

  

  
  

 

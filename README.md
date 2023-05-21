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

  Created the confusion matrix for Test dataset using wandb's prexisting function:
```python
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  wandb.init(entity = 'cs22m031',project = 'CS6910_DL_assignment_1',name = 'Confusion Matrix')
  wandb.log({"conf_mat" : wandb.sklearn.plot_confusion_matrix(
                        Y_test,predicted,
                        class_names)})
```

## Question 10:

The best configurations used for mnist dataset is:

Configuration 1:

```python
        'optimizer' : { 'values' : ['nadam']},
        'batchsize' : { 'values' : [256]},
        'no_of_features' : {'values' : [784]},
        'no_of_classes' : {'values' : [10]},
        'no_of_layers' : { 'values' : [6]},
        'no_of_neurons' : {'values' : [512]},
        'max_epochs' : {'values' : [15]},
        'eta' : { 'values' : [1e-4]},
        'initialization' : { 'values' :['he']},
        'activation' : { 'values' : ['relu']},
        'loss' : { 'values' : ['cross']},
        'weight_decay'  : { 'values' : [0]}
```
  Training accuracy = 99.91852
  Validation accuracy = 98.08333
  
Configuration 2:

```python
        'optimizer' : { 'values' : ['nadam']},
        'batchsize' : { 'values' : [256]},
        'no_of_features' : {'values' : [784]},
        'no_of_classes' : {'values' : [10]},
        'no_of_layers' : { 'values' : [5]},
        'no_of_neurons' : {'values' : [256]},
        'max_epochs' : {'values' : [15]},
        'eta' : { 'values' : [0.002]},
        'initialization' : { 'values' :['he']},
        'activation' : { 'values' : ['relu']},
        'loss' : { 'values' : ['cross']},
        'weight_decay'  : { 'values' : [0]}
```
  Training Accuracy = 99.83704
  Validation Accuracy = 98.0166
  
Configuration 3:

```python
        'optimizer' : { 'values' : ['nadam']},
        'batchsize' : { 'values' : [128]},
        'no_of_features' : {'values' : [784]},
        'no_of_classes' : {'values' : [10]},
        'no_of_layers' : { 'values' : [6]},
        'no_of_neurons' : {'values' : [256]},
        'max_epochs' : {'values' : [10]},
        'eta' : { 'values' : [1e-3]},
        'initialization' : { 'values' :['he']},
        'activation' : { 'values' : ['relu']},
        'loss' : { 'values' : ['cross']},
        'weight_decay'  : { 'values' : [0]}
```
  
  Training Accuracy = 99.77407
  Validation Accuracy = 97.8667
  
## Instructions about Train.py

- For train.py you need to login into your wandb account from the terminal using the api key.

- In the arguments into the terminal please give valid credentials like project_name and entity_name.

- You can add new initialization method by adding member function in the class NeuralNetwork and adding the name of the initializer into the Initialization_list dictionary.

- You can add new activation functions by  adding member function in the class NeuralNetwork and adding the name of the activation function into the activation_list dictionary.

- You can add new optimization algorithms by adding the code in the space specified inside the fit method.

- For calling train.py with appropriate argument please follow the given example:

  `python train.py -e 1 -w_i Xavier -b 32 -o nag -nhl 3 -sz 128 -d fashion_mnist -l mean_squared_error -lr 0.001`
  




  

  

  

  
  

 

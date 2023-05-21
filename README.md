# CS6910-Assignment-3
Assignment 3 Submission for CS6910 Fundamentals of Deep Learning

`Bibhuti Majhi CS22M031`

## Question 1

The ipynb file reads the data from akshantar_sampled folder uploaded into kaggle.
I chose the Hindi Language dataset for the task of Transliteration.
Using pandas read all three train,valid and test data.
Created Functions for Encoder,Decoder and Seq2Seq.

## Question 2-4

Created Functions for Encoder,Decoder,Valevaluate(evaluates the validation data),withoutattention function(used in sweeps) and sweep configuration

The sweep configuration is :
```python

sweep_configuration = {
    'method' : 'bayes',
    'metric' : { 'goal' : 'maximize',
    'name' : 'validation_accuracy'},
    'parameters':{
        'batchsize' : {'values' : [1024]},
        'input_embedding_size' : {'values' : [128]},
        'no_of_layers' : {'values' : [2]},
        'hidden_size' : {'values' : [256]},
        'cell_type' : {'values' : ['LSTM']},
        'bidirectional' : {'values' : ['Yes']},
        'dropout' : {'values' : [0.3]},
        'epochs' : {'values' : [10]}
    }
}
```

### Steps to build the Neural Network.

Inside the class NeuralNetwork I have written necessary functions for the implementing Forward propagation,Back propagation and different Gradient descent algorithms.

An Instance of NeuralNetwork is as follows:

```python
NN = NeuralNetwork(optimizer,batchsize,no_of_features,no_of_classes,no_of_layers,no_of_neurons_in_each_layer, \
                    max_epochs,eta,initialization_method,activation_method,weight_decay, \
                    epsilon,momentum,beta,beta1,beta2)
```

It can be implemented using the following parameters:

- optimizer:
  
  The optimizer value is passed as a string. Inside the Class NeuralNetwork, there is a fit method that has if-else statements for selecting the specified optimizer that is passed.
  
- batchsize:

  The batchsize is passed as integer that specifies the size of minibatch of datapoints needed for different gradient descent algorithms.
  
- no_of_features:

  The no_of_features is passed as integer denoting the size of the flatten datapoints.
  
- no_of_classes:

  The no_of_classes is passed as integer denoting the no of labels of the dataset used.
  
- no_of_layers:
  
  no_of_layers is passed as integer denoting the no of layers used in the network including input layer, output layer and all the hidden layers.
 
- no_of_neurons_in_each_layers:

  no_of_neurons_in_each_layers is passed as a list of integers which denotes the sizes of hidden layers.
 
- max_epochs:

  max_epochs is passed as integer which denotes the no of times gradient_descent algorithm is called.
  
- eta:
  
  eta is passed as float which denotes the learning rate.
  
- initialization_method:
 
  initialization_method is passed as a string denoting the initialization method used to initialize the weights of the network. There is an   `Initialization_list` which is a dictionary of `key:string value:function pair` that stores all the types of initialization functions.
  
- activation_method:
  
  activation_method is passed as a string denoting the activation function used in the network. There is an `activation_list` which is a dictionary of  `key:string value:function` pair that stores all the types of activation functions.
  
- weight_decay:
  
  weight_decay is passed as a float. This is used for implementing `L2 regularization` during the weight updates of the network.
  
- epsilon:
  
  epsilon is passed as float. This is a very small number used be optimizers.
  
- momentum:
  
  momentum is passed as float. This is used for momentum and Nesterov optimizers.
  
- beta:
  
  beta is passed as float. This is used for RMSprop optimizers.
  
- beta1:

  beta1 is passed as float. This is used for adam and nadam optimizers.
  
- beta2
  
  beta2 is pased as float. This is used for adam and nadam optimizers.
  
### Training the Neural Network


The model can be trained using the `NeuralNetwork.fit()` method.

- for an instance of Neural network given earlier passing the data as X_train as the training images and Y_train as the corresponding labels:

```python
    NN.fit(X_train,Y_train)
```
- inside the fit method using

```python
thetas = NN.initialization_list[self.initialization_method] 
```

  we initialize the weights of the given network.

- For NN.max_epochs using both forward and backward propagation algorithm for different optimizers
```python
activation,preactivation = NN.feed_forward(X_train[x:x+NN.batchsize],thetas,NN.no_of_layers)
grads = NN.back_propagate(activation,preactivation,thetas,Y_train[x:x+NN.batchsize])
```

### Testing the Neural Network

- For testing purpose the class NeuralNetwork has `NN.predict`, `NN.accuracy_score`, `NN.compute_loss`.

```python
  predictions = NN.predict(X_test)
  accuracy = NN.accuracy_score(Y_test,predicted)
  loss = NN.compute_loss(predictions,Y_test)
```

## Question 7

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
  




  

  

  

  
  

 

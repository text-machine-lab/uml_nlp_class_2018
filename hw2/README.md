# Assignment #2

In this homework you'll train a neural language model on a subset 
of Donald Trump's and Barak Obama's tweets.
Upon completion, you will have a model that is able to generate tweets that are similar 
to their tweets but not exactly the same.

This assignment consists of several consequent steps:

1. Loading the data
2. Building a neural language model using a recurrent neural network cell provided by PyTorch
3. Implementing the GRU cell from scratch

After finishing the third item, you need to train a language model 
on the Trump's tweets and observe the behaviour of the losses 
on training (called `dataset_train` in the code) and validation (`dataset_val`) sets,
taken from his tweets, as well as on the Obama's tweets (`dataset_val_ext`), and vice versa 
(you can control the dataset used for training with the `train_on` variable). 


## Structure of this template
The code for this assignments is split into several files:
1. `train.py` - the main file which contains the class for the language model itself as well as the training loop.
2. `utils.py` - contains different utility functions. For the purpose of this assignment you do not need to modify it.
3. `dataset.py` - contains dataset classes. See a details description below in the section "Loading the data"
4. `gru.py` - contains a class for the GRU cell that you need to implement.

The directory `data` contains several data files that you can use. 
In particular, the file `trump_tweets.txt` contains 30649 Donald Trump's tweets that we will use in this assignment. 


## Loading the data

The file `dataset.py` contains a `LanguageModelDataset` class that, given a list of sentences, builds 
the vocabulary and implements the required `__len__` and `__getitem__` methods.

Notice how the `__getitem__` method pads the sentences with padding token so they all have the same length.
For example, a sentence `['I', 'am', 'great']` would be padded to the maximum length of 5 
as follows: `['I', 'am', 'great', '<pad>', '<pad>']`. By doing such padding, we can combine 
several sentences into a batch of samples in the form of a single matrix and efficiently process it on a GPU. 


Your goal is to implement the `_load_file` method iin the `TwitterFileArchiveDataset` class. This method 
should read the file, specified in the `filename` argument and return a list of sentence, where each sentence 
is represented as a list of tokens.

For example, for the following content
```text
I would like to wish everyone A HAPPY AND HEALTHY NEW YEAR. WE MUST ALL WORK TOGETHER TO, FINALLY, MAKE AMERICA SAFE AGAIN AND GREAT AGAIN!
I would feel sorry for @JebBush and how badly he is doing with his campaign other than for the fact he took millions of $'s of hit ads on me
My ties &amp; shirts at Macy’s are doing great. Stupid @GoAngelo is making people aware of how good they are!
...
```  
it should produce the folowing:
```python
[
    ['I', 'would', 'like', 'to', ...],
    ['I', 'would', 'feel', 'sorry', 'for', '@JebBush', 'and', ...],
    ['My', 'ties', '&', 'shirts', ...]
]
```

Notice how:
1. The token `@JebBush` was correctly recognized as a Twitter's username and returns as a single token, 
and not two tokens `@` and `JebBush`. To achieve such behaviour, you can use the `TweetTokenizer` class from NLTK.
2. The HTML entity `&amp;` was transformed into the corresponding character `&`.

Your code should handle such cases, as well as remove any http links from the tweets.

## Building a neural language model

The file `train.py` contains a template code with the `NeuralLanguageModel` class and a basic training loop.

The `NeuralLanguageModel` class implements a neural language model and should has three methods: `__init__`, `forward`, 
and `produce`.

The `__init__` methods takes the following arguments:

 - `embedding_size` - the size of the embeddings (word vectors). Note that we do not use pretrained vectors here and 
just initialize them randomly and train during the training of the network.    
 - `hidden_size` - the size of the GRU cell.  
 - `vocab_size` - the size of the vocabulary (i.e. the number of unique tokens in the dataset).  
 - `init_token` - the index of a special "init" token. This token serves as the initial input during the process 
of generation of a sentence.     
 - `eos_token` - the index of a special "end-of-sentence" token. Every sentence ends with this token, and the generation 
of a sentence will stop when it reaches this token.   
 - `teacher_forcing` - the probability of using the truth input token at the current timestep insted of 
the output of the model at the previous timestep. See section 10.2.1 of 
the [Deep Learning Book](http://www.deeplearningbook.org/contents/rnn.html) 
or [this blog post](https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/) for details.  

This method should declare all layers that are needed. In our case just three layers should be enough:

 - An embedding layer
 - A GRU cell (either `torch.nn.GRUCell` or your implementation)
 - A projection layer (similar to the previous homework)
 

The `forward` method accepts a batch of sentences in form of token indices and performs 
the forward pass of the network.

More precisely, at each timestep it performs the following: 
1. Obtains embeddings of the input tokens on the current timestep
2. Passes them into the GRU cell alongside with its hidden state from the previous timestep 
3. Passes the current hidden state trough a fully connected layer to get a non-normalied probability 
distribution over the tokens in the vocabulary - the output of the network at each timestep.
4. If rand() < teach forcing probability, use the truth input token as the input to the next timestep, otherwise use 
the current output of the network to get the most probable token (that is, the token with maximum probability) 
and use it the input to the next timestep 

Note that at the timestep 0 (the very first timestep) we should use the "init" token as the input, 
and a matrix of zeros as the hidden state. The method `cell_zero_state` creates such hidden state.  


The `produce` method is very similar to the `forward` method. Given a list of starting tokens and a maximum length
of a sentence, it should just run the GRU cell and produce the output tokens until either the maximum length is reached
or the network produced the "end-of-sentence" token. The list of starting tokens should be used as the input 
to the network; after it has been extinguished, the output of the network at the previous timestep should be used 
as the input on the current timestep. 

For example, if the list of starting tokens contains two word: `['This', 'is']`, the input to the network, 
the hidden states of the GRU, and the outputs should be as follows:

Timestep 0:  
x_i = init token  
hidden = zero state  
output = This  

Timestep 1:  
x_i = This  
hidden = hidden from timestep 0
output = is     

Timestep 2:  
x_i = is  
hidden = hidden from timestep 1    
output = sample from the probability distribution over the tokens   

Timestep 3:  
x_i = output from timestep 2  
hidden = hidden from timestep 2    
output = sample from the probability distribution over the tokens  

Timestep 4:  
x_i = output from timestep 3  
hidden = hidden from timestep 3    
output = sample from the probability distribution over the tokens

...

You might want to use functions `torch.nn.functional.softmax` and `torch.multinomial` in this method.  
  

## Implementing the GRU cell from scratch

You need to implement a GRU cell in the `GRUCell` class in the file `gru.py`.
Use the following paper as the reference (section 3.2):  

Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). 
Empirical evaluation of gated recurrent neural networks on sequence modeling. 
[arXiv](https://arxiv.org/abs/1412.3555)


# What to submit
Submit all files in the `hw2/` in this repository with your code written in the marked places.
Please make sure that you completed the code in the files `datasest.py`, `train.py`, and `gru.py`. 
You code should be able to run in docker using the provided `docker_build.sh` and `docker_run.sh` scripts.

In addition, you should submit two plots which reflect the losses 
on the training and two validation sets as the training progresses 
(in other words, on the X-axis you should have epochs 
and on the Y-axis you should have the values of the losses).
 
The first plot should be produced using the Trump's tweets for training 
and the second plot should be produced using the Obama's tweets for training.

Submit as follows using the submit utility available on the cs.uml.edu machines: 
```bash
$ submit arum hw2 items-to-submit
```
where `items-to-submit` can be the `hw2` directory for simplicity.

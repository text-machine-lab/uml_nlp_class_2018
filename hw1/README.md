# Assignment #1

In this assignment you will implement and train 
a [Word2Vec](https://en.wikipedia.org/wiki/Word2vec) 
skip-gram model [\[1\]](https://arxiv.org/abs/1301.3781)
using the [Text8](http://mattmahoney.net/dc/text8.zip) data.

If you choose to use the provided code, you will familiarize yourself 
with the [PyTorch library ](http://pytorch.org/) and its build-in widely used 
classes for datasets and models.


## What you should do
You should provide your code along with a `Dockerfile` 
which builds an image that, upon running, trains 
a Word2Vec skip-gram model and prints the loss and the closest words 
(according to the 
[cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity))
for the words in the `test_words.txt` file in this repository 
as training progresses  in the following format:
```
Starting training...
Iteration: 1000 loss: 7.29280709028244
Closest to one: five, two, three, nine, four
Closest to are: were, chengdu, mice, impiety, is
Closest to you: installing, lderlin, src, kazan, detract
Closest to time: mixture, bread, grind, gentoo, cv

Iteration: 2000 loss: 6.86459051990509
Closest to one: two, five, three, nine, four
Closest to are: were, is, impiety, emerson, chengdu
Closest to you: installing, lderlin, kazan, libro, moabite
Closest to time: cv, gentoo, bnp, bread, vampyre

...
``` 



## Code template
We provide a base code template written in Python3 
which you can use as a starting point. 
However, feel free to write your code from scratch 
or use a different programming language, as long as you provide 
the corresponding `Dockerfile` to run it.  

### Overview

The most of the code, including the loading of the data and the training loop,
is already written for you. All you need to do is to implement 
some functions in the `SkipGramDataset` and `SkipGramModel` classes. 

First, you would need to implement the `__len__` and the `__getitem__` methods 
of the `SkipGramDataset` class.  

#### `SkipGramDataset` class 

This class is based on the 
[`torch.utils.data.Dataset`](http://pytorch.org/docs/0.3.0/data.html#torch.utils.data.Dataset) 
class which is necesseraily
in order to use the standard 
[`DataLoader`](http://pytorch.org/docs/0.3.0/data.html#torch.utils.data.DataLoader) 
class from PyTorch.

The `__len__` method should return 
the total length of the dataset. Note that in our case the length of the dataset
depends on the skip window (the `skip_window` parameter), since we can choose the context token on the left 
or on the right sides of the target token.
For example, consider the following dataset, consisted of the following 
tokens: `['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first', 'used', 'against']`.
With the skip-window 2, the minimum index of the target word 
is 2 (the third token), since we need to be able to index 
the context words in the left. The same applies to the maximum index. 
Thus, this method should correctly calculates the length of the dataset
based on the skip window size.

The `__getitem__` method should accept an index of the target word 
and randomly choose a context word from the context on the left or on 
the right inside the context window.
For example above, if the `index` is 0 and the skip window is 2 
it would correspond to the target word `'as'` and this method would 
randomly a context word from the words 
`'anarchism'`, `'originated'`, `'a'`, or `'term'`.


#### `SkipGramModel` class 
This class is based on the 
[`torch.nn.Module`](http://pytorch.org/docs/0.3.0/nn.html#torch.nn.Module) 
class which is a base class for all neural network modules in PyTorch and 
user-defined models should subclass it. 

In general, you need to implement two method: `__init__` and `__forward__`.

The `__init__` method should decalare all layers or parameters of the network that
you are going to use. For example, in our case you will need two layers:
1. An embedding layer 
([torch.nn.Embedding](http://pytorch.org/docs/0.3.0/nn.html#torch.nn.Embedding)) 
that would convert words indices into dense vectors 
by just selecting an appropritate vectors at the corresponding position.
2. A projection layer 
([torch.nn.Linear](http://pytorch.org/docs/0.3.0/nn.html#torch.nn.Linear))
that would transform the embedding of a word into a probability distribution 
over the context words. Note that the used loss function 
([torch.nn.CrossEntropyLoss](http://pytorch.org/docs/0.3.0/nn.html#torch.nn.CrossEntropyLoss))
accepts unnormalized probabilities so you do not need an activation function here.

You should declare the mentioned above layers in the `__init__` method.

The `forward` method performs the forward pass on the network on the input data. 
In our case that means using the embedding and the projection layers to get the output
distribution over the vocabulary for every input word in the batch.


### Running the code
First, build the image uisng the `docker_build.sh` script. 
Next, run the code with the `docker_run.sh` script. 

If successful, you should see an output similar the the following:
 ```
Data: 17005207
['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first', 'used', 'against']
Vocab size: 50001
Unknown tokens: 418382
Dataset: 17005207
[5237, 3083, 12, 6, 195, 2, 3135, 46, 59, 156]
Starting training...
Iteration: 1000 loss: 7.29280709028244
Closest to one: five, two, three, nine, four
Closest to are: were, chengdu, mice, impiety, is
Closest to you: installing, lderlin, src, kazan, detract
Closest to time: mixture, bread, grind, gentoo, cv
Closest to thai: tonk, recalling, bluish, guidebook, scraped
Closest to luciano: cartoonish, damages, parking, heartbeat, soup
Closest to painted: documented, murdock, cukor, induced, santorum

Iteration: 2000 loss: 6.86459051990509
Closest to one: two, five, three, nine, four
Closest to are: were, is, impiety, emerson, chengdu
Closest to you: installing, lderlin, kazan, libro, moabite
Closest to time: cv, gentoo, bnp, bread, vampyre
Closest to expected: choke, bluff, implement, sco, cephalopods
Closest to velocities: tet, loony, inactivity, hatchet, botanists
Closest to nine: six, eight, five, one, zero

...
```

## Reading materials

1. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). 
Efficient estimation of word representations in vector space. 
*arXiv preprint arXiv:1301.3781.* [\[arXiv\]](https://arxiv.org/abs/1301.3781)
2. [Deep Learning with PyTorch: A 60 Minute Blitz](http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
3. [Get started with Docker](https://docs.docker.com/get-started/)


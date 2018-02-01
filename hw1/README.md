# Assignment #1

This homework consists of two parts. In the first part you will use 
the [fastText](https://fasttext.cc/) [\[1\]](https://arxiv.org/abs/1607.04606) library to train 
your very own word vectors on the [Text8](http://mattmahoney.net/dc/text8.zip) data.
In the second part you will implement and train 
a [Word2Vec](https://en.wikipedia.org/wiki/Word2vec) 
skip-gram model [\[2\]](https://arxiv.org/abs/1301.3781)
on the same data.

If you choose to use the provided code, you will familiarize yourself 
with the [PyTorch library ](http://pytorch.org/) and its build-in widely used 
classes for datasets and models.


## Part 1 - fastText library
The [fastText](https://fasttext.cc/) library implements an algorithm, 
described in [\[2\]](https://arxiv.org/abs/1301.3781)
and efficiently learns vectors representation of words. In addition, it can generate vectors for
out-of-vocabulary words and preform text classification. 

To use this library in a docker container, build the image with the `docker_build.sh` script 
and then run `docker_run_part1.sh`.


### Training word vectors
To train a skip-gram model on the provided data run `fasttext` as follows:
```bash
$ fasttext skipgram -input text8.txt -dim 128 -output model_text8
```

This command will train a model and save it in a binary format in the file `model_text8.bin`.
The trained vectors are be saved in the file `model_text8.vec`. 

To see the vectors themselves, you can use the following command: 
```bash
$ head -n 300 model_text8.vec | tail
```

You can print word vectors for the word in the file `test_words.txt` as follows:
```bash
$ cat test_words.txt | fasttext print-word-vectors model_text8.bin
```

To query the closest words for a given word in an interactive mode run the `fasttext` libary with the `nn` option:
```bash
$ fasttext nn model_text8.bin
```

### Examine your word vectors

Please do the following:
* select 20 words from your corpus with frequency > 50 
* for each of these words, identify top-15 closest words
* cluster them into three clusters using k-means++ implementation in `scikit-learn`



## Part 2 - Training a skip-gram Word2Vec model

### What you should do
You should provide your code along with a `Dockerfile` 
which builds an image that, upon running, trains 
a Word2Vec skip-gram model and prints the loss and the closest words 
(according to the 
[cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity))
for the words in the `test_words.txt` file in this repository 
as training progresses  in the following format:
```
Starting training...
Iteration: 1000, elapsed time: 0:00:31.748593 [31 sec/1000 iters], loss: 7.296961873531342
Closest to one: two, flanagan, collage, heaven, eight
Closest to are: were, is, sampled, require, competed
Closest to you: haldeman, poised, orpheus, astros, cessna
Closest to time: sj, ayn, revisions, freebase, mushroom
Closest to lambeth: escaping, centralised, annuity, cardinal, blackmailed
Closest to to: dpp, infinitives, transsexualism, deviation, amano
Closest to is: was, are, gaul, hatta, shuttle

Iteration: 2000, elapsed time: 0:01:01.879741 [30 sec/1000 iters], loss: 6.859302348136902
Closest to one: two, seven, flanagan, eight, nine
Closest to are: were, is, sampled, require, amassing
Closest to you: haldeman, orpheus, poised, astros, i
Closest to time: sj, ayn, revisions, freebase, hazards
Closest to cells: votes, tiffany, committing, fifteen, rickenbacker
Closest to linux: christensen, negativity, liberals, figured, offer
Closest to bottle: rb, talmadge, nasal, precludes, kindred

...
``` 



### Code template
We provide a base code template written in Python3 using 
the [PyTorch](http://pytorch.org/) library
which you can use as a starting point. 
However, feel free to write your code from scratch 
or use a different programming language, as long as you provide 
the corresponding `Dockerfile` to run it.  

#### Overview

The most of the code, including the loading of the data and the training loop,
is already written for you. All you need to do is to implement 
some functions in the `SkipGramDataset` and `SkipGramModel` classes. 

First, you would need to implement the `__len__` and the `__getitem__` methods 
of the `SkipGramDataset` class.  

##### `SkipGramDataset` class 

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


##### `SkipGramModel` class 
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


#### Running the code
First, build the image using the `docker_build.sh` script. 
Next, run the code with the `docker_run_part2.sh` script. 

If successful, you should see an output similar the the following:
 ```
Data: 17005207
['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first', 'used', 'against']
Vocab size: 50001
Unknown tokens: 418382
Dataset: 17005207
[5234, 3081, 12, 6, 195, 2, 3134, 46, 59, 156]
Starting training...
Iteration: 1000, elapsed time: 0:00:31.748593 [31 sec/1000 iters], loss: 7.296961873531342
Closest to one: two, flanagan, collage, heaven, eight
Closest to are: were, is, sampled, require, competed
Closest to you: haldeman, poised, orpheus, astros, cessna
Closest to time: sj, ayn, revisions, freebase, mushroom
Closest to lambeth: escaping, centralised, annuity, cardinal, blackmailed
Closest to to: dpp, infinitives, transsexualism, deviation, amano
Closest to is: was, are, gaul, hatta, shuttle

Iteration: 2000, elapsed time: 0:01:01.879741 [30 sec/1000 iters], loss: 6.859302348136902
Closest to one: two, seven, flanagan, eight, nine
Closest to are: were, is, sampled, require, amassing
Closest to you: haldeman, orpheus, poised, astros, i
Closest to time: sj, ayn, revisions, freebase, hazards
Closest to cells: votes, tiffany, committing, fifteen, rickenbacker
Closest to linux: christensen, negativity, liberals, figured, offer
Closest to bottle: rb, talmadge, nasal, precludes, kindred

...
```

### Where should I run my code?
Since this assignment requires many matrix multiplications, it's possible to get 
a huge speed-up by running the code on a GPU. PyTorch supports both GPU and CPU and this code 
will automatically use GPU if it is available, which will give a x70 speed-up 
(30 seconds per 1000 iterations on a GPU vs 2000 seconds on a CPU).

If you do not have a GPU available, you can use one of the department's GPU machines.
Another alternative is to use 
[https://colab.research.google.com](https://colab.research.google.com), which is 
a jupyter notebook-like free to use environment. Moreover, allows users to use GPU resources.
To enable GPU support open a notebook, click "Runtime" -> "Change runtime type", and select 
"GPU" under the hardware accelerator section. Click "Save" to save thee changes. 

## What to submit

1. For part 1, submit a printout of the words you selected, along with clusters of top-K neighbours
1. For part 2, submit
* your code
* a README file explaining the overall structure of your implementation 
* a Dockerfile what else needs to be installed and how to run it.


Homework assignments should be submitted using the submit utility available on the cs.uml.edu machines. Submit as follows:
```
$ submit arum assignment-name items-to-submit
```
Please use CS submit utility to submit this assignment.



## Reading materials

1. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). 
Efficient estimation of word representations in vector space. 
*arXiv preprint arXiv:1301.3781.* [\[arXiv\]](https://arxiv.org/abs/1301.3781)
1. J. Eisenstein. NLP Notes, chapter 13 [\[pdf\]](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf)
1. Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2016). 
Enriching word vectors with subword information. 
*arXiv preprint arXiv:1607.04606.* [\[arXiv\]](https://arxiv.org/abs/1607.04606)
1. [Deep Learning with PyTorch: A 60 Minute Blitz](http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
1. [Get started with Docker](https://docs.docker.com/get-started/)
1. [PyTorch tutorials](http://pytorch.org/tutorials/)
1. [PyTorch documentation](http://pytorch.org/docs/0.3.0/)
1. [PyTorch examples](https://github.com/pytorch/examples)


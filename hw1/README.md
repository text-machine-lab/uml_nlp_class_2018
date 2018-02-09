# Assignment #1

This homework consists of two parts. In the first part you will use 
the [fastText](https://fasttext.cc/) [\[4\]](https://arxiv.org/abs/1607.04606) library to train 
your very own word vectors on the [Text8](http://mattmahoney.net/dc/text8.zip) data.
In the second part you will implement and train 
a [Word2Vec](https://en.wikipedia.org/wiki/Word2vec) 
skip-gram model [\[1\]](https://arxiv.org/abs/1301.3781)
on the same data.

If you choose to use the provided code, you will familiarize yourself 
with the [PyTorch library ](http://pytorch.org/) and its build-in widely used 
classes for datasets and models.


## Part 1 - fastText library
The [fastText](https://fasttext.cc/) library implements an algorithm, 
described in [\[4\]](https://arxiv.org/abs/1301.3781)
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
The trained vectors are saved in the file `model_text8.vec`. The format of this file is as follows:
```
<number of tokens> <vectors dim>
<token 1> <val1> <val2> ... <val vectors dim>
<token 2> <val1> <val2> ... <val vectors dim>

<token number of tokens> <val1> <val2> ... <val vectors dim>
``` 

For example, the actual `.vec` file can look as this:
```
2000000 300
, -0.0282 -0.0557 -0.0451 -0.0434 0.0712 -0.0855 -0.1085 -0.0561 -0.4523 -0.0202 0.0975 0.1047 0.1962 -0.0693 0.0213 -0.0235 0.1336 -0.0420 -0.0564 -0.0798 0.0424 -0.0409 -0.0536 -0.0252 0.0135 0.0064 0.1235 0.0461 0.0120 -0.0372 0.0650 0.0041 -0.1074 -0.0263 0.1133 -0.0029 0.0671 0.1065 0.0234 -0.0160 0.0070 0.4355 -0.0752 -0.4328 0.0457 0.0604 -0.0740 -0.0055 -0.0089 -0.2926 -0.0545 -0.1519 0.0990 -0.0193 -0.0050 0.0511 0.0404 0.1023 -0.0128 0.0488 -0.1567 -0.0759 -0.0190 0.1442 0.0047 -0.0186 0.0140 -0.0385 -0.0853 0.1572 0.1770 0.0084 -0.0250 -0.1145 -0.0663 -0.1244 -0.3977 -0.0124 -0.4586 -0.0220 0.5746 0.0218 -0.0754 0.0099 0.0397 -0.0154 0.0424 -0.0150 -0.0016 0.0305 0.0101 0.2266 0.1394 0.0189 0.0069 0.0394 0.0355 -0.0111 -0.0687 -0.0078 0.0224 0.0817 -0.1949 0.0001 0.4047 -0.0237 -0.0656 -0.0684 0.0233 0.0438 0.1203 -0.0276 0.0416 0.0114 -0.4529 0.1538 0.1323 -0.0186 -0.0914 -0.0312 0.1051 0.0212 0.0798 -0.0104 -0.0206 -0.0025 0.0043 -0.0378 0.2689 0.0747 -0.0418 -0.0048 -0.0387 0.0432 0.1704 0.0614 0.0905 -0.0436 -0.0141 -0.0315 0.0276 0.0151 -0.0103 -0.0266 -0.0512 -0.0408 -0.0651 0.0662 -0.0936 0.1371 0.0458 -0.1366 -0.0075 -0.0104 -0.0732 0.1205 0.1035 0.0106 -0.0317 -0.0316 0.6639 -0.0022 -0.1343 0.0144 -0.0338 0.0034 -0.0429 -0.0821 0.0037 0.1029 -0.0204 -0.0269 0.0052 -0.1034 0.1068 0.0121 0.0980 -0.0458 0.0199 -0.0132 0.1936 -0.0213 0.0209 -0.0025 0.0416 -0.0337 0.0516 -0.1014 0.0203 0.0198 -0.0305 -0.0313 0.0543 -0.0106 0.1441 -0.0178 -0.0627 0.0475 0.0352 -0.0254 -0.0949 0.0401 0.0317 0.0055 -0.0536 0.0191 -0.0511 -0.0409 -0.0030 0.1582 0.0108 0.5237 0.0436 0.0306 -0.0392 0.0177 0.0069 0.0605 0.1206 -0.0216 -0.0633 -0.2965 0.0521 -0.0150 -0.2207 -0.0642 -0.0906 -0.0121 0.0569 0.0944 -0.0652 -0.0108 -0.0477 0.0023 0.0077 -0.1547 0.0463 0.0698 -0.0376 -0.0291 0.0033 -0.0102 -0.0743 0.0085 0.0805 -0.0291 -0.0674 -0.0586 -0.0653 0.0283 -0.0255 0.0869 -0.0868 0.0090 0.3245 -0.0573 -0.0289 0.0470 -0.0117 0.0174 0.0132 -0.0226 -0.0664 0.0188 0.0263 0.0111 -0.0049 -0.0656 0.0295 0.0435 0.0290 0.1163 0.0448 -0.1139 -0.0553 -0.0528 0.1745 -0.0146 -0.1308 -0.0607 -0.0134 0.0781 0.0378 0.0228 -0.0728 -0.0059 0.0158 -0.0141 -0.0002 0.0193 -0.0148 -0.0463 0.0444 0.3034 0.1020 -0.0871 0.0317 -0.0370 -0.0725 -0.0042
the 0.0231 0.0170 0.0157 -0.0773 0.1088 0.0031 -0.1487 -0.2672 -0.0357 -0.0487 0.0807 0.1532 -0.0739 -0.0291 -0.0445 -0.0014 0.1014 0.0186 -0.0253 0.0200 -0.0026 -0.0179 0.0005 0.0054 -0.0134 0.0233 -0.0755 -0.0156 0.0415 -0.4985 0.0410 -0.0616 0.0047 0.0325 -0.0162 -0.0172 0.0988 0.0766 -0.0796 -0.0345 0.0124 -0.1007 -0.0292 -0.0762 -0.1261 -0.0531 0.0424 0.0144 -0.0683 0.2859 0.0399 0.0201 0.3240 -0.0656 -0.0497 0.0090 0.0902 -0.0138 -0.0412 -0.0297 0.3139 -0.1428 0.0166 -0.0219 -0.0575 0.1359 -0.1655 0.0019 0.0323 -0.0013 -0.3033 -0.0091 0.1462 0.1860
...
```
In this example this file contains 2,000,000 tokens with 300-dimension vectors for each of them.


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

* Select 20 words from your corpus with frequency > 50 (it can be just random words or 
any words you come up with (eg `cake`, `dog`, `cat`, ...)
* For each of these words
  * Identify top-15 closest words
  * Cluster these words into three clusters using k-means++ implementation in `scikit-learn`


#### Python interface for fastText
It might be easier to use a python interface for fastText to complete this part of the assignment.
The [pyfasttext](https://github.com/vrasneur/pyfasttext) library provides a convenient 
interface for a trained model (`*.bin` file) that you can use to efficiently find the closest words and 
their embeddings (which in turn will be used as the input to k-means).

The file 
[hw1_fasttext_python.py](https://github.com/text-machine-lab/uml_nlp_class/tree/master/hw1/hw1_fasttext_python.py)
contains an example how to use this library. This script print loads a trained model
and prints the first 10 dimension of a word embedding for a target word.
After that, it finds top 15 closest to the target words and prints them.  

To run this script, execute the following command:
```bash
$ python3 hw1_fasttext_python.py
```
Note that you need to train a fastText model before running this script.



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
class which is necessary
in order to use the standard 
[`DataLoader`](http://pytorch.org/docs/0.3.0/data.html#torch.utils.data.DataLoader) 
class from PyTorch.

To simplify the code, the version of the dataset class, described below, is actually different
from the original version in the paper [\[2\]](https://arxiv.org/abs/1310.4546). 
This version randomly samples a word from the context window
whereas the original version provides a separate training example for every word in the context window.
It is possible, however, to implement the original version and 
you are encouraged to do so for an extra credit.

The `__len__` method should return 
the total length of the dataset. Note that in our case the length of the dataset
depends on the skip window (the `skip_window` parameter), since we can choose the context token on the left 
or on the right sides of the target token.
For example, consider the following dataset, consisted of the following 
tokens: `['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first', 'used', 'against']`.
With the skip window equals 2, the minimum index of the target word 
is 2 (the third token), since we need to be able to index 
the context words in the left. The same applies to the maximum index. 
Thus, this method should correctly calculate the length of the dataset
based on the skip window size.

The `__getitem__` method should accept an index of the target word 
and randomly choose a context word from the context on the left or on 
the right inside the context window.
For example above, if the `index` is 0 and the skip window is 2 
it would correspond to the target word `'as'` and this method would 
randomly a context word from the words 
`'anarchism'`, `'originated'`, `'a'`, or `'term'`.

Note that since neural netoworks require a numerical input, we associate each unique word (token) 
with a unique integer, and use these integers, not words, as the input into the network.
This conversion is done inside the `build_dataset` function. A better design solution would be to have the dataset class done this 
association, but for simplicity we separated it into a stand-alone function. 


##### `SkipGramModel` class 
This class is based on the 
[`torch.nn.Module`](http://pytorch.org/docs/0.3.0/nn.html#torch.nn.Module) 
class which is a base class for all neural network modules in PyTorch and 
user-defined models should subclass it. 

In general, you need to implement three method: `__init__`, `__forward__` and `find_closest`.

The `__init__` method should declare all layers or parameters of the network that
you are going to use. For example, in our case you will need two layers:
1. An embedding layer 
([torch.nn.Embedding](http://pytorch.org/docs/0.3.0/nn.html#torch.nn.Embedding)) 
that would convert words indices into dense vectors 
by just selecting appropritate vectors at the corresponding position.
2. A projection layer 
([torch.nn.Linear](http://pytorch.org/docs/0.3.0/nn.html#torch.nn.Linear))
that would transform the embedding of a word into a probability distribution 
over the context words. Note that the used loss function 
([torch.nn.CrossEntropyLoss](http://pytorch.org/docs/0.3.0/nn.html#torch.nn.CrossEntropyLoss))
accepts unnormalized probabilities so you **do not need** an activation function here.

You should declare the aforementioned layers in the `__init__` method.

The `forward` method performs the forward pass of the network on the input data. 
In our case that means using the embedding and the projection layers to get the output
distribution over the vocabulary for every input word in the batch.


The `find_closest` method finds the closest tokens (in terms of cosine similarity) to each token in the provided list.
To do this, you can either use the built-in
[F.cosine_similarity](http://pytorch.org/docs/0.3.0/nn.html#torch.nn.functional.cosine_similarity) function or 
just implement it manually using batch matrix multiplication (using the 
[torch.matmul](http://pytorch.org/docs/0.3.0/torch.html#torch.matmul) function). 


#### Running the code
First, build the image using the `docker_build.sh` script. 
Next, run the code with the `docker_run_part2.sh` script. 

If successful, you should see an output similar the following:
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
a jupyter notebook-like free to use environment. Moreover, it allows users to utilize GPU resources.
To enable GPU support open a notebook, click "Runtime" -> "Change runtime type", and select 
"GPU" under the hardware accelerator section. Click "Save" to save thee changes. 

## What to submit

1. For part 1, submit a printout of the words you selected, along with clusters of top-K neighbours
2. For part 2, submit
    1. Your code
    2. A README file explaining the overall structure of your implementation 
    3. A Dockerfile that specifies what else needs to be installed and how to run it. Please use the base Docker image we provide.


Homework assignments should be submitted using the submit utility available on the cs.uml.edu machines. Submit as follows:
```
$ submit arum assignment-name items-to-submit
```



## Reading materials

1. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). 
Efficient estimation of word representations in vector space. 
*arXiv preprint arXiv:1301.3781.* [\[arXiv\]](https://arxiv.org/abs/1301.3781)
1. Mikolov, T., Sutskever, I., Chen, K., Corrado, G. and Dean, J.
Distributed Representations of Words and Phrases and their Compositionality
*arXiv preprint arXiv:1310.4546.* [[\arXiv\]](https://arxiv.org/abs/1310.4546)
1. J. Eisenstein. NLP Notes, chapter 13 [\[pdf\]](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf)
1. Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2016). 
Enriching word vectors with subword information. 
*arXiv preprint arXiv:1607.04606.* [\[arXiv\]](https://arxiv.org/abs/1607.04606)
1. [Deep Learning with PyTorch: A 60 Minute Blitz](http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
1. [Get started with Docker](https://docs.docker.com/get-started/)
1. [PyTorch tutorials](http://pytorch.org/tutorials/)
1. [PyTorch documentation](http://pytorch.org/docs/0.3.0/)
1. [PyTorch examples](https://github.com/pytorch/examples)


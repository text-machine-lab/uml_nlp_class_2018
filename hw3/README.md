# Assignment #3

In this homework you'll train a neural machine translation system
to translate sentences from English to French.

This assignment is similar to the the second homework.
As in the HW#2 you will generate some text.
However, the difference is that here this text will be conditioned
on the source sentence. In other words, if in the previous homework
the model generated just some text (maybe, starting with one or two
specific words), in this assignment the output sentence should be
a translation of the input sentence.
To achieve this, we will use a sequence-to-sequence model
proposed by \[1].


![Overview](https://www.tensorflow.org/images/seq2seq/encdec.jpg)

An overview of a Neural Machine Translation system.
Image taken from \[2\].

A brief overview how the model works:

 1. An encoder (LSTM network) encodes the input sequence into a hidden vector **z**
 2. This vector **z** serves as the initial hidden state of the decoder
 (another LSTM network).
 3. The decoder decodes the output sequence starting with
 a special `<s>` (start-of-sentence) token
 until it produces a special `</s>` (end-of-sentence) token
 or the maximum length is reached.
 4. The whole system, consisting (in the simplest case)
 of an encoder, a decoder, and an output projection layer is trained end-to-end
 on samples of parallel sentences (i.e. a source sentence in english and
 the corresponding sentence in french).


![Detailed](https://www.tensorflow.org/images/seq2seq/seq2seq.jpg)

A more detailed structure of the model. Note two layers of RNNs.
In this homework, we will use just one layer. Image taken from \[2\].


# Implementation details

You will need to implement/modify six functions in the `Seq2SeqModel` class in the `model.py` file:
`__init__`, `zero_state`, `encode_sentence`,
`decoder_state`, `decoder_initial_inputs`, and `decode_sentence`.

The rest of the code is written for you.

## Input data
As the training data we will use the data from http://www.manythings.org/anki/.
It contains 135842 English sentences with the corresponding French translation.
For simplicity, we  filter out long sentences so the resulting training set
contains 119854 pairs and the validation set contains 5000 paris.

The dataset class is already implemented for you in the `dataset.py` file.

## Model
The model class, `Seq2SeqModel` is located in the file `model.py`.
Conceptually the methods of this class are divided into three groups:

 1. Initialization: `__init__`
 2. Encoder methods: `zero_state` and `encode_sentence`
 3. Decoder methods:  `decoder_state`, `decoder_initial_inputs`,
 and `decode_sentence`

In the `__init__` method you should declare all layers that you
are going to use: an embedding layer, an LSTM layer for the encoder,
an LSTM cell for the decoder, and an output projection layer.
Note that you should use the `torch.nn.LSTM` class for the encoder
and the `torch.nn.LSTMCell` class for the decoder.
The reason for this is that we will use special PyTorch functions
that correctly handle variable lengths sequences (since we have sentence of different length in a batch)
for the encoder, and just a for loop for the decoder.
The `torch.nn.LSTMCell` class is more suitable in the latter case.

The `zero_state` method returns initial hidden state of the encoder. Namely, it should return a tuple
of tensors for the hidden state `h` and the cell state `c` correspondingly
(see LSTM lecture for details on those states). This method is meant to be called from the `encode_sentence` method.

The `encode_sentence` method encodes sentences in a batch into hidden vectors `z`, which is the last hidden state
of the LSTM network. Note that since we have sequences of different lengths in a single batch, we cannot
just take the last hidden state as follows: `z = h[:,-1,:]`. Instead, you will need to use the
`torch.nn.utils.rnn.pack_padded_sequence` function. You might find the function `get_sequences_lengths` in `utils.py`
useful for this task.

The `decoder_state` method creates initial hidden state of the decoder. Since we want to decode a translation of
the input sequence, it takes the hidden vectors `z` as the argument and returns a tuple of `(z,c)`,
where `c` is a vector of zeros (as in the `zero_state`).
This method is meant to be called from the `decode_sentence` method.

The `decoder_initial_inputs` method should just return a batch of indices of the `<s>` token to be served as input
to the decoder at the first timestep. This method is meant to be called from the `decode_sentence` method.

Finally, the `decode_sentence` method decodes the outputs translation using a for loop. Its implementation is
very similar to that of the second homework.

## Running the code
To run the code, use the following command:
`python3 train.py`

You out should be similar to this:
```
Vocab pruned: 53985 -> 33142
Dataset: 124854
Train 119854, val: 5000
Epoch 000 | train loss: 5.085, val loss: 4.570
> it's very kind of you.
= c'est très gentil de votre part.
< c'est très très

...

Epoch 011 | train loss: 1.850, val loss: 2.672
> he danced all night long.
= il a dansé toute la nuit.
< il a toute toute la nuit.
```

`>` indicates the input sentence, `=` is the true translation, and `<` is the output of the model.

### Bonus points
Implement neural machine translation with attention.

# What to submit
Submit all files in the `hw3/` in this repository with your code written in the marked places.
In addition, submit the output of the script in a file called `output.txt`.

Please make sure that you completed the code in the file `model.py`
You code should be able to run in docker using the provided `docker_build.sh` and `docker_run.sh` scripts.


# References
\[1\]. Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. "Sequence to sequence learning with neural networks." Advances in neural information processing systems. 2014.
[\[link\]](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural)

\[2\]. [Neural Machine Translation (seq2seq) Tutorial](https://www.tensorflow.org/tutorials/seq2seq)

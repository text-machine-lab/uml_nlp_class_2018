import re
import html
import itertools
from collections import Counter

import numpy as np
import torch
import torch.utils.data

from nltk import word_tokenize, sent_tokenize
from nltk.tokenize import TweetTokenizer

import logging
from collections import Counter


class Vocab(object):
    def __init__(self, special_tokens=None):
        super(Vocab, self).__init__()

        self.nb_tokens = 0

        self.token2id = {}
        self.id2token = {}

        self.token_counts = Counter()

        self.special_tokens = []
        if special_tokens is not None:
            self.special_tokens = special_tokens
            self.add_document(self.special_tokens)

    def add_document(self, document):
        for token in document:
            self.token_counts[token] += 1

            if token not in self.token2id:
                self.token2id[token] = self.nb_tokens
                self.id2token[self.nb_tokens] = token

                self.nb_tokens += 1

    def add_documents(self, documents):
        for doc in documents:
            self.add_document(doc)

    def prune_vocab(self, min_count=2):
        nb_tokens_before = len(self.token2id)

        tokens_to_delete = set([t for t, c in self.token_counts.items() if c < min_count])
        tokens_to_delete ^= set(self.special_tokens)

        for token in tokens_to_delete:
            self.token_counts.pop(token)

        self.token2id = {t: i for i, t in enumerate(self.token_counts.keys())}
        self.id2token = {i: t for t, i in self.token2id.items()}
        self.nb_tokens = len(self.token2id)

        print(f'Vocab pruned: {nb_tokens_before} -> {self.nb_tokens}')

    def __getitem__(self, item):
        return self.token2id[item]

    def __contains__(self, item):
        return item in self.token2id

    def __len__(self):
        return self.nb_tokens

    def __str__(self):
        return f'Vocab: {self.nb_tokens} tokens'


class LanguageModelDataset(torch.utils.data.Dataset):
    PAD_TOKEN = '<pad>'
    INIT_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'
    UNK_TOKEN = '<unk>'

    def __init__(self, sentences, max_len=20, vocab=None):
        super().__init__()

        self.max_len = max_len

        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = Vocab([
                LanguageModelDataset.PAD_TOKEN, LanguageModelDataset.UNK_TOKEN,
                LanguageModelDataset.INIT_TOKEN, LanguageModelDataset.EOS_TOKEN,
            ])

        self.vocab.add_documents(sentences)

        # cut to max len and append the end-of-sentence tokens
        sentences = [s[:max_len - 1] for s in sentences]
        sentences = [s + [LanguageModelDataset.EOS_TOKEN, ] for s in sentences]

        self.sentences = sentences
        self.nb_sentences = len(sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]

        # pad to max_len
        nb_pads = self.max_len - len(sentence)
        if nb_pads > 0:
            sentence = sentence + [LanguageModelDataset.PAD_TOKEN] * nb_pads

        # convert to indices
        sentence = [
            self.vocab.token2id[t] if t in self.vocab.token2id else self.vocab.token2id[LanguageModelDataset.UNK_TOKEN]
            for t in sentence
        ]
        sentence = np.array(sentence)

        return sentence

    def __len__(self):
        return self.nb_sentences


class TwitterFileArchiveDataset(LanguageModelDataset):
    def __init__(self, filename, *args, **kwargs):
        sentences = self._load_file(filename)
        super().__init__(sentences, *args, **kwargs)

    def _load_file(self, filename):
        """
        Load a file with tweet, one tweet per line
        :param filename: The path to the file
        :return: A list of lists of tokens, e.g. [ [I, am, great, ...], [It, is, going, to, ...], ...]
        """
        ##############################
        ### Insert your code below ###
        # open the file, read it line by line, and tokenize using the TweetTokenizer class
        # tweets should be a list of tweets' tokens (i.e. a list of lists of tokens - see the README)
        ##############################
        raise NotImplementedError()

        ###############################
        ### Insert your code above ####
        ###############################
        return tweets

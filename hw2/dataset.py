import re
import html
import itertools
from collections import Counter

import numpy as np
import torch
import torch.utils.data

from nltk import word_tokenize, sent_tokenize
from nltk.tokenize import TweetTokenizer


class LanguageModelDataset(torch.utils.data.Dataset):
    PAD_TOKEN = '<pad>'
    INIT_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'
    UNK_TOKEN = '<unk>'

    def __init__(self, sentences, max_len=20):
        super().__init__()

        self.max_len = max_len

        self.token2id = {}
        self.id2token = {}
        self.token_counts = Counter()

        self.special_tokens = [
            LanguageModelDataset.PAD_TOKEN, LanguageModelDataset.UNK_TOKEN,
            LanguageModelDataset.INIT_TOKEN, LanguageModelDataset.EOS_TOKEN,
        ]

        self._build_vocab([self.special_tokens, ])
        self._build_vocab(sentences)
        self._prune_vocab(min_count=2)

        # cut to max len and append the end-of-sentence tokens
        sentences = [s[:max_len - 1] for s in sentences]
        sentences = [s + [LanguageModelDataset.EOS_TOKEN, ] for s in sentences]

        self.sentences = sentences
        self.nb_sentences = len(sentences)

    def _build_vocab(self, data):
        for token in itertools.chain.from_iterable(data):
            self.token_counts[token] += 1

            if token not in self.token2id:
                self.token2id[token] = len(self.token2id)

                self.id2token = {i: t for t, i in self.token2id.items()}

    def _prune_vocab(self, min_count=2):
        nb_tokens_before = len(self.token2id)

        tokens_to_delete = set([t for t, c in self.token_counts.items() if c < min_count])
        tokens_to_delete ^= set(self.special_tokens)

        for token in tokens_to_delete:
            self.token_counts.pop(token)

        self.token2id = {t: i for i, t in enumerate(self.token_counts.keys())}
        self.id2token = {i: t for t, i in self.token2id.items()}

        print('Vocab pruned: {} -> {}'.format(nb_tokens_before, len(self.token2id)))

    def __getitem__(self, index):
        sentence = self.sentences[index]

        # pad to max_len
        nb_pads = self.max_len - len(sentence)
        if nb_pads > 0:
            sentence = sentence + [LanguageModelDataset.PAD_TOKEN] * nb_pads

        # convert to indices
        sentence = [
            self.token2id[t] if t in self.token2id else self.token2id[LanguageModelDataset.UNK_TOKEN]
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
        raise NotImplementedError('Implement the `_load_file` method')

        ###############################
        ### Insert your code above ####
        ###############################
        return tweets

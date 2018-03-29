import csv
from collections import Counter

import numpy as np
import torch.utils.data


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

        print('Vocab pruned: {} -> {}'.format(nb_tokens_before, self.nb_tokens))

    def __getitem__(self, item):
        return self.token2id[item]

    def __contains__(self, item):
        return item in self.token2id

    def __len__(self):
        return self.nb_tokens

    def __str__(self):
        return 'Vocab: {} tokens'.format(self.nb_tokens)


class SentencePairDataset(torch.utils.data.Dataset):
    PAD_TOKEN = '<pad>'
    EOS_TOKEN = '</s>'
    INIT_TOKEN = '<s>'
    UNK_TOKEN = '<unk>'

    def __init__(self, sentence1, sentence2, max_len=64, min_count=300):
        self.max_len = max_len
        self.min_count = min_count

        sentence1 = [s.lower().split() for s in sentence1]
        sentence2 = [s.lower().split() for s in sentence2]

        self.sentence1 = sentence1
        self.sentence2 = sentence2

        assert len(self.sentence1) == len(self.sentence2)

        self.vocab = Vocab(special_tokens=[SentencePairDataset.PAD_TOKEN, SentencePairDataset.EOS_TOKEN,
                                           SentencePairDataset.UNK_TOKEN, SentencePairDataset.INIT_TOKEN])
        self.vocab.add_documents(self.sentence1)
        self.vocab.add_documents(self.sentence2)
        self.vocab.prune_vocab(min_count=min_count)


    def _process_sentence(self, sentence):
        sentence = sentence[:self.max_len - 1]
        sentence.append(SentencePairDataset.EOS_TOKEN)

        needed_pads = self.max_len - len(sentence)
        if needed_pads > 0:
            sentence = sentence + [SentencePairDataset.PAD_TOKEN] * needed_pads

        sentence = [
            self.vocab[token] if token in self.vocab else self.vocab[SentencePairDataset.UNK_TOKEN]
            for token in sentence
        ]

        sentence = np.array(sentence, dtype=np.long)

        return sentence

    def __getitem__(self, index):
        sentence1 = self._process_sentence(self.sentence1[index])
        sentence2 = self._process_sentence(self.sentence2[index])

        return sentence1, sentence2

    def __len__(self):
        return len(self.sentence1)

class TSVSentencePairDataset(SentencePairDataset):
    def __init__(self, filename, *args, **kwargs):
        sentence1, sentence2 = self._read_tsv(filename)
        super().__init__(sentence1, sentence2, *args, **kwargs)

    def _read_tsv(self, filename):
        with open(filename, 'r', encoding='utf8') as f:
            reader = csv.reader(f, delimiter='\t')

            sentence1 = []
            sentence2 = []
            for sent1, sent2 in reader:
                if sent1.count(' ') < 10 and sent2.count(' ') <= 10:
                    sentence1.append(sent1)
                    sentence2.append(sent2)

        return sentence1, sentence2
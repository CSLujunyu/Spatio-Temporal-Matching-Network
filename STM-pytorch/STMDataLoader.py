# -*- coding: utf-8 -*-

import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset
from nltk.tokenize import TreebankWordTokenizer

def build_corpus_tokenizer(training_examples, vocab_size):
    """
    Use training set to build tokenizer
    :param training_examples: 
    :return: 
    """
    assert training_examples[0].guid.startswith('train')

    all_seq = []
    for (ex_index, example) in enumerate(training_examples):
        all_seq += example.text_a + example.text_b

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token='<UNK>', split=' ', lower=True)
    tokenizer.fit_on_texts(all_seq)

    return tokenizer


def build_corpus_dataloader(examples, max_turn_length, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    print("#examples", len(examples))

    contexts = []
    candidates = []
    labels = []
    for (ex_index, example) in enumerate(examples):

        tokens_a = pad_sequences(tokenizer.texts_to_sequences(example.text_a), maxlen=max_seq_length, padding='post', truncating='pre')

        tokens_b = pad_sequences(tokenizer.texts_to_sequences(example.text_b), maxlen=max_seq_length, padding='post', truncating='pre')

        # Zero-pad up to the sequence length.
        if len(tokens_a) < max_turn_length:
            tokens_a = np.concatenate([tokens_a,np.zeros([max_turn_length - len(tokens_a), max_seq_length])])
        else:
            tokens_a = tokens_a[-max_turn_length:]

        assert len(tokens_a) == max_turn_length

        contexts.append(tokens_a)
        candidates.append(tokens_b)
        labels.append(example.label)

    all_contexts = torch.LongTensor(contexts)
    all_candidates = torch.LongTensor(candidates)
    all_labels = torch.LongTensor(labels)

    tensor_dataset = TensorDataset(all_contexts, all_candidates, all_labels)

    return tensor_dataset

def build_corpus_embedding(vocab_size, emb_dim, pretrain_dir, tokenizer):

    ### Get vocabulary
    vocab = ['<PAD>'] + list(tokenizer.word_index.keys())[:vocab_size-1]
    vocab_dict = {}
    for i, w in enumerate(vocab):
        vocab_dict[w] = i
    ### Initialize word embedding
    word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(vocab), emb_dim))

    emb_count = 0
    with open(pretrain_dir) as f:
        for line in f.readlines():
            row = line.split(' ')
            assert len(row[1:]) == emb_dim
            try:
                word_embeds[vocab_dict[row[0]]] = [float(v) for v in row[1:]]
                emb_count += 1
            except:
                pass
    word_embeds[0] = [0.0]*emb_dim
    word_embeds = torch.FloatTensor(word_embeds)
    print('Loaded %i pretrained embeddings.' % emb_count)

    return word_embeds, vocab
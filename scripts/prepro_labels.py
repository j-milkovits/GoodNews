"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the 
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image, 
  such as in particular the 'split' it was assigned to.
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
from random import seed

import h5py
import numpy as np
import skimage.io
import torch
import torchvision.models as models
from torch.autograd import Variable


def build_vocab(imgs, params):
    count_thr = params["word_count_threshold"]

    # Count up the number of words
    counts = {}
    for img in imgs:
        for sent in img["sentences"]:
            for w in sent["tokens"]:
                counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    print("Top words and their counts:")
    print("\n".join(map(str, cw[:20])))

    # Print some stats
    total_words = sum(counts.values())
    print("Total words:", total_words)
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print(
        "Number of bad words: %d/%d = %.2f%%"
        % (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts))
    )
    print("Number of words in vocab would be %d" % (len(vocab),))
    print(
        "Number of UNKs: %d/%d = %.2f%%"
        % (bad_count, total_words, bad_count * 100.0 / total_words)
    )

    # Let's look at the distribution of lengths as well
    sent_lengths = {}
    for img in imgs:
        for sent in img["sentences"]:
            txt = sent["tokens"]
            nw = len(txt)
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print("Max length sentence in raw data: ", max_len)
    print("Sentence length distribution (count, number of words):")
    sum_len = sum(sent_lengths.values())
    for i in range(max_len + 1):
        print(
            "%2d: %10d   %f%%"
            % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0) * 100.0 / sum_len)
        )

    # Let's now produce the final annotations
    if bad_count > 0:
        # Additional special UNK token we will use below to map infrequent words to
        print("Inserting the special UNK token")
        vocab.append("UNK")

    for img in imgs:
        img["final_captions"] = []
        for sent in img["sentences"]:
            txt = sent["tokens"]
            caption = [w if counts.get(w, 0) > count_thr else "UNK" for w in txt]
            img["final_captions"].append(caption)

    return vocab


def encode_captions(imgs, params, wtoi):
    """
    Encode all captions into one large array, which will be 1-indexed.
    Also produces label_start_ix and label_end_ix which store 1-indexed
    and inclusive (Lua-style) pointers to the first and last caption for
    each image in the dataset.
    """

    max_length = params["max_length"]
    N = len(imgs)
    M = sum(len(img["final_captions"]) for img in imgs)  # Total number of captions

    label_arrays = []
    label_start_ix = np.zeros(N, dtype="uint32")  # Note: these will be one-indexed
    label_end_ix = np.zeros(N, dtype="uint32")
    label_length = np.zeros(M, dtype="uint32")
    caption_counter = 0
    counter = 1
    for i, img in enumerate(imgs):
        n = len(img["final_captions"])
        assert n > 0, "Error: some image has no captions"

        Li = np.zeros((n, max_length), dtype="uint32")
        for j, s in enumerate(img["final_captions"]):
            label_length[caption_counter] = min(
                max_length, len(s)
            )  # Record the length of this sequence
            caption_counter += 1
            for k, w in enumerate(s):
                if k < max_length:
                    Li[j, k] = wtoi[w]

        label_arrays.append(Li)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1

        counter += n

    L = np.concatenate(label_arrays, axis=0)  # Put all the labels together
    assert L.shape[0] == M, "Lengths don't match? That's weird"
    assert np.all(label_length > 0), "Error: some caption had no words?"

    print("Encoded captions to array of size ", L.shape)
    return L, label_start_ix, label_end_ix, label_length


def main(params):
    with open(params["input_json"], "r") as f:
        imgs = json.load(f)

    seed(123)  # Make reproducible

    # Create the vocab
    vocab = build_vocab(imgs, params)
    itow = {
        i + 1: w for i, w in enumerate(vocab)
    }  # A 1-indexed vocab translation table
    wtoi = {w: i + 1 for i, w in enumerate(vocab)}  # Inverse table

    #


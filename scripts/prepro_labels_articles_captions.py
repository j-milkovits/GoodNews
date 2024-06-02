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
import string
from random import seed

import h5py
import numpy as np
import skimage.io
import spacy
import torch
import torchvision.models as models
import tqdm
from torch.autograd import Variable


def build_vocab(imgs, params):
    count_thr = params["word_count_threshold"]
    counts = {}
    for img in imgs:
        for sent in img["sentences"]:
            for w in sent["tokens"]:
                counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    print("Top words and their counts:")
    print("\n".join(map(str, cw[:20])))

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

    if bad_count > 0:
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
    max_length = params["max_length"]
    N = len(imgs)
    M = sum(len(img["final_captions"]) for img in imgs)

    label_arrays = []
    label_start_ix = np.zeros(N, dtype="uint32")
    label_end_ix = np.zeros(N, dtype="uint32")
    label_length = np.zeros(M, dtype="uint32")
    caption_counter = 0
    counter = 1
    for i, img in enumerate(imgs):
        n = len(img["final_captions"])
        assert n > 0, "Error: some image has no captions"

        Li = np.zeros((n, max_length), dtype="uint32")
        for j, s in enumerate(img["final_captions"]):
            label_length[caption_counter] = min(max_length, len(s))
            caption_counter += 1
            for k, w in enumerate(s):
                if k < max_length:
                    Li[j, k] = wtoi[w]

        label_arrays.append(Li)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1
        counter += n

    L = np.concatenate(label_arrays, axis=0)
    assert L.shape[0] == M, "Lengths don't match? That's weird"
    assert np.all(label_length > 0), "Error: some caption had no words?"

    print("Encoded captions to array of size ", L.shape)
    return L, label_start_ix, label_end_ix, label_length


def open_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_word_vector(sen):
    sen = nlp(sen)
    return sen.vector


def save_h5(file_save, data):
    f_lb = h5py.File(file_save, "w")
    dt = h5py.special_dtype(vlen=np.dtype("float64"))
    ds = f_lb.create_dataset("average", (len(data), 300), dtype=dt)
    for i, d in tqdm.tqdm(enumerate(data)):
        ds[i] = d
    f_lb.close()


def create_rep(art):
    data = []
    keys = []
    for k, v in tqdm.tqdm(art.items()):
        v = v["sentence"]
        if len(v) < sen_len + 1:
            temp = np.zeros([300, len(v)])
            for i, sents in enumerate(v):
                temp[:, i] = get_word_vector(sents.lower())
        else:
            temp = np.zeros([300, sen_len + 1])
            for i, sents in enumerate(v[:sen_len]):
                temp[:, i] = get_word_vector(sents.lower())
            temp[:, sen_len] = np.average(
                [get_word_vector(sents.lower()) for sents in v[sen_len:]]
            )
        data.append(temp)
        keys.append(k)
    return keys, data


def main(params):
    imgs = open_json(params["input_json"])
    seed(123)
    vocab = build_vocab(imgs, params)
    itow = {i + 1: w for i, w in enumerate(vocab)}
    wtoi = {w: i + 1 for i, w in enumerate(vocab)}

    L, label_start_ix, label_end_ix, label_length = encode_captions(imgs, params, wtoi)

    N = len(imgs)
    f_lb = h5py.File(params["output_h5"] + "_label.h5", "w")
    f_lb.create_dataset("labels", dtype="uint32", data=L)
    f_lb.create_dataset("label_start_ix", dtype="uint32", data=label_start_ix)
    f_lb.create_dataset("label_end_ix", dtype="uint32", data=label_end_ix)
    f_lb.create_dataset("label_length", dtype="uint32", data=label_length)
    f_lb.close()

    out = {}
    out["ix_to_word"] = itow
    out["images"] = []
    for i, img in enumerate(imgs):
        jimg = {}
        jimg["split"] = img["split"]
        if "filename" in img:
            jimg["file_path"] = os.path.join(img["filepath"], img["filename"])
        if "cocoid" in img:
            jimg["id"] = img["cocoid"]
        out["images"].append(jimg)

    json.dump(out, open(params["output_json"], "w"))
    print("Wrote ", params["output_json"])

    articles = open_json(params["input_article_json"])
    global sen_len
    sen_len = params["sen_len"]
    keys, data = create_rep(articles)
    json.dump(keys, open(params["output_keys_json"], "w"))
    save_h5(params["output_avg_h5"], data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_json",
        required=True,
        help="Input JSON file with image paths and captions",
    )
    parser.add_argument("--output_json", required=True, help="Output JSON file")
    parser.add_argument("--output_h5", required=True, help="Output HDF5 file")
    parser.add_argument(
        "--max_length", default=31, type=int, help="Max length of a caption"
    )
    parser.add


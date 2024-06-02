import json
import math
from collections import Counter

import h5py
import numpy as np
import spacy
import tqdm
from nltk.tokenize import word_tokenize


def open_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_word_vector(sen, nlp):
    sen = nlp(sen)
    return sen.vector


def save_h5(file_save, data):
    with h5py.File(file_save, "w") as f_lb:
        dt = h5py.special_dtype(vlen=np.dtype("float64"))
        ds = f_lb.create_dataset(
            "average",
            (
                len(data),
                300,
            ),
            dtype=dt,
        )
        for i, d in tqdm.tqdm(enumerate(data)):
            ds[i] = d


def getLog(frequency, a=10**-3):
    return a / (a + math.log(1 + frequency))


def get_vector_avg_weighted_full(sent, nlp, count_full):
    sent = nlp(sent)
    vectors = []
    weights = []
    for token in sent:
        if token.has_vector:
            frequency = count_full.get(token.text.lower(), 10)
            weight = getLog(frequency)
            vectors.append(token.vector)
            weights.append(weight)
    try:
        doc_vector = np.average(vectors, weights=weights, axis=0)
    except:
        doc_vector = sent.vector
    return doc_vector


def get_weighted_avg_data(article, get_vector_avg_weighted, nlp, count_full, sen_len):
    data, keys = [], []
    for k, v in tqdm.tqdm(article.items()):
        if len(v) < sen_len + 1:
            temp = np.zeros([300, len(v)])
            for i, sents in enumerate(v):
                temp[:, i] = get_vector_avg_weighted(sents.lower(), nlp, count_full)
        else:
            temp = np.zeros([300, sen_len + 1])
            for i, sents in enumerate(v[:sen_len]):
                temp[:, i] = get_vector_avg_weighted(sents.lower(), nlp, count_full)
            temp[:, sen_len] = np.average(
                [
                    get_vector_avg_weighted(sents.lower(), nlp, count_full)
                    for sents in v[sen_len:]
                ]
            )
        data.append(temp)
        keys.append(k)

    return keys, data


if __name__ == "__main__":
    sen_len = 54
    np.random.seed(42)
    nlp = spacy.load("en_core_web_lg", disable=["parser", "tagger"])

    count_full = Counter()

    full = open_json("../data/article.json")
    for v_fu in full.values():
        for elm in v_fu["sentence"]:
            count_full.update([t.lower() for t in word_tokenize(elm)])

    keys, data = get_weighted_avg_data(
        full, get_vector_avg_weighted_full, nlp, count_full, sen_len
    )
    with open("../data/articles_full_WeightedAvg_keys.json", "w") as keys_file:
        json.dump(keys, keys_file)
    save_h5("../data/articles_full_WeightedAvg.h5", data)


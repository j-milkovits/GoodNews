import argparse
import json

import h5py
import numpy as np
import spacy
import tqdm


def open_json(path):
    """Open and load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def get_word_vector(sen):
    """Convert a sentence to its word vector representation using SpaCy."""
    sen = nlp(sen)
    return sen.vector


def save_h5(file_save, data):
    """Save data to an HDF5 file."""
    with h5py.File(file_save, "w") as f_lb:
        dt = h5py.special_dtype(vlen=np.dtype("float64"))
        ds = f_lb.create_dataset("average", (len(data), 300), dtype=dt)
        for i, d in tqdm.tqdm(enumerate(data)):
            ds[i] = d


def create_rep(art):
    """Create representation for each article."""
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process articles and convert sentences to word vectors."
    )
    parser.add_argument(
        "--input_json",
        required=True,
        help="Path to the input JSON file containing articles.",
    )
    parser.add_argument(
        "--output_keys_json",
        required=True,
        help="Path to save the output JSON file with article keys.",
    )
    parser.add_argument(
        "--output_h5",
        required=True,
        help="Path to save the output HDF5 file with word vectors.",
    )
    parser.add_argument(
        "--sen_len",
        type=int,
        default=54,
        help="Maximum number of sentences to consider for each article.",
    )

    args = parser.parse_args()

    sen_len = args.sen_len
    np.random.seed(42)
    nlp = spacy.load("en_core_web_lg", disable=["parser", "tagger"])

    full = open_json(args.input_json)
    keys, data = create_rep(full)
    json.dump(keys, open(args.output_keys_json, "w"))
    save_h5(args.output_h5, data)


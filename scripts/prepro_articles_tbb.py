import json

import h5py
import numpy as np
import tqdm
from sklearn.decomposition import TruncatedSVD


def open_json(path):
    """Open and load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_h5(file_save, data):
    """Save data to an HDF5 file."""
    with h5py.File(file_save, "w") as f_lb:
        dt = h5py.special_dtype(vlen=np.dtype("float64"))
        ds = f_lb.create_dataset("average", (len(data), 300), dtype=dt)
        for i, d in tqdm.tqdm(enumerate(data)):
            ds[i] = d


if __name__ == "__main__":
    np.random.seed(42)
    # Load the data from the HDF5 file
    data_com = h5py.File("../data/articles_full_WeightedAvg.h5")
    data_com = [np.stack(d) for d in tqdm.tqdm(data_com.get("average"))]

    # Determine the lengths of each article
    lengths = [d.shape[1] for d in data_com]

    # Reshape the data for dimensionality reduction
    data_com = [d.reshape(-1, 300) for d in data_com]
    data_com = [i for d in data_com for i in d]

    # Perform TruncatedSVD for dimensionality reduction
    svd = TruncatedSVD(n_components=10, n_iter=7, random_state=42)
    svd.fit(data_com)
    pc = svd.components_

    # Subtract the projection of the data onto principal components
    data_com = data_com - np.dot(data_com, np.transpose(pc).dot(pc))

    # Reshape the data back to original shapes
    new = []
    index = 0
    for l in lengths:
        new.append(data_com[index : index + l].reshape(300, -1))
        index += l

    # Save the processed data and keys
    keys = open_json("../data/articles_full_WeightedAvg_keys.json")
    json.dump(keys, open("../data/articles_full_TBB_keys.json", "w"))
    save_h5("../data/articles_full_TBB.h5", new)


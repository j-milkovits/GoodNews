import json
import os

import tqdm
from PIL import Image


def open_json(path):
    with open(path, "r") as f:
        return json.load(f)


whole_dataset = {}
years = range(2010, 2019)
months = range(1, 13)
json_files = "./data/nytimes_%d_%d.json"
img_file = "../../images/"

for y in years:
    for i in tqdm.tqdm(months):
        try:
            temp = open_json(json_files % (y, i))
            for key, value in temp.items():
                if value.get("article", 0):
                    whole_dataset[key] = {
                        k: v for k, v in value.items() if not isinstance(v, list)
                    }
                    if value.get("images", 0):
                        whole_dataset[key].update({"images": value["images"]})
        except Exception as e:
            print(e)
            print(i)

# There are some corrupt images.
corrupt = []
for i in os.listdir(img_file):
    try:
        im = Image.open(os.path.join(img_file, i))
        im.verify()
    except Exception as e:
        print(e)
        corrupt.append(i)

# Delete corrupt images and their corresponding captions from the dataset
for c in corrupt:
    ix = c.split("_")[0]
    id_img = int(c.split("_")[1].split(".")[0])

    if ix in whole_dataset:
        if whole_dataset[ix]["images"].get(id_img, 0):
            print(ix, id_img)
            del whole_dataset[ix]["images"][id_img]

for c in corrupt:
    os.remove(os.path.join(img_file, c))

# Save the whole dataset
with open("../../data/dataset.json", "w") as f:
    json.dump(whole_dataset, f)

# Create a new dataset that has at least one image-caption pair
captioning_dataset = {
    key: value for key, value in whole_dataset.items() if value.get("images", 0)
}

with open("../../data/captioning_dataset.json", "w") as f:
    json.dump(captioning_dataset, f)


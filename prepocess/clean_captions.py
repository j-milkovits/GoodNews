import json
import re
import unicodedata
from itertools import groupby

import inflect
import nltk
import numpy as np
import spacy
import tqdm
import unidecode
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("stopwords")


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = (
            unicodedata.normalize("NFKD", word)
            .encode("ascii", "ignore")
            .decode("utf-8", "ignore")
        )
        new_words.append(new_word)
    return new_words


def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r"[^\w\s]", "", word)
        if new_word != "":
            new_words.append(new_word)
    return new_words


def replace_numbers(words):
    """Replace all integer occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    stop_words = set(stopwords.words("english"))
    new_words = [word for word in words if word.lower() not in stop_words]
    return new_words


def normalize(words):
    words = remove_non_ascii(words)
    words = remove_punctuation(words)
    return words


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def remove_between_square_brackets(text):
    return re.sub(r"\[[^]]*\]", "", text)


def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text


def preprocess_sentence(sen):
    sen = sen.strip()
    sen = unidecode.unidecode(sen)
    sen = denoise_text(sen)
    sen = nltk.tokenize.word_tokenize(sen)
    sen = normalize([str(s) for s in sen])
    return sen


def NER(sen):
    doc = nlp(sen)
    tokens = [d.text for d in doc]
    temp = [d.ent_type_ + "_" if d.ent_iob_ != "O" else d.text for d in doc]
    return [x[0] for x in groupby(temp)], tokens


def get_split():
    rand = np.random.uniform()
    if rand > 0.95:
        split = "test"
    elif 0.91 < rand <= 0.95:
        split = "val"
    else:
        split = "train"
    return split


if __name__ == "__main__":
    np.random.seed(42)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])
    print("Loading spacy modules.")
    news_data = []
    counter = 0

    print("Loading the json.")
    with open("../data/captioning_dataset.json", "r") as f:
        captioning_dataset = json.load(f)

    for k, anns in tqdm.tqdm(captioning_dataset.items()):
        for ix, img in anns["images"].items():
            try:
                split = get_split()
                img = preprocess_sentence(img)
                template, full = NER(" ".join(img))
                if len(" ".join(template)) != 0:
                    news_data.append(
                        {
                            "filename": k + "_" + str(ix) + ".jpg",
                            "filepath": "resized",
                            "cocoid": counter,
                            "imgid": k + "_" + str(ix),
                            "sentences": [],
                            "sentences_full": [],
                            "split": split,
                        }
                    )
                    news_data[counter]["sentences"].append(
                        {
                            "imgid": counter,
                            "raw": " ".join(template),
                            "tokens": template,
                        }
                    )
                    news_data[counter]["sentences_full"].append(
                        {"imgid": counter, "raw": " ".join(full), "tokens": full}
                    )
                    counter += 1
            except Exception as e:
                print(img, e)

    split_to_ix = {i: n["split"] for i, n in enumerate(news_data)}
    val = [news_data[k] for k, v in split_to_ix.items() if v == "val"]
    test = [news_data[k] for k, v in split_to_ix.items() if v == "test"]

    with open("../data/test.json", "w") as f:
        json.dump(test, f)
    with open("../data/val.json", "w") as f:
        json.dump(val, f)
    with open("../data/news_dataset.json", "w") as f:
        json.dump(news_data, f)


import json
import sys

import numpy as np
import spacy
from clean_captions import preprocess_sentence

np.random.seed(42)

# Load SpaCy model
nlp = spacy.load("en", disable=["parser", "tagger"])
nlp.add_pipe("sentencizer")

# Load datasets
with open("../data/news_dataset.json", "r", encoding="utf-8") as f:
    news_dataset = json.load(f)

with open("../data/captioning_dataset.json", "r", encoding="utf-8") as f:
    captioning_dataset = json.load(f)

# Extract unique article IDs
ids = list(set([news["imgid"].split("_")[0] for news in news_dataset]))
articles = [captioning_dataset[i]["article"] for i in ids]
len_articles = len(articles)

print(
    "Sanity check len of ids: %d, len of cap dataset: %d, len of news dataset %d"
    % (len(ids), len(captioning_dataset), len(news_dataset))
)

# Free up memory
del news_dataset, captioning_dataset

article_dataset = {}
for ix, sentences in enumerate(nlp.pipe(articles, n_process=12, batch_size=2000)):
    key = ids[ix]
    art_ner = {ent.text: ent.label_ for ent in sentences.ents}
    article_sentence = []
    article_sentence_ner = []

    for sen_ in sentences.sents:
        try:
            sen = preprocess_sentence(sen_.text.encode("ascii", errors="ignore"))
            sen_text = " ".join(sen)

            # Append sentence if it contains named entities
            if any(d.ent_iob_ != "O" for d in sen_):
                article_sentence_ner.append(sen_text)

            article_sentence.append(sen_text)
        except Exception as e:
            print(f"Error processing sentence for key {key}: {e}")

    article_dataset[key] = {
        "sentence": article_sentence,
        "sentence_ner": article_sentence_ner,
        "ner": art_ner,
    }

    sys.stdout.write("\rPercentage done: {:.4f}".format(ix / float(len_articles)))
    sys.stdout.flush()

# Save processed article dataset
with open("../data/article.json", "w", encoding="utf-8") as f:
    json.dump(article_dataset, f, ensure_ascii=False, indent=4)
    print("Finished!")

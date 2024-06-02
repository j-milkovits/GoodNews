import json
import os
import posixpath
from multiprocessing.dummy import Pool as ThreadPool
from urllib.parse import urlparse, urlunparse

import requests
import unidecode
from bs4 import BeautifulSoup
from goose3 import Goose
from six.moves.html_parser import HTMLParser


def resolveComponents(url):
    parsed = urlparse(url)
    new_path = posixpath.normpath(parsed.path)
    if parsed.path.endswith("/"):
        new_path += "/"
    cleaned = parsed._replace(path=new_path)
    return urlunparse(cleaned)


def get_soup(url):
    req = requests.get(url, headers=hdr)
    soup = BeautifulSoup(req.content, "html.parser")
    [s.extract() for s in soup("script")]
    [s.extract() for s in soup("noscript")]
    figcap = soup.find_all("figcaption")
    return soup, figcap


def retrieve_articles(m):
    articles = json.load(open(root + "api/nyarticles_%s_%s.json" % (year, m), "r"))
    month_data = {}
    leftovers = {}
    len_articles = len(articles["response"]["docs"])
    for num, a in enumerate(articles["response"]["docs"]):
        try:
            data = {}
            url = resolveComponents(a["web_url"])
            extract = g.extract(url=url)
            data["headline"] = a.get("headline", None)
            data["article_url"] = url
            data["article"] = unidecode.unidecode(extract.cleaned_text)
            data["abstract"] = a.get("abstract", None)
            if a["multimedia"]:
                data["images"] = {}
                soup, figcap = get_soup(url)
                figcap = [c for c in figcap if c.text]
                for ix, cap in enumerate(figcap):
                    if cap.parent.attrs.get("itemid", 0):
                        img_url = resolveComponents(cap.parent.attrs["itemid"])
                        img_data = requests.get(img_url, stream=True).content
                        with open(
                            os.path.join(root + "/images/%s_%d.jpg" % (a["_id"], ix)),
                            "wb",
                        ) as f:
                            f.write(img_data)
                        text = cap.get_text().split("credit")[0]
                        text = text.split("Credit")[0]
                        data["images"].update({ix: text})
            sys.stdout.write(
                "\r%d/%d text documents processed..." % (num, len_articles)
            )
            sys.stdout.flush()
            month_data[a["_id"]] = data
        except Exception as e:
            leftovers[a["_id"]] = a["web_url"]
            print(e, url)

    with open(data_root + "nytimes_%d_%d.json" % (year, m), "w") as f:
        json.dump(month_data, f)
    with open(data_root + "leftovers_%d_%d.json" % (year, m), "w") as f:
        json.dump(leftovers, f)


def main(num_pool, months):
    pool = ThreadPool(num_pool)
    leftovers = {}
    pool.map(retrieve_articles, months)
    pool.close()
    pool.join()
    with open(root + "leftovers_%s.json" % year, "w") as f:
        json.dump(leftovers, f)


if __name__ == "__main__":
    hdr = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Charset": "ISO-8859-1,utf-8;q=0.7,*;q=0.3",
        "Accept-Encoding": "none",
        "Accept-Language": "en-US,en;q=0.8",
        "Connection": "keep-alive",
    }

    h = HTMLParser()
    root = "./"
    data_root = os.path.join(root, "data/")
    g = Goose()

    # Special case for 2018
    year = 2018
    main(6, range(1, 7))

    num_pool = 12
    months = range(1, 13)
    years = range(2010, 2018)
    for y in reversed(years):
        year = y
        main(num_pool, months)


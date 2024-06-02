import argparse
import json
import os
import posixpath
import sys
import urllib.parse

import requests
from joblib import Parallel, delayed


def resolveComponents(url):
    """
    >>> resolveComponents('http://www.example.com/foo/bar/../../baz/bux/')
    'http://www.example.com/baz/bux/'
    >>> resolveComponents('http://www.example.com/some/path/../file.ext')
    'http://www.example.com/some/file.ext'
    """

    parsed = urllib.parse.urlparse(url)
    new_path = posixpath.normpath(parsed.path)
    if parsed.path.endswith("/"):
        # Compensate for issue1707768
        new_path += "/"
    cleaned = parsed._replace(path=new_path)

    return cleaned.geturl()


def retrieve_imgs(args):
    thread_num, keys = args
    leftovers = []
    len_images = len(keys)

    for num, key in enumerate(keys):
        for ix, url in all_img_url[key].items():
            try:
                img_url = resolveComponents(url)
                img_data = requests.get(img_url, stream=True).content
                with open(
                    os.path.join("../../images/%s_%d.jpg" % (key, int(ix))), "wb"
                ) as f:
                    f.write(img_data)
            except Exception as e:
                leftovers.append(key)
                print(e, url)

        sys.stdout.write("\r%d/%d text documents processed..." % (num + 1, len_images))
        sys.stdout.flush()
    if leftovers:
        with open(root + "leftovers_%s.json" % thread_num, "w") as f:
            json.dump(leftovers, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_thread", type=int, default=8, help="How many threads you want to use"
    )
    opt = parser.parse_args()

    root = "./"
    num_thread = opt.num_thread

    with open("./img_urls_all.json") as f:
        all_img_url = json.load(f)

    keys = list(all_img_url.keys())
    thread_range = len(keys) // num_thread + 1
    args = [
        (i + 1, keys[thread_range * i : thread_range * (i + 1)])
        for i in range(num_thread)
    ]

    results = Parallel(n_jobs=num_thread, verbose=0, backend="loky")(
        map(delayed(retrieve_imgs), args)
    )

import os
import numpy
import pandas as pd
import random
import math


def sigmoid(vector):
    out = []
    for x in vector:
        out.append(1 / (1 + math.exp(-x)))
    return out


def probably(chance):
    return random.random() < chance


def flatten(tensor):
    tensor = tensor.reshape(1, -1)
    tensor = tensor.squeeze()
    return tensor


def sum_lists(a1, a2):
    sum = []
    for (x, y) in zip(a1, a2):
        sum.append(x + y)
    return sum


def divide_list(a1, n):
    div = []
    for x in a1:
        div.append(x/n)
    return div


def invert_list(a1):
    inv = []
    for x in a1:
        inv.append(1/x)
    return inv


def average_list(a1):
    return sum(a1)/len(a1)


def flatten_list(t):
    return [item for sublist in t for item in sublist]


def softmax(vector):
    e = numpy.exp(vector)
    return e / e.sum()


def load_langdata(lang, ex=False):
    dir = "./data/document_embeds/" + lang
    distances = {}
    if ex:
        ex_list = ["results", "add", "add_w", "mult", "mult_w"]
    else:
        ex_list = ["results"]
    for incarnation in [x for x in os.listdir(dir) if ".csv" in x]:
        i_name = os.path.splitext(incarnation)[0]
        sepr = " "
        if i_name not in ex_list:
            distances[i_name] = pd.read_csv(dir + "/" + incarnation, sep=sepr, engine='python', index_col=0)

    chunks = distances["lemma"].columns
    colnamesx = {}
    rownamesx = {}
    for i, chunk in enumerate(chunks):
        rownamesx[i] = chunk
        colnamesx[str(i)] = chunk

    for d in distances:
        if distances[d].columns.tolist()[0] == "Unnamed: 0":
            distances[d] = distances[d].set_index("Unnamed: 0")
        distances[d].rename(rownamesx, inplace=True, axis=0)
        distances[d].rename(colnamesx, inplace=True, axis=1)
        if d == "bert":
            distances[d] = distances[d].transform(lambda x: 1-x)

    return distances


def get_langs(path="./data/document_embeds"):
    return next(os.walk(path))[1]


def a_n(lang, plus=False):
    data = load_langdata(lang)
    chunks = [x for x in data["lemma"].columns if x!= "Unnamed: 0"]
    try:
        novels = list(set([x.split("_")[0] + "_" + x.split("_")[1] for x in chunks]))
    except:
        print([x for x in chunks if "_" not in x])
    authors = [x.split("_")[0] for x in novels]
    authors = list(set(authors))
    authors_novels = {}

    for a in authors:
        authors_novels[a] = dict.fromkeys([x for x in novels if x.split("_")[0] == a])
        for n in authors_novels[a]:
            authors_novels[a][n] = [x for x in chunks if x.split("_")[0] + "_" + x.split("_")[1] == n]

    if plus:
        return authors_novels, authors, novels, chunks
    else:
        return authors_novels


def get_author_single(authors_novels):

    list = []
    for a in authors_novels:
        if len(authors_novels[a]) < 2:
            list.append(a)
    return list

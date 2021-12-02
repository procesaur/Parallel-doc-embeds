import os
import numpy
import pandas as pd


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


def softmax(vector):
    e = numpy.exp(vector)
    return e / e.sum()


def load_langdata(lang, ex=False):
    dir = "./data/" + lang
    distances = {}
    if ex:
        ex_list = ["results", "add", "add_w", "mult", "mult_w"]
    else:
        ex_list = ["results"]
    for incarnation in os.listdir(dir):
        i_name = os.path.splitext(incarnation)[0]
        if i_name not in ex_list:
            distances[i_name] = pd.read_csv(dir + "/" + incarnation, sep=" ")
    return distances


def get_langs(path="./data"):
    return next(os.walk(path))[1]


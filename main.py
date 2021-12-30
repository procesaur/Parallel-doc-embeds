import random
import torchworks
from sklearn.metrics import classification_report, precision_recall_fscore_support as score, accuracy_score

from helpers import *
import math
import numpy as np


def classify_and_report(df):
    df['guess'] = df[df.columns].idxmin(axis=1)
    guesses = []
    correct = []
    for i, row in df.iterrows():
        guesses.append(row["guess"].split("_")[0])
        correct.append(i.split("_")[0])


    macro_prec, macro_rcl, macro_f, support = score(correct, guesses,
                                                    average='macro', zero_division=1)
    #w_prec, w_rcl, w_f, support = score(correct, guesses, average='weighted', zero_division=1)
    acc = accuracy_score(correct, guesses)
    vals = [macro_prec, macro_rcl, macro_f, acc]
    return vals


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


def classification_test(lang, easy=False):
    data = load_langdata(lang)
    authors_novels, authors, novels, chunks = a_n(lang, plus=True)

    results_easy = {}
    results_hard = {}
    loss = {}

    items, classes = get_test_set(authors_novels)
    single_authors= get_author_single(authors_novels)

    for df_name in data:
        if easy:
            # easy test
            df_easy = data[df_name].copy()
            df_easy = df_easy.drop(index=[x for x in chunks if x not in items],
                                   columns=[x for x in chunks if x not in classes])
            results_easy[df_name] = classify_and_report(df_easy)

        else:
            #hard test
            df_hard = data[df_name].copy()
            chunks = df_hard
            drop_items = []
            for x in chunks:
                for y in chunks:
                    xs = x.split("_")
                    if xs[0] in single_authors:
                        drop_items.append(x)
                    else:
                        ys = y.split("_")
                        if xs[0] == ys[0] and xs[1] == ys[1]:
                            df_hard.at[x, y] = np.inf

            df_hard = df_hard.drop(index=drop_items)
            results_hard[df_name] = classify_and_report(df_hard)

    #for df_name in data:
    #    loss[df_name] = get_loss(data[df_name])

    print("\t".join(["model", "prec", "rec", "f1", "acc"]))
    embeds = ["word", "pos", "lemma", "add", "mult", "add_w", "mult_w"]
    embeds = ["bert", "word", "pos", "lemma"]

    if easy:
        for df_name in results_easy:
            print(df_name+"\t" + "\t".join([str(round(x, 4)) for x in results_easy[df_name]]))
    else:
        for df_name in results_hard:
            print(df_name + "\t" + "\t".join([str(round(x, 4)) for x in results_hard[df_name]]))


def generate_comp(lemma, pos, word, method, lang, name):
    data = load_langdata(lang)
    if method == "mult":
        df = (data["lemma"]+lemma)*(data["pos"]+pos)*(data["word"]+word)
    else:
        df = data["lemma"] * lemma + data["pos"] * pos + data["word"] * word
    df.to_csv(path_or_buf="./data/document_embeds/"+lang+"/"+name+".csv", sep=" ", float_format='%.7f')
    return df


def generate_comp_all(method, lang, name, bert=False):
    data = load_langdata(lang)
    methods = ["add", "mult", "min", "max", "vnorm"]
    if bert:
        csvs = [x for x in data if x not in methods]
    else:
        csvs = [x for x in data if x not in methods and x != "bert"]
    dflist = [data[x] for x in data if x in csvs]
    if method == "mult":
        for i, x in enumerate(csvs):
            if i == 0:
                df = data[x]
            else:
                df = df*data[x]

    elif method == "add":
        for i, x in enumerate(csvs):
            if i == 0:
                df = data[x]
            else:
                df = df+data[x]
        df = df/len(data)

    elif method == "min":
        df = pd.concat(dflist).min(level=0)

    elif method == "max":
        df = pd.concat(dflist).max(level=0)

    elif method == "vnorm":
        for i, x in enumerate(csvs):
            if i == 0:
                df = data[x]**2
            else:
                df = df+data[x]**2
        df = df.transform(lambda x: [math.sqrt(y) for y in x])

    if bert:
        df.to_csv(path_or_buf="./data/document_embeds/" + lang + "/" + name + "_b.csv", sep=" ", float_format='%.7f')
    else:
        df.to_csv(path_or_buf="./data/document_embeds/" + lang + "/" + name + ".csv", sep=" ", float_format='%.7f')
    return df


def gen_combinations(bert=False, lang=""):
    langs = get_langs()
    if lang != "":
        langs = [lang]
    for lang in langs:
        for method in ["mult", "add", "min", "max", "vnorm"]:
            generate_comp_all(method, lang, method, bert)


def generate_csvs_with_weights():

    lemma = {"mult": {}, "add": {}}
    word = {"mult": {}, "add": {}}
    pos = {"mult": {}, "add": {}}

    with open("weights.csv", "r", encoding="utf8") as w:
        weights = w.readlines();

    for x in weights:
        info = x.rstrip().split("\t")
        ln = info[0]
        method = info[1]
        l = float(info[2])
        p = float(info[3])
        w = float(info[4])

        lemma[method][ln] = l
        pos[method][ln] = p
        word[method][ln] = w

    langs = next(os.walk('./data/document_embeds'))[1]
    for lang in langs:
        for method in ["mult", "add"]:
            if method == "mult":
                generate_comp_all(method, lang, method)
                if weights:
                    generate_comp(lemma[method][lang], pos[method][lang], word[method][lang], method, lang, method+"_w")
            else:
                generate_comp_all(method, lang, method)
                if weights:
                    generate_comp(lemma[method][lang], pos[method][lang], word[method][lang], method, lang, method+"_w")


def all_classification_report():
    langs = get_langs()
    for lang in langs:
        print(lang + "  > ")
        classification_test(lang)


def get_author_single(authors_novels):

    list = []
    for a in authors_novels:
        if len(authors_novels[a])<2:
            list.append(a)
    return list


def get_test_set(authors_novels):
    random.seed(1)
    rand1 = random.randint(0, 2)
    rand2 = random.randint(0, 2)
    items = []
    classes = []
    for author in authors_novels:

        novels_with_n_chunks = [x for x in authors_novels[author] if len(authors_novels[author][x])>2]
        # if there is at list n such novels
        if len(novels_with_n_chunks) > 2:
            ablenovs = novels_with_n_chunks[:3]
        else:
            ablenovs = []

        for i, novel in enumerate(ablenovs):
            if i == rand1:
                classes.append(authors_novels[author][novel][rand2])
            else:
                items += authors_novels[author][novel][:3]
            #testch[novel] = random.sample(authors_novels[author][novel], 3)

    return items, classes


# for each language
for lang in get_langs():
    # generate simple combinations without bert
    gen_combinations(bert=False, lang=lang)
    # generate simple combinations without bert
    gen_combinations(bert=True, lang=lang)

    # train weights without bert
    torchworks.train_mini(lang=lang, bert=False)
    # train weights with bert
    torchworks.train_mini(lang=lang, bert=True)

    # all_classification_report()
    classification_test(lang)

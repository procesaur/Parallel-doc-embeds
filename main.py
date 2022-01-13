import torch
from sklearn.metrics import precision_recall_fscore_support as score, multilabel_confusion_matrix, confusion_matrix
from sklearn.metrics import accuracy_score, fbeta_score
import numpy as np
from helpers import *
import torchworks
import sys
numpy.set_printoptions(threshold=sys.maxsize)


def classify_and_report(df):
    df['guess'] = df[df.columns].idxmin(axis=1)
    guesses = []
    correct = []
    for i, row in df.iterrows():
        guesses.append(row["guess"].split("_")[0])
        correct.append(i.split("_")[0])

    macro_prec, macro_rcl, macro_f, support = score(correct, guesses, average='macro', zero_division=1)
    fbeta = fbeta_score(correct, guesses, beta=0.5, average='macro')
    acc = accuracy_score(correct, guesses)
    vals = [acc, macro_prec, macro_rcl, macro_f, fbeta]
    conf = confusion_matrix(correct, guesses)
    corr_guess = ""
    for i, x in enumerate(correct):
        corr_guess += "\n" + x + " > " + guesses[i]
    conf = corr_guess

    return vals, macro_f, conf


def classification_test(lang, easy=False):
    print("\n" + lang + ">>")
    data = load_langdata(lang)
    authors_novels, authors, novels, chunks = a_n(lang, plus=True)

    baseline = ["bert", "word", "pos", "lemma"]
    baseline2 = ["masked_2"]
    comb = ["add", "max", "min", "mult", "vnorm"]
    comb = flatten_list([[x, x + "_b"] for x in comb])
    weighted = [x for x in data if "weight" in x]
    csvs = baseline + baseline2 + comb + weighted

    results = {}

    items, classes = get_test_set(authors_novels)
    single_authors = get_author_single(authors_novels)

    confusion = {}

    for df_name in csvs:
        confusion[df_name] = {}

        df = data[df_name].copy()
        chunks = df
        drop_items = []
        for x in chunks:
            for y in chunks:
                xs = x.split("_")
                if xs[0] in single_authors:
                    drop_items.append(x)
                else:
                    ys = y.split("_")
                    if xs[0] == ys[0] and xs[1] == ys[1]:
                        df.at[x, y] = np.inf

        df = df.drop(index=drop_items)
        vals, f1, conf = classify_and_report(df)
        results[df_name] = vals
        confusion[df_name] = f1, conf

    base_top, base_conf = top(confusion, baseline)
    imp_top, inp_conf = top(confusion, comb+weighted)

    print("\t".join(["model", "acc", "prec", "rec", "f-1", "f-0.5"]))

    for df_name in results:
        print(df_name + "\t" + "\t".join([str(round(x, 4)) for x in results[df_name]]))
    print("best base > " + base_top + " > ")
    print(base_conf)
    print("best new > " + imp_top + " > ")
    print(inp_conf)


def top(dic, list):
    best = 0
    name = ""
    conf = ""
    for x in list:
        val = dic[x][0]
        if val >= best:
            best = val
            name = x
            conf = dic[x][1]
    return name, conf


def generate_comp_all(method, lang, name, bert=False):
    data = load_langdata(lang)
    if bert:
        csvs = ["bert", "lemma", "pos", "word", "masked_2"]
    else:
        csvs = ["lemma", "pos", "word", "masked_2"]
    columns = data["lemma"].columns
    dflist = [data[x] for x in data if x in csvs]
    if method == "mult":
        df = data[0]
        for i, x in enumerate(csvs):
            if i > 0:
                df = df*data[x]

    elif method == "add":
        df = data[0]
        for i, x in enumerate(csvs):
            if i > 0:
                df = df+data[x]
        df = df/len(data)

    elif method == "min":
        df = pd.concat(dflist).min(level=0)

    elif method == "max":
        df = pd.concat(dflist).max(level=0)

    elif method == "vnorm":
        df = data[0] ** 2
        for i, x in enumerate(csvs):
            if i > 0:
                df = df+data[x]**2
        df = df.transform(lambda x: [math.sqrt(y) for y in x])

    else:
        dfs = []
        for x in csvs:
            dfs.append(torch.tensor(data[x].values))
        dfst = torch.stack(dfs)
        mpath = "./data/weights/universal_M"
        if bert:
            mpath += "_b"
        mpath += "-"+lang
        df = torchworks.test_mini(dfst, modelpath=mpath, bert=bert)
        df = df.detach().numpy()
        df = numpy.squeeze(df, axis=2)
        df = pd.DataFrame(df, columns=columns, index=columns)

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


def generate_csvs_with_weights(lang_weights, lang_apply, bert=False, exc=""):

    wanted = ["pos", "word", "lemma", "masked_2"]

    if bert:
        lang_weights += "_b"
        wanted.append("bert")

    if exc != "":
        lang_weights += "-" + exc
    path = "./data/weights/" + lang_weights
    weights = torchworks.get_weights(path)
    path_apply = "./data/document_embeds/" + lang_apply + "/"

    wanted = sorted(wanted)
    matrices = []
    for x in wanted:
        matrices.append(pd.read_csv(path_apply + x + ".csv", sep=" ", engine='python', index_col=0))

    for i, w in enumerate(weights):
        dfs = matrices[i]
        if wanted[i] == "bert":
            dfs = 1-dfs
        if i == 0:
            df = dfs*w
        else:
            df += dfs*w

    df = df / numpy.sum(weights)

    df.to_csv(path_or_buf=path_apply + "/weights_" + lang_weights + ".csv", sep=" ", float_format='%.7f')


def all_classification_report():
    langs = get_langs()
    for lang in langs:
        classification_test(lang)


def get_test_set(authors_novels):
    random.seed(1)
    rand1 = random.randint(0, 2)
    rand2 = random.randint(0, 2)
    items = []
    classes = []
    for author in authors_novels:

        novels_with_n_chunks = [x for x in authors_novels[author] if len(authors_novels[author][x]) > 2]
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
            # testch[novel] = random.sample(authors_novels[author][novel], 3)

    return items, classes


def transfer_learning(bert=False):
    path = "./data/weights/"
    langs = get_langs()
    savepath = "./data/lang_weight_distances.csv"
    if bert:
        savepath = "./data/lang_weight_distances_b.csv"
        langs = [x+"_b" for x in langs]
    df = pd.DataFrame(columns=langs, index=langs)
    for lang1 in langs:
        l1w = torchworks.get_weights(path + lang1)
        for lang2 in langs:
            l2w = torchworks.get_weights(path + lang2)
            value = numpy.linalg.norm(np.array(l1w)-np.array(l2w))
            if lang1 == lang2:
                df[lang1][lang2] = numpy.Inf
            else:
                df[lang1][lang2] = value

    df.to_csv(savepath, sep=",", float_format='%.7f')
    df = df.astype(float)
    df['closest'] = df.idxmin(axis=1)
    for lang1 in langs:
        lang2 = df['closest'][lang1]
        generate_csvs_with_weights(lang2.replace("_b", ""), lang1.replace("_b", ""), bert=bert)


def write_weights(path="./data/weights/"):
    with_b = [x for x in os.listdir(path) if "_b" in x]
    no_b = [x for x in os.listdir(path) if "_b" not in x]
    wanted = ["pos", "word", "lemma", "masked_2"]
    wanted = sorted(wanted)
    print("\t" + "\t".join(wanted)+"\tbert")
    for x in no_b:
        print(x+"\t"+"\t".join([str(v) for v in torchworks.get_weights(path + x)]))
    for x in with_b:
        res = torchworks.get_weights(path + x)
        res.append(res.pop(0))
        print(x+"\t"+"\t".join([str(v) for v in res]))


def main():
    # for each language
    for lang in get_langs():
        if lang != "srp":

            # train weights without bert
            torchworks.train_mini(lang=lang, bert=False)
            # train weights with bert
            torchworks.train_mini(lang=lang, bert=True)

            # train universal weights without bias
            torchworks.train_mini(lang="universal", exc=lang, bert=False)
            torchworks.train_mini(lang="universal", exc=lang, bert=True)

            # generate simple combinations without bert
            gen_combinations(bert=False, lang=lang)
            # generate simple combinations without bert
            gen_combinations(bert=True, lang=lang)

            # generate combinations using universal weights
            generate_csvs_with_weights("universal", lang, exc=lang, bert=False)
            generate_csvs_with_weights("universal", lang, exc=lang, bert=True)

    transfer_learning()
    transfer_learning(True)
    write_weights()
    all_classification_report()


classification_test("srp")
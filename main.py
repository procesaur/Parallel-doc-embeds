from sklearn.metrics import classification_report, precision_recall_fscore_support as score, accuracy_score
import numpy as np
from helpers import *
import torchworks


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


def classification_test(lang, easy=False):
    print("\n" + lang + ">>")
    data = load_langdata(lang)
    authors_novels, authors, novels, chunks = a_n(lang, plus=True)

    results_easy = {}
    results_hard = {}
    loss = {}

    items, classes = get_test_set(authors_novels)
    single_authors = get_author_single(authors_novels)

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


def generate_csvs_with_weights(lang_weights, lang_apply, bert=False):
    wanted = ["pos", "word", "lemma", "masked_2", "masked_3"]
    modelname = "miniNN"
    if bert:
        wanted.append("bert")
        modelname = "miniNN_b"

    wanted = sorted(wanted)
    path = "./data/document_embeds/" + lang_weights + "/" + modelname
    weights = torchworks.get_weights(path)
    #weights = sigmoid(weights)
    path_apply = "./data/document_embeds/" + lang_apply + "/"
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
    #df = df.transform(lambda x: [-y for y in x])
    df.to_csv(path_or_buf="./data/document_embeds/" + lang_apply + "/" + modelname + ".csv", sep=" ", float_format='%.7f')


def all_classification_report():
    langs = get_langs()
    for lang in langs:
        print(lang + "  > ")
        classification_test(lang)


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


#torchworks.train_mini(lang="srp", bert=False)
#torchworks.train_mini(lang="srp", bert=True)
#torchworks.train_mini(lang="por", bert=False)
#torchworks.train_mini(lang="por", bert=True)
# generate_csvs_with_weights("srp", "srp", bert=False)
#generate_csvs_with_weights("srp", "srp", bert=True)
#generate_csvs_with_weights("srp", "slv", bert=False)
#generate_csvs_with_weights("srp", "slv", bert=True)
generate_csvs_with_weights("srp", "fra", bert=True)
generate_csvs_with_weights("srp", "fra", bert=False)
#generate_csvs_with_weights("por", "por", bert=True)
#generate_csvs_with_weights("por", "por", bert=False)

#classification_test("srp")
#classification_test("slv")
#classification_test("por")
classification_test("fra")


# pipeline
if False:
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

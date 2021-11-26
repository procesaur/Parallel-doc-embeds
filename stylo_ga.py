import json
import numpy
import ga
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support as score, accuracy_score


def eltec_fitness(distances, pop, method):
    fitness = []
    onedf = distances[list(distances.keys())[0]]

    cols = onedf.columns.tolist()
    rows = onedf.index.tolist()

    for peep in pop:
        if method == "mult":
            data = pd.DataFrame(numpy.ones((100, 100)), columns=cols, index=rows)
        else:
            data = pd.DataFrame(numpy.zeros((100, 100)), columns=cols, index=rows)

        for i, inc in enumerate(distances.keys()):
            if method == "mult":
                data = data * (distances[inc] + peep[i])
            else:
                data = data + (distances[inc] * peep[i])

        if method != "mult":
            data = data/len(distances.keys())

        if method == "mult":
            df = data - numpy.prod(peep)
        else:
            df = data

        fitness.append(1/get_loss(df))

        good = []
        bad = []

        #for row in rows:
        #    for col in cols:
        #        val = df.loc[row, col]
        #        if val != 0:
        #            if col.split("_")[0] == row.split("_")[0]:
        #                good.append(val)
        #            else:
        #                bad.append(val)

        #fitness.append(get_error(good, bad))

    return fitness


def test():
    data = load_langdata("de")
    lemma = 7.27
    pos = 2.68
    word = 0.52

    df = (data["lemma"]+lemma)*(data["pos"]+pos)*(data["word"]+word)
    val = lemma*pos*word
    df = (df - val)/20
    fitness = ga.eltec_single_fitness(df)
    i=1
    return fitness


def classify_and_report(df):

    df['guess'] = df[df.columns].idxmin(axis=1)
    guesses = []
    correct = []
    for i, row in df.iterrows():
        guesses.append(row["guess"].split("_")[0])
        correct.append(i.split("_")[0])


    macro_prec, macro_rcl, macro_f, support = score(correct, guesses,
                                                    average='macro', zero_division=1)
    w_prec, w_rcl, w_f, support = score(correct, guesses, average='weighted', zero_division=1)
    acc = accuracy_score(correct, guesses)
    vals = [w_prec, w_rcl, w_f, macro_prec, macro_rcl, macro_f, acc]
    return vals


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


def get_loss(df):
    if df.columns.tolist()[0] == "Unnamed: 0":
        df = df.set_index("Unnamed: 0")
    loss = []
    for index, row in df.iterrows():
        p = []
        qlist = [x for x in row.tolist() if x != 0]
        q = softmax([max(qlist)-x for x in qlist])
        rowdict = row.to_dict()
        for col in rowdict:
            if rowdict[col] != 0:
                if col.split("_")[0] == index.split("_")[0]:
                    p.append(1)
                else:
                    p.append(0)

        cel = ([p[i]*numpy.log2(q[i]) for i in range(len(p)) if p[i]!=0])
        if cel:
            loss.append(-average_list(cel))


    loss = [x for x in loss if x != 0]
    lossX = average_list(loss)
    return lossX


def classification_test(lang, single_limit=0, class_limit=0, item_limit=0, n=100):
    data = load_langdata(lang)
    novels = data["lemma"].columns
    authors = [x.split("_")[0] for x in novels]
    authors = list(set(authors))
    authors_novels = {}

    for a in authors:
        authors_novels[a] = [x for x in novels if x.split("_")[0] == a]

    single = []
    classes = []
    items = []
    results = {}
    loss = {}

    for df_name in data:
        loss[df_name] = get_loss(data[df_name])

    for a in authors_novels:
        nn = len(authors_novels[a])
        if nn == 1:
            single.append(authors_novels[a][0])
        else:
            pick = numpy.random.randint(0, nn)
            for i, x in enumerate(authors_novels[a]):
                if i == pick:
                    classes.append(authors_novels[a][i])
                else:
                    items.append(authors_novels[a][i])

    for i in range(0, n):

        if single_limit > 0:
            single = numpy.random.choice(single, single_limit).tolist()
        if class_limit > 0:
            classes = numpy.random.choice(classes, class_limit).tolist()
        if item_limit > 0:
            items = numpy.random.choice(items, item_limit).tolist()

        all_classes = single + classes

        for df_name in data:
            df = data[df_name]
            if df.columns.tolist()[0] == "Unnamed: 0":
                df = df.set_index("Unnamed: 0")
            df = df.drop(index=[x for x in novels if x not in items],
                         columns=[y for y in novels if y not in all_classes])

            if i == 0:
                results[df_name] = classify_and_report(df)
            else:
                results[df_name] = sum_lists(results[df_name], classify_and_report(df))

    if n > 1:
        for df_name in results:
            results[df_name] = divide_list(results[df_name], n)

    print("\t".join(["model", "w_prec", "w_rec", "w_f1", "m_prec", "m_rec", "m_f1", "acc", "loss"]))
    for df_name in ["word", "pos", "lemma", "add", "mult", "add_w", "mult_w"]:
    #for df_name in results:
        print(df_name+"\t" + "\t".join([str(round(x, 4)) for x in results[df_name]]) + "\t" + str(loss[df_name]))


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


def generate_comp(lemma, pos, word, method, lang, name):
    data = load_langdata(lang)
    if method == "mult":
        df = (data["lemma"]+lemma)*(data["pos"]+pos)*(data["word"]+word)
    else:
        df = data["lemma"] * lemma + data["pos"] * pos + data["word"] * word
        ind = data["lemma"].index.tolist()
    df.to_csv(path_or_buf="./data/"+lang+"/"+name+".csv", sep=" ")
    return df


def get_langs(path="./data"):
    return next(os.walk(path))[1]


def fitness_comparison():

    langs = get_langs()
    for lang in langs:
        print(lang+": ")
        df = load_langdata(lang)
        for data in ["word", "lemma", "pos", "add", "mult", "add_w", "mult_w"]:
            if df[data].columns.tolist()[0] == "Unnamed: 0":
                df[data] = df[data].set_index("Unnamed: 0")


            fitness = ga.eltec_single_fitness(df[data])
            print(data+ " > " + str(fitness))


def generate_csvs():
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

    langs = next(os.walk('./data'))[1]
    for lang in langs:
        for method in ["mult", "add"]:
            if method == "mult":
                generate_comp(0, 0, 0, method, lang, method)
                generate_comp(lemma[method][lang], pos[method][lang], word[method][lang], method, lang, method+"_w")
            else:
                generate_comp(1, 1, 1, method, lang, method)
                generate_comp(lemma[method][lang], pos[method][lang], word[method][lang], method, lang, method+"_w")


def get_weights(out="weights.csv"):
    sol_per_pop = 48
    num_parents_mating = 16
    num_generations = 220
    stale_max = 15
    tries = 2

    methods = ["add", "mult"]
    langs = next(os.walk('./data'))[1]
    for method in methods:
        for lang in langs:
            for _ in range(0, tries):
                data = load_langdata(lang, True)
                num_weights = len(data.keys())

                # Defining the population size.
                pop_size = (sol_per_pop, num_weights)
                # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
                # Creating the initial population.
                new_population = numpy.random.uniform(low=0.5, high=1.5, size=pop_size)

                #print(new_population)
                best_fit = 0
                best_peep = []
                stale_c = 0
                bar = tqdm(range(num_generations))
                for generation in bar:

                    # Measing the fitness of each chromosome in the population.
                    #fitness = ga.cal_pop_fitness(equation_inputs, new_population)

                    fitness = eltec_fitness(data, new_population, method)
                    fit_max = numpy.max(fitness)

                    best_match_idx = numpy.where(fitness == fit_max)
                    peep_max = numpy.round(new_population[best_match_idx, :][0][0],4)

                    if fit_max > best_fit:
                        best_fit = fit_max
                        best_peep = peep_max
                        stale_c = 0
                    else:
                        stale_c += 1

                    if stale_c > stale_max:
                        break

                    # Selecting the best parents in the population for mating.
                    parents = ga.select_mating_pool(new_population, fitness, num_parents_mating)

                    # Generating next generation using crossover.
                    offspring_crossover = ga.crossover(parents, offspring_size=(pop_size[0]-parents.shape[0], num_weights))

                    # Adding some variations to the offsrping using mutation.
                    offspring_mutation = ga.mutation(offspring_crossover)

                    # Creating the new population based on the parents and offspring.
                    new_population[0:parents.shape[0], :] = parents
                    new_population[parents.shape[0]:, :] = offspring_mutation

                    # The best result in the current iteration.
                    bar.set_description(lang + " > Best result : " + str(round(1/best_fit, 2)) + " (stale:" + str(stale_c) +
                                        ") with " + json.dumps(dict(zip(data.keys(), best_peep))))
                with open(out, "a+", encoding="utf8") as o:
                    o.write("\t".join([lang, method, str(best_peep[0]), str(best_peep[1]), str(best_peep[2])])+"\n")


def all_classification_report():
    langs = get_langs()
    for lang in langs:
        print(lang + "  > ")
        classification_test(lang, n=10, single_limit=10)


#generate_csvs()
#fitness_comparison()
all_classification_report()
#get_weights()
import json
import numpy
import ga
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support as score


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


def do_classification(df, name):
    print(name+" > ")
    df['guess'] = df[df.columns].idxmin(axis=1)
    guesses = []
    correct = []
    for i, row in df.iterrows():
        guesses.append(row["guess"].split("_")[0])
        correct.append(i.split("_")[0])

    print(classification_report(correct, guesses, zero_division=1, digits=3))



def classification_test(lang, single_limit=0, class_limit=0, item_limit=0, n=1):
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

    for _ in range(0, n):

        if single_limit > 0:
            single = numpy.random.choice(single, single_limit)
        if class_limit > 0:
            classes = numpy.random.choice(classes, class_limit)
        if item_limit > 0:
            items = numpy.random.choice(items, item_limit)

        all_classes = single + classes

        for df_name in data:
            df = data[df_name]
            if df.columns.tolist()[0] == "Unnamed: 0":
                df = df.set_index("Unnamed: 0")
            df = df.drop(index=[x for x in novels if x not in items],
                         columns=[y for y in novels if y not in all_classes])

            do_classification(df, df_name)


def load_langdata(lang):
    dir = "./data/" + lang
    distances = {}
    for incarnation in os.listdir(dir):
        i_name = os.path.splitext(incarnation)[0]
        #if i_name not in ["results", "add", "add_w", "mult", "mult_w"]:
        if i_name not in ["results"]:
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


def fitness_comparison():

    langs = next(os.walk('./data'))[1]
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

    lemma["mult"]["de"] = -1.3304
    pos["mult"]["de"] = 1.8224
    word["mult"]["de"] = 1.4776

    lemma["add"]["de"] = 2.0391
    pos["add"]["de"] = -0.4901
    word["add"]["de"] = 0.5054

    lemma["mult"]["en"] = 3.4581
    pos["mult"]["en"] = -2.0777
    word["mult"]["en"] = 1.3092

    lemma["add"]["en"] = 2.024
    pos["add"]["en"] = 4.7051
    word["add"]["en"] = 0.5185

    lemma["mult"]["fr"] = -1.955
    pos["mult"]["fr"] = 3.9049
    word["mult"]["fr"] = 1.4654

    lemma["add"]["fr"] = 2.3647
    pos["add"]["fr"] = -0.7044
    word["add"]["fr"] = 0.5135

    lemma["mult"]["hu"] = -1.0439
    pos["mult"]["hu"] = 3.2411
    word["mult"]["hu"] = 1.3983

    lemma["add"]["hu"] = 3.0283
    pos["add"]["hu"] = -0.1518
    word["add"]["hu"] = 0.5156

    lemma["mult"]["pt"] = 1.6999
    pos["mult"]["pt"] = 0.0579
    word["mult"]["pt"] = 1.4602

    lemma["add"]["pt"] = 0.3619
    pos["add"]["pt"] = 1.6807
    word["add"]["pt"] = 1.0965

    lemma["mult"]["rs"] = 0.6507
    pos["mult"]["rs"] = 1.8714
    word["mult"]["rs"] = 0.998

    lemma["add"]["rs"] = 1.6053
    pos["add"]["rs"] = 1.0068
    word["add"]["rs"] = 1.3089

    lemma["mult"]["si"] = -1.1162
    pos["mult"]["si"] = 3.4362
    word["mult"]["si"] = 1.4785

    lemma["add"]["si"] = 4.7349
    pos["add"]["si"] = -0.6223
    word["add"]["si"] = 0.6187


    langs = next(os.walk('./data'))[1]
    for lang in langs:
        for method in ["mult", "add"]:
            if method == "mult":
                generate_comp(0, 0, 0, method, lang, method)
                generate_comp(lemma[method][lang], pos[method][lang], word[method][lang], method, lang, method+"_w")
            else:
                generate_comp(1, 1, 1, method, lang, method)
                generate_comp(lemma[method][lang], pos[method][lang], word[method][lang], method, lang, method+"_w")


def get_weights():
    sol_per_pop = 48
    num_parents_mating = 16
    num_generations = 200
    stale_max = 25
    tries = 3

    methods = ["add", "mult"]
    langs = next(os.walk('./data'))[1]
    for method in methods:
        for lang in langs:
            for _ in range(0,tries):
                data = load_langdata(lang)
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

                    fitness = ga.eltec_fitness(data, new_population, method)
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
                    bar.set_description(lang + " > Best result : " + str(round(best_fit, 4)) + " (stale:" + str(stale_c) +
                                        ") with " + json.dumps(dict(zip(data.keys(), best_peep))))

#generate_csvs()
#fitness_comparison()
classification_test("rs")
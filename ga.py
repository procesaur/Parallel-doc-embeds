import numpy
import seaborn as sns
# This project is extended and a library called PyGAD is released to build the genetic algorithm.
# PyGAD documentation: https://pygad.readthedocs.io
# Install PyGAD: pip install pygad
# PyGAD source code at GitHub: https://github.com/ahmedfgad/GeneticAlgorithmPython


def cal_pop_fitness(equation_inputs, pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function caulcuates the sum of products between each input and its corresponding weight.
    fitness = numpy.sum(pop*equation_inputs, axis=1)
    return fitness


def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next.
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents


def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually it is at the center.
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k % parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1) % parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


def mutation(offspring_crossover):
    # Mutation changes a single gene in each offspring randomly.
    for idx in range(offspring_crossover.shape[0]):
        gene = numpy.random.randint(high=offspring_crossover.shape[1]-1, low=0)
        # The random value to be added to the gene.
        random_value = numpy.random.uniform(-0.2, 0.3, 1)
        offspring_crossover[idx, gene] = offspring_crossover[idx, gene] + random_value
    return offspring_crossover


def get_error_hist(x1, x2, bins=100):
    h1 = numpy.histogram(x1, bins=bins, density=True)
    h2 = numpy.histogram(x2, bins=bins, density=True)

    sm = 0
    for i in range(bins):
        sm += min(h1[0][i], h2[0][i])
    return 100-sm


def get_error_int(x1, x2):
    clip = {'clip': (0, 2)}
    sns.distplot(x1, kde_kws=clip)
    ax = sns.distplot(x2, kde_kws=clip)

    # area1 = numpy.trapz(ax.lines[0].get_ydata(), ax.lines[0].get_xdata())
    # area2 = numpy.trapz(ax.lines[1].get_ydata(), ax.lines[1].get_xdata())
    ymin = numpy.minimum(ax.lines[0].get_ydata(), ax.lines[1].get_ydata())
    area_overlap = numpy.trapz(ymin, ax.lines[0].get_xdata())
    return 1-area_overlap


def get_error(a, b):
    maxi = max(a)
    mini = min(b)
    total = len(a)+len(b)
    if maxi < mini:
        return 100
    else:
        int_a = [x for x in a if x > mini]
        int_b = [x for x in b if x < maxi]
        # err_a = len(int_a)/len(a)
        # err_b = len(int_b) / len(b)
        err = (len(int_a)+len(int_b))*100/total
        return 100 - err


def eltec_single_fitness(distances):
    cols = distances.columns.tolist()
    rows = distances.index.tolist()
    df = distances
    good = []
    bad = []
    for row in rows:
        for col in cols:
            val = df.loc[row, col]
            if val != 0:
                if col.split("_")[0] == row.split("_")[0]:
                    good.append(val)
                else:
                    bad.append(val)

    return get_error(good, bad)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

item_count = 10  # number of items
limit = 40  # size of knapsack

weight = np.random.randint(1, 30, size=item_count)  # weight of items
profit = np.random.randint(10, 100, size=item_count)  # profit for each
print('Weight\tProfit')
for i in range(item_count):
    print('%s\t%s' % (weight[i], profit[i]))

population_size = 12  # number of chromosomes in population
population = np.random.randint(2, size=(population_size, item_count))
population = pd.DataFrame(population, columns=(weight))
population['Fitness'] = np.zeros(population_size).astype('int')
print(population)


def calFitness(population, p, w, l):
    for c in range(len(population)):
        chrom = np.array(population.loc[c])[:-1]
        fit = np.dot(chrom, p)
        we = np.dot(chrom, w)
        if (we <= l):
            population.loc[c]['Fitness'] = fit
        else:
            population.loc[c]['Fitness'] = 0


generation = 1  # number of generation
meanfitness = []
maxfitness = []
for g in range(generation):
    calFitness(population, profit, weight, limit)
    population = population.sort_values(by='Fitness')
    population.reset_index(drop=True, inplace=True)

    parent = population[-4:].reset_index(drop=True)  # select 4 fittest parents
    offspring = parent.copy()

    # one point crossover
    for ind in range(int(len(parent) / 2)):
        split = np.random.randint(0, item_count)
        for i in range(split, item_count):
            temp = offspring.loc[ind].iloc[i]
            offspring.loc[ind].iloc[i] = offspring.loc[len(offspring) - 1 -
                                                       ind].iloc[i]
            offspring.loc[len(offspring) - 1 - ind].iloc[i] = temp

    # bit flip mutation
    mutation_probability = 0.5
    for ind in range(len(offspring)):
        for i in range(item_count):
            if (np.random.randn() > mutation_probability):
                offspring.loc[ind].iloc[i] = offspring.loc[ind].iloc[i] ^ 1

    population = population[len(offspring):].append(offspring)
    mean = np.mean(population['Fitness'])
    meanfitness.append(mean)
    mx = max(population['Fitness'])
    maxfitness.append(mx)

    if ((mx == mean) and generation > 50):
        break
    generation += 1

print(population)
plt.plot(meanfitness, label='Mean Fitness')
plt.plot(maxfitness, 'r', label='Max Fitness')
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.legend()
plt.grid()

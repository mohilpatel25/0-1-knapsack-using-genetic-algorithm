"""Implementation of 0/1 knapsack problem using Genetic Algorithm."""

import numpy as np


class Chromosome:
  """Class to manage individual chromosomes in genetic algorithm.
  """

  def __init__(self, weights, profits, knapsack_size) -> None:
    self.weights = weights
    self.profits = profits
    self.knapsack_size = knapsack_size
    self.genes = np.random.randint(0, 2, len(weights))
    self.size = len(self.genes)

  @property
  def fitness(self):
    total_weight = np.dot(self.genes, self.weights)
    fitness = np.dot(self.genes, self.profits)
    if total_weight <= self.knapsack_size:
      return fitness
    return 0

  def __lt__(self, o: object) -> bool:
    return self.fitness > o.fitness

  def __eq__(self, o: object) -> bool:
    return self.fitness == o.fitness

  def __gt__(self, o: object) -> bool:
    return self.fitness < o.fitness

  def single_point_crossover(self, chromosome):
    crossover_point = np.random.randint(1, self.size - 1)
    offspring1 = Chromosome(self.weights, self.profits, self.knapsack_size)
    offspring1.genes = np.concatenate(
        (self.genes[:crossover_point], chromosome.genes[crossover_point:]))
    offspring2 = Chromosome(self.weights, self.profits, self.knapsack_size)
    offspring2.genes = np.concatenate(
        (chromosome.genes[:crossover_point], self.genes[crossover_point:]))
    return offspring1, offspring2

  def mutate(self, mutation_probability):
    self.genes = np.where(
        np.random.random(self.size) < mutation_probability, self.genes ^ 1,
        self.genes)


class GeneticAlgorithm:
  """Class to manage genetic algorithm for 0/1 Knapsack problem.
  """

  def __init__(self,
               weights,
               profits,
               knapsack_size,
               population_size,
               selection_ratio=0.4,
               mutation_prob=0.5) -> None:
    self.weights = weights
    self.profits = profits
    self.knapsack_size = knapsack_size
    self.population_size = population_size
    self.selection_ratio = selection_ratio
    self.mutation_prob = mutation_prob
    self.chromosomes = sorted([
        Chromosome(self.weights, self.profits, self.knapsack_size)
        for i in range(population_size)
    ])

  def crossover(self, parents):
    return parents[0].single_point_crossover(parents[1])

  def mutatation(self, offsprings, mutation_prob):
    for offspring in offsprings:
      offspring.mutate(mutation_prob)
    return offsprings

  def next_generation(self):
    n_selection = int(self.population_size * self.selection_ratio)
    n_selection = (n_selection // 2) * 2
    fittest_individuals = self.chromosomes[:n_selection]

    offsprings = []
    for i in range(0, n_selection, 2):
      offsprings += self.crossover(fittest_individuals[i:i + 2])

    offsprings = self.mutatation(offsprings, self.mutation_prob)

    self.chromosomes += offsprings
    self.chromosomes = sorted(self.chromosomes)[:self.population_size]

  def fittest_chromosome(self):
    return self.chromosomes[0]

  def evolve(self, generations, log_freq=1000):
    for generation in range(1, generations):
      ga.next_generation()
      if generation % log_freq == 0:
        max_profit = self.fittest_chromosome().fitness
        print(f'Generation {generation}: Max Profit = {max_profit}')
      generations += 1
    return self.fittest_chromosome()


if __name__ == '__main__':
  item_count = 5  # number of items
  knapsack_size = 15  # size of knapsack
  population_size = 10

  weights = np.random.randint(1, knapsack_size,
                              size=item_count)  # weight of each item
  profits = np.random.randint(1, 50, size=item_count)  # profit for each item

  print(f'Knapsack Size: {knapsack_size}')
  print('Weight\tProfit')
  for weight, profit in zip(weights, profits):
    print(f'{weight}\t{profit}')

  ga = GeneticAlgorithm(weights=weights,
                        profits=profits,
                        knapsack_size=knapsack_size,
                        population_size=population_size)

  solution = ga.evolve(100)

  print('\nSolution Found')
  print('Weight\tProfit\tSelect')
  for weight, profit, gene in zip(weights, profits, solution.genes):
    print(f'{weight}\t{profit}\t{gene}')
  print(f'Max Profit: {solution.fitness}')

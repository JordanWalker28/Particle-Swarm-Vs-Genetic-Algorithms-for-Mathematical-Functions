#########################################################
# References                                           #

#Mirjalili, S. (2018). [online] Udemy. Available at: https://www.udemy.com/geneticalgorithm/learn/v4/t/lecture/10322200?start=0 [Accessed 24 Nov. 2018].

#De Rainville, F. (2018). DEAP/deap. [online] GitHub. Available at: https://github.com/DEAP/deap/blob/master/deap/tools/crossover.py?fbclid=IwAR0bUGtAvt9eeSD2JJdb912vr4CGppyOB5Z9fTpJjFgv3HQZu0iUUFHYDXk [Accessed 24 Nov. 2018].

#Cℓinton's Blog. (2018). Genetic Algorithm Hello World with Python. [online] Available at: https://handcraftsman.wordpress.com/2015/06/09/evolving-a-genetic-solver-in-python-part-1/ [Accessed 25 Nov. 2018].


#########################################################


import optproblems.cec2005
import optproblems
from numpy import array
import numpy as np
import random


class GA_Class():
    
    def __init__(self, test_function, num_dimensions, bounds ,population, generation_max):
        self.testfunction = test_function
        self.num_dimensions = num_dimensions
        self.bounds = bounds
        self.population = []
        self.maxIterations = 100
        self.crossover_prob = 0.6
        self.mutation = 0.1
        self.popSize = population

        for i in range(0,self.popSize):
            random_indiv = np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=(self.num_dimensions,))
            self.population.append((list(random_indiv),0))
        
    def select_p_tournament(self):
        u1 = 0;
        u2 = 0;
        parent = 99;
        while (u1 == 0 and u2 == 0):
            u1 = np.random.random_integers(self.popSize - 1)
            u2 = np.random.random_integers(self.popSize - 1)
            if self.population[u1][1] <= self.population[u2][1]:
                parent = self.population[u1]
            else:
                parent = self.population[u2]
        return parent

    def two_point_crossover(self, parenta, parentb):
        ind1 = list(parenta)
        ind2 = list(parentb)
        size = min(len(ind1), len(ind2))
        cxpoint1 = random.randint(0, size)
        cxpoint2 = random.randint(0, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else: # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        temp_ind1 = list(ind1)
        ind1[cxpoint1:cxpoint2] = list(ind2[cxpoint1:cxpoint2])
        ind2[cxpoint1:cxpoint2] = temp_ind1[cxpoint1:cxpoint2]
        return ind1, ind2

    def eval_fitness(self, individual):
        test_individual = optproblems.base.Individual(individual)
        self.testfunction.evaluate(test_individual)
        return test_individual.objective_values

    def mutate(self, individual):
        for i in range(self.num_dimensions):
            prob = np.random.uniform(0,1)
            if prob < self.mutation:
                change = np.random.uniform(-5,5)
                individual[i] = individual[i]+change
        return list(np.clip(individual, self.bounds[0], self.bounds[1]))
            
        
    def run(self):

        best = ([],10000000000)
        
        for x in range(self.maxIterations):
            for i in range(self.popSize):
                error = self.eval_fitness(self.population[i][0])
                self.population[i] = (self.population[i][0],error)
                if best[1] > error:
                    best = (self.population[i][0],error)

            new_pop = []
            for i in range(int(self.popSize/2)):
                par1 = self.select_p_tournament()
                par2 = self.select_p_tournament()

                child1, child2 = self.two_point_crossover(par1[0], par2[0])

                mchild1 = self.mutate(child1)
                mchild2 = self.mutate(child2)
                new_pop.append((mchild1, 1000000000))
                new_pop.append((mchild2, 1000000000))
            self.population = new_pop

        print("fitness", best[1])
        print("solution", best[0])

#Main Method

NO_Dimensions = 10
BOUNDS = (-5,5)
Population = 50  # Population size
generation_max = 150

benchmark_f1 = optproblems.cec2005.F6(NO_Dimensions)
algorithm1 = GA_Class(benchmark_f1, NO_Dimensions, BOUNDS, Population, generation_max)
algorithm1.run()

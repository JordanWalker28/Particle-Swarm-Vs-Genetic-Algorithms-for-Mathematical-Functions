#########################################################
# References                                           #
# Lones, M 2018, Lecture 7: Particle Swarm Optimization, lecture slides, F20BC, Heriot Watt University, Delivered Week 7

#Turing Finance. (2018). Portfolio Optimization using Particle Swarm Optimization. [online] Available at: http://www.turingfinance.com/portfolio-optimization-using-particle-swarm-optimization/ [Accessed 26 Nov. 2018].
#Luke, S. (2013). Essentials of metaheuristics. Lulu.com.#

#Rooy, N. (2018). Particle Swarm Optimization from Scratch with Python. [online] nathanrooy.github.io. Available at: https://nathanrooy.github.io/posts/2016-08-17/simple-particle-swarm-optimization-with-python/ [Accessed 26 Nov. 2018].

#########################################################




import optproblems
import optproblems.cec2005
from numpy import array
import numpy
import random

class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.pos_best = position
        self.pos_best_i= position          # best position individual
        self.err_best = 10000000000
        self.err_best_i=10000000000          # best error individual
        self.err=10000000000              # error individual
        
class PSO_Class():

    def __init__(self, test_function, num_particles, num_dimensions, bounds):
        self.testfunction = test_function
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.num_informants = 20
        self.bounds = bounds
        self.alpha = 0.85
        self.beta = 1.2
        self.gamma = 1
        self.delta = 1
        self.jumpsize = 1
        self.swarm = []
        
        #initalise random swarm
        for i in range(0,self.num_particles):
            self.swarm.append(Particle(numpy.random.uniform(low=self.bounds[0], high=self.bounds[1], size=(self.num_dimensions,)), numpy.random.uniform(low=self.bounds[0], high=self.bounds[1], size=(self.num_dimensions,))))

    def eval_error(self, individual_vector):
        test_individual = optproblems.base.Individual(individual_vector)
        self.testfunction.evaluate(test_individual)
        return test_individual.objective_values

    def run(self, maxIterations):

        best = Particle([],[])

        for x in range(maxIterations):
            #assess error
            for i in range(self.num_particles):
                fitness = self.eval_error(self.swarm[i].position)
                self.swarm[i].err=float(fitness)
                if(fitness<best.err):
                    best = Particle(self.swarm[i].position,self.swarm[i].velocity)
                    best.err = fitness
                if self.swarm[i].err_best > fitness:
                    self.swarm[i].err_best = float(fitness)
                    self.swarm[i].pos_best = list(self.swarm[i].position)

            #assess informants
            for i in range(self.num_particles):
                for j in range(self.num_informants):
                    informant = numpy.random.randint(low=0, high=self.num_particles)
                    if self.swarm[informant].err_best < self.swarm[i].err_best_i:
                        self.swarm[i].err_best_i = self.swarm[informant].err_best
                        self.swarm[i].pos_best_i = self.swarm[informant].pos_best

            #update velocity
            for i in range(self.num_particles):
                for j in range(self.num_dimensions):
                    b = numpy.random.uniform(low=0, high=self.beta)
                    c = numpy.random.uniform(low=0, high=self.gamma)
                    self.swarm[i].velocity[j] = (self.alpha * self.swarm[i].velocity[j]) + (b * (self.swarm[i].pos_best[j] - self.swarm[i].position[j])) + (c * (self.swarm[i].pos_best_i[j] - self.swarm[i].position[j]))

            for i in range(self.num_particles):
                 temp = numpy.add(self.swarm[i].position, self.swarm[i].velocity)
                 self.swarm[i].position = numpy.clip(temp, self.bounds[0], self.bounds[1])

        print("best: ", best.err)
        print("best solution:", best.position)

#Main Method

NO_DIMS = 10
NUM_PARTICLES = 100
BOUNDS=(-100,100)

benchmark_f1 = optproblems.cec2005.F5(NO_DIMS)
algorithm1 = PSO_Class(benchmark_f1, NUM_PARTICLES, NO_DIMS, BOUNDS)
algorithm1.run(100)

import numpy as np
import random
import matplotlib.pyplot as plt

''' define the objective function '''


def objective_function(reg, lux):
    return 4 * lux + 3 * reg


''' define constraints '''


def check_leather(reg, lux):
    return reg + lux <= 40


def check_hours(reg, lux):
    return reg + 2 * lux <= 60


''' Genetic algorithm'''


def genetic_algorithm(population_size, number_of_generations):
    """ Initialization """
    population = []
    fitness_history = []

    for i in range(population_size):
        reg = random.uniform(0, 60)
        lux = random.uniform(0, 40)
        population.append((reg, lux))

    best_solution = None
    best_fitness = float('-inf')

    for generation in range(number_of_generations):

        """ Evaluation """
        fitness = [objective_function(reg, lux) for reg, lux in population]

        feasible_population = [individual for individual in population
                               if check_leather(individual[0], individual[1])
                               and check_hours(individual[0], individual[1])]

        """ Selection """
        if feasible_population:
            feasible_fitness = [objective_function(reg, lux) for reg, lux in feasible_population]
            parents = random.choices(feasible_population, weights=feasible_fitness, k=population_size)
        else:
            parents = []
            while len(parents) < population_size:
                potential_parents = random.choice(population, weights=fitness)
                if not (not check_leather(potential_parents[0], potential_parents[1]) or not check_hours(
                        potential_parents[0], potential_parents[1])):
                    parents.append(potential_parents)

        """ Crossover """
        offspring = []
        for i in range(population_size):
            parent1, parent2 = random.choices(parents, k=2)
            reg_child = random.uniform(min(parent1[0], parent2[0]), max(parent1[0], parent2[0]))
            lux_child = random.uniform(min(parent1[1], parent2[1]), max(parent1[1], parent2[1]))
            offspring.append((reg_child, lux_child))


        """ Mutation """
        # mutation_rate = 1/(generation+1)
        mutation_rate = 0.1
        for i in range(population_size):
            if random.random()<mutation_rate:
                offspring[i] = (random.uniform(0,30), random.uniform(0, 60))


        """ Elitism """
        if best_solution is not None:
            offspring[0] = best_solution

        population = offspring

        feasible_solutions = [(reg, lux) for(reg, lux) in population if check_leather(reg, lux) and check_hours(reg, lux)]
        if feasible_solutions:
            best_solution = max(feasible_solutions, key=lambda x: objective_function(x[0], x[1]))
            best_fitness = objective_function(best_solution[0], best_solution[1])
        fitness_history.append(best_fitness)

        print(f"Generation {generation+1}: Best solution = {best_solution}; Best fitness = {best_fitness}")


    """ Plot the progress of fitness """
    plt.plot(range(1, number_of_generations+1), fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Genetic Algorithm - Fitness Progress")
    plt.show()

    return best_solution, best_fitness

population_size = 4
number_of_generations = 100

best_solution, best_fitness = genetic_algorithm(population_size, number_of_generations)

if best_solution is not None:
    print(f"Final best solution: {best_solution}")
    print(f"Final best fitness: {best_fitness}")
else:
    print("No feasible solution found within the constraints")
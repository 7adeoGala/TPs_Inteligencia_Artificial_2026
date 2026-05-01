
import random
import numpy as np
import math
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt


# Maximizamos la cantidad de cláusulas satisfechas
if "FitnessMax" in creator.__dict__:
    del creator.FitnessMax
if "Individual" in creator.__dict__:
    del creator.Individual


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


# Fórmula a: (x0 v x1)(x0 v ¬x1)(¬x0 v x1)(¬x0 v ¬x1)
def evaluar_sat_a(individual):


    x0 = bool(individual[0])
    x1 = bool(individual[1])
   
    c1 = x0 or x1
    c2 = x0 or not x1
    c3 = not x0 or x1
    c4 = not x0 or not x1
   
    clausulas_satisfechas = sum([c1, c2, c3, c4])
    return clausulas_satisfechas,


def resolver_instancia_a():
    toolbox_a = base.Toolbox()
    toolbox_a.register("attr_bool", random.randint, 0, 1)
   
    toolbox_a.register("individual", tools.initRepeat, creator.Individual, toolbox_a.attr_bool, 2)
    toolbox_a.register("population", tools.initRepeat, list, toolbox_a.individual)
   
    toolbox_a.register("evaluate", evaluar_sat_a)
    toolbox_a.register("mate", tools.cxTwoPoint)
    toolbox_a.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox_a.register("select", tools.selTournament, tournsize=3)


    pop = toolbox_a.population(n=10)
    hof = tools.HallOfFame(1)
   
    algorithms.eaSimple(pop, toolbox_a, cxpb=0.8, mutpb=0.2, ngen=20, halloffame=hof, verbose=False)
   
    mejor = hof[0]
    print("--- RESULTADOS INSTANCIA A ---")
    print(f"Mejor asignación [x0, x1]: {mejor}")
    print(f"Cláusulas satisfechas: {evaluar_sat_a(mejor)[0]} de 4 posibles.\n")


# Fórmula b: (x4 v x2 v x3)(x5 v x1 v x2)(x4 v x1 v ¬x3)(x3 v x1 v x2)(x4 v x1 v ¬x2)(¬x5 v ¬x1 v x4)


def evaluar_sat_b(individual):
   
    x1 = bool(individual[0])
    x2 = bool(individual[1])
    x3 = bool(individual[2])
    x4 = bool(individual[3])
    x5 = bool(individual[4])
   
    c1 = x4 or x2 or x3
    c2 = x5 or x1 or x2
    c3 = x4 or x1 or not x3
    c4 = x3 or x1 or x2
    c5 = x4 or x1 or not x2
    c6 = not x5 or not x1 or x4
   
    clausulas_satisfechas = sum([c1, c2, c3, c4, c5, c6])
    return clausulas_satisfechas,


def resolver_instancia_b():
    toolbox_b = base.Toolbox()
    toolbox_b.register("attr_bool", random.randint, 0, 1)
   
    toolbox_b.register("individual", tools.initRepeat, creator.Individual, toolbox_b.attr_bool, 5)
    toolbox_b.register("population", tools.initRepeat, list, toolbox_b.individual)
   
    toolbox_b.register("evaluate", evaluar_sat_b)
    toolbox_b.register("mate", tools.cxTwoPoint)
    toolbox_b.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox_b.register("select", tools.selTournament, tournsize=3)


    pop = toolbox_b.population(n=20)
    hof = tools.HallOfFame(1)
   
    algorithms.eaSimple(pop, toolbox_b, cxpb=0.8, mutpb=0.2, ngen=30, halloffame=hof, verbose=False)
   
    mejor = hof[0]
    print("--- RESULTADOS INSTANCIA B ---")
    print(f"Mejor asignación [x1, x2, x3, x4, x5]: {mejor}")
    print(f"Cláusulas satisfechas: {evaluar_sat_b(mejor)[0]} de 6 posibles.")


# Ejecutamos ambas resoluciones
resolver_instancia_a()
resolver_instancia_b()


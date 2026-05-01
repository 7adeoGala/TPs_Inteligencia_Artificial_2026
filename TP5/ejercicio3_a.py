import numpy as np

def sphere(x):
    return np.sum(x**2)

def schwefel(x):
    V = 4189.829101
    return V + np.sum(-x * np.sin(np.sqrt(np.abs(x))))

def griewank(x): 
    sum_term = np.sum(x**2) / 4000.0
    # np.arange(1, len(x) + 1) crea el array [1, 2, ..., n] para el divisor del coseno
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1 + sum_term - prod_term

def simple_genetic_algorithm(objective_fn, bounds, pop_size=100, generations=500, mutation_rate=0.1, crossover_rate=0.8):
    dimensions = len(bounds)

    # Inicializar la población aleatoria dentro de los límites
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    population = np.random.uniform(low=lows, high=highs, size=(pop_size, dimensions))
    
    best_solution = None
    best_fitness = float('inf')
    
    for gen in range(generations):
        # Evaluar la aptitud de cada individuo de la población
        fitness = np.array([objective_fn(ind) for ind in population])
        
        # Encontrar al mejor individuo de la generación actual y compararlo con el mejor histórico
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_solution = population[current_best_idx].copy()
            
        new_population = []
        for _ in range(pop_size):
            # Elegimos 3 individuos al azar y nos quedamos con el mejor
            tour1 = np.random.choice(pop_size, 3, replace=False)
            parent1 = population[tour1[np.argmin(fitness[tour1])]]
            
            tour2 = np.random.choice(pop_size, 3, replace=False)
            parent2 = population[tour2[np.argmin(fitness[tour2])]]
            
            # Combinar dos padres para crear un hijo
            child = parent1.copy()
            if np.random.rand() < crossover_rate:
                alpha = np.random.rand() # Factor de peso aleatorio entre 0 y 1
                child = alpha * parent1 + (1 - alpha) * parent2
                
            # Alteramos los genes del hijo
            for d in range(dimensions):
                if np.random.rand() < mutation_rate:
                    # El tamaño de paso será del 5% del rango total de los límites
                    scale = (bounds[d][1] - bounds[d][0]) * 0.05
                    child[d] += np.random.normal(0, scale)
                    
                    # Asegurar que el gen mutado no se salga de los límites
                    child[d] = np.clip(child[d], bounds[d][0], bounds[d][1])
            
            new_population.append(child)
            
        # Reemplazamos la vieja población por la nueva
        population = np.array(new_population)
        
    return best_solution, best_fitness


if __name__ == "__main__":
    np.random.seed(42) # Fijar semilla para que los resultados sean reproducibles

    # Función Sphere 
    bounds_sphere = [(-5.0, 5.0)] * 2
    best_x_sphere, min_sphere = simple_genetic_algorithm(
        sphere, bounds=bounds_sphere, pop_size=50, generations=200
    )
    print(" Sphere ")
    print(f"Mejor vector X encontrado: {best_x_sphere}")
    print(f"Minimo alcanzado: {min_sphere:.6f}  (Minimo real esperado: 0)\n")

    # Función Schwefel 
    bounds_schwefel = [(-500.0, 500.0)] * 10
    best_x_schwefel, min_schwefel = simple_genetic_algorithm(
        schwefel, bounds=bounds_schwefel, pop_size=200, generations=500
    )
    print(" Schwefel ")
    print(f"Mejor vector X encontrado: \n{best_x_schwefel}")
    print(f"Minimo alcanzado: {min_schwefel:.6f}  (Minimo real esperado: 0)\n")

    # Función Griewank 
    bounds_griewank = [(-600.0, 600.0)] * 10
    best_x_griewank, min_griewank = simple_genetic_algorithm(
        griewank, bounds=bounds_griewank, pop_size=200, generations=500
    )
    print(" Griewank ")
    print(f"Mejor vector X encontrado: \n{best_x_griewank}")
    print(f"Minimo alcanzado: {min_griewank:.6f}  (Minimo real esperado: 0)\n")

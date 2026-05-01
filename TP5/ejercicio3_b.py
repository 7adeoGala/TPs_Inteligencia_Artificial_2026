import numpy as np
import time

# FUNCIONES OBJETIVO

def sphere(x):
    return np.sum(x**2)

def schwefel(x):
    V = 4189.829101
    return V + np.sum(-x * np.sin(np.sqrt(np.abs(x))))

def griewank(x): 
    sum_term = np.sum(x**2) / 4000.0
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1 + sum_term - prod_term


# AGS

def simple_genetic_algorithm(objective_fn, bounds, pop_size=100, generations=500, mutation_rate=0.1, crossover_rate=0.8):
    dimensions = len(bounds)
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    population = np.random.uniform(low=lows, high=highs, size=(pop_size, dimensions))
    
    best_solution = None
    best_fitness = float('inf')
    
    for gen in range(generations):
        fitness = np.array([objective_fn(ind) for ind in population])
        
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_solution = population[current_best_idx].copy()
            
        new_population = []
        for _ in range(pop_size):
            tour1 = np.random.choice(pop_size, 3, replace=False)
            parent1 = population[tour1[np.argmin(fitness[tour1])]]
            
            tour2 = np.random.choice(pop_size, 3, replace=False)
            parent2 = population[tour2[np.argmin(fitness[tour2])]]
            
            child = parent1.copy()
            if np.random.rand() < crossover_rate:
                alpha = np.random.rand()
                child = alpha * parent1 + (1 - alpha) * parent2
                
            for d in range(dimensions):
                if np.random.rand() < mutation_rate:
                    scale = (bounds[d][1] - bounds[d][0]) * 0.05
                    child[d] += np.random.normal(0, scale)
                    child[d] = np.clip(child[d], bounds[d][0], bounds[d][1])
            new_population.append(child)
            
        population = np.array(new_population)
        
    return best_solution, best_fitness


# PSO

def particle_swarm_optimization(objective_fn, bounds, num_particles=100, max_iter=500):
    dimensions = len(bounds)
    w_i = 0.729
    w_c = 1.494
    w_s = 1.494
    
    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])
    
    # Inicializar posiciones y velocidades de las partículas aleatoriamente 
    positions = np.random.uniform(low=lows, high=highs, size=(num_particles, dimensions))

    velocities = np.random.uniform(low=-abs(highs-lows)*0.1, high=abs(highs-lows)*0.1, size=(num_particles, dimensions))
    
    # Mejores personales (pbest) inician siendo la posición inicial de cada partícula
    pbest_positions = positions.copy()
    pbest_scores = np.array([objective_fn(p) for p in positions])
    
    # Mejor global (gbest) inicia como el mejor de la población inicial
    gbest_idx = np.argmin(pbest_scores)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_score = pbest_scores[gbest_idx]
    
    # Ciclo Actualización
    for _ in range(max_iter):
        # Componentes estocásticos r1 y r2
        r1 = np.random.rand(num_particles, dimensions)
        r2 = np.random.rand(num_particles, dimensions)
        
        # Ecuación de actualización de la velocidad
        velocities = (w_i * velocities + 
                      w_c * r1 * (pbest_positions - positions) + 
                      w_s * r2 * (gbest_position - positions))
        
        # Ecuación de actualización de la posición
        positions = positions + velocities
        
        # Aplicamos los límites del espacio de búsqueda a las posiciones
        for d in range(dimensions):
            positions[:, d] = np.clip(positions[:, d], lows[d], highs[d])
            
        scores = np.array([objective_fn(p) for p in positions])
        
        # Actualizar mejores personales y el global
        for i in range(num_particles):
            if scores[i] < pbest_scores[i]:
                pbest_scores[i] = scores[i]
                pbest_positions[i] = positions[i].copy()
                
                # Verificamos si también mejora a todo el enjambre
                if scores[i] < gbest_score:
                    gbest_score = scores[i]
                    gbest_position = positions[i].copy()
                    
    return gbest_position, gbest_score


# BLOQUE DE COMPARACIÓN

if __name__ == "__main__":
    def run_comparison(name, fn, bounds, pop_size_val=100, iter_val=500):
        print(f" FUNCION: {name}")
        
        # Ejecutar AGS
        np.random.seed(42)  # Misma semilla para tener condiciones justas de inicio aleatorio
        start_ags = time.time()
        best_x_ags, min_ags = simple_genetic_algorithm(fn, bounds, pop_size=pop_size_val, generations=iter_val)
        time_ags = time.time() - start_ags
        
        # Ejecutar PSO
        np.random.seed(42)
        start_pso = time.time()
        best_x_pso, min_pso = particle_swarm_optimization(fn, bounds, num_particles=pop_size_val, max_iter=iter_val)
        time_pso = time.time() - start_pso
        
        # Mostrar la métrica del mínimo alcanzado y tiempo tardado
        print("[ RESULTADOS AGS ]")
        print(f"  > Minimo Encontrado : {min_ags:.6f}")
        print(f"  > Tiempo: {time_ags:.4f} seg")
        print(f"  > Mejor Valor X     : {best_x_ags[:3]}... (mostrando primeros 3)" if len(best_x_ags) > 3 else f"  > Mejor Valor X : {best_x_ags}")
        print()
        
        print("[ RESULTADOS PSO ]")
        print(f"  > Minimo Encontrado : {min_pso:.6f}")
        print(f"  > Tiempo: {time_pso:.4f} seg")
        print(f"  > Mejor Valor X     : {best_x_pso[:3]}... (mostrando primeros 3)" if len(best_x_pso) > 3 else f"  > Mejor Valor X : {best_x_pso}")
        print()
        
        # Conclusion
        if min_pso < min_ags:
            print("=> CONCLUSION: PSO obtuvo un mejor resultado (un minimo más cercano a 0).")
        elif min_ags < min_pso:
            print("=> CONCLUSION: AGS obtuvo un mejor resultado (un minimo más cercano a 0).")
        else:
            print("=> CONCLUSION: Ambos algoritmos se aproximaron identicamente.")
            
        print("\n")


    # Sphere
    bounds_sphere = [(-5.0, 5.0)] * 2
    run_comparison("SPHERE", sphere, bounds_sphere, pop_size_val=50, iter_val=200)

    # Schwefel
    bounds_schwefel = [(-500.0, 500.0)] * 10
    run_comparison("SCHWEFEL", schwefel, bounds_schwefel, pop_size_val=200, iter_val=500)

    # Griewank
    bounds_griewank = [(-600.0, 600.0)] * 10
    run_comparison("GRIEWANK", griewank, bounds_griewank, pop_size_val=200, iter_val=500)

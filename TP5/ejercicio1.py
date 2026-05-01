import random
import os
import sys
import matplotlib.pyplot as plt


def f(x):
    return x**5 - x**3 - 2 * (x**2)


def decode(chromosome):
    return int("".join(str(bit) for bit in chromosome), 2)


def generate_population(size, chrom_length):
    return [[random.randint(0, 1) for _ in range(chrom_length)] for _ in range(size)]


def roulette_wheel_selection(population, fitnesses):
    min_fit = min(fitnesses)
    if min_fit < 0:
        shifted_fitness = [fit - min_fit + 1 for fit in fitnesses]
    else:
        if sum(fitnesses) == 0:
            shifted_fitness = [1 for _ in fitnesses]
        else:
            shifted_fitness = fitnesses
           
    total_fitness = sum(shifted_fitness)
    pick = random.uniform(0, total_fitness)
    current = 0
    for i, ind in enumerate(population):
        current += shifted_fitness[i]
        if current >= pick:
            return ind
    return population[-1]


def crossover(parent1, parent2, p_c=0.8):
    if random.random() < p_c:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1[:], parent2[:]


def mutate(chromosome, p_m=0.05):
    for i in range(len(chromosome)):
        if random.random() < p_m:
            chromosome[i] = 1 - chromosome[i]
    return chromosome


def plot_results(best_fitness_history, avg_fitness_history, best_overall_individual, best_overall_fitness):
    script_path = os.path.abspath(sys.argv[0])
    base_name = os.path.splitext(os.path.basename(script_path))[0]
    out_dir = os.path.join(os.path.dirname(script_path), base_name)
   
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    plt.figure(figsize=(10, 6))
    generations = range(1, len(best_fitness_history) + 1)
    plt.plot(generations, best_fitness_history, label='Mejor Fitness', color='#3498db', linewidth=2)
    plt.plot(generations, avg_fitness_history, label='Fitness Promedio', color='#f39c12', linestyle='--')
    plt.title('Evolución de la Función de Adaptación (Fitness)')
    plt.xlabel('Generación')
    plt.ylabel('Fitness f(x)')
    plt.legend()
    plt.grid(True)
   
    img1_path = os.path.join(out_dir, f"{base_name}_Evolucion_Fitness.jpg")
    plt.savefig(img1_path, format='jpg', dpi=300)
    plt.close()
   
    plt.figure(figsize=(10, 6))
    x_vals = list(range(64))
    y_vals = [f(x) for x in x_vals]
    best_x = decode(best_overall_individual)
   
    plt.plot(x_vals, y_vals, label='f(x) = x^5 - x^3 - 2x^2', color='#2ecc71', linewidth=2)
    plt.plot(best_x, best_overall_fitness, 'ro', markersize=10, label=f'Óptimo hallado (x={best_x})')
    plt.title('Espacio de Búsqueda y Solución Encontrada')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
   
    img2_path = os.path.join(out_dir, f"{base_name}_Espacio_Busqueda.jpg")
    plt.savefig(img2_path, format='jpg', dpi=300)
    plt.close()


    print(f"\n[+] Se guardaron exitosamente las imágenes en el directorio: {out_dir}")
    print(f"    - {os.path.basename(img1_path)}")
    print(f"    - {os.path.basename(img2_path)}")


def run_ags(pop_size=30, generations=100, p_c=0.85, p_m=0.05):
    """ Ejecuta el Algoritmo Genético Simple (AGS) """
    chrom_length = 6
   
    population = generate_population(pop_size, chrom_length)
   
    best_overall_individual = None
    best_overall_fitness = float('-inf')
   
    best_fitness_history = []
    avg_fitness_history = []
   
    print("Iniciando Algoritmo Genético Simple...\n")
   
    for generation in range(generations):
        fitnesses = [f(decode(ind)) for ind in population]
       
        best_gen_fitness = max(fitnesses)
        avg_gen_fitness = sum(fitnesses) / len(fitnesses)
       
        best_gen_idx = fitnesses.index(best_gen_fitness)
        best_gen_ind = population[best_gen_idx]
       
        best_fitness_history.append(best_gen_fitness)
        avg_fitness_history.append(avg_gen_fitness)
       
        if best_gen_fitness > best_overall_fitness:
            best_overall_fitness = best_gen_fitness
            best_overall_individual = best_gen_ind
           
        if (generation + 1) % 10 == 0 or generation == 0:
            print(f"Generación {generation + 1:2d} | Mejor x = {decode(best_gen_ind):2d} | Fitness = {best_gen_fitness}")
       
        new_population = []
       
        while len(new_population) < pop_size:
            p1 = roulette_wheel_selection(population, fitnesses)
            p2 = roulette_wheel_selection(population, fitnesses)
           
            c1, c2 = crossover(p1, p2, p_c)
           
            c1 = mutate(c1, p_m)
            c2 = mutate(c2, p_m)
           
            new_population.extend([c1, c2])
           
        population = new_population[:pop_size]


    best_x = decode(best_overall_individual)
    print("\n" + "="*30)
    print("        RESULTADO FINAL")
    print("="*30)
    print(f"Mejor cromosoma : {best_overall_individual}")
    print(f"Mejor valor (x) : {best_x}")
    print(f"Mejor f(x)      : {best_overall_fitness}")
    print("="*30)
   
    plot_results(best_fitness_history, avg_fitness_history, best_overall_individual, best_overall_fitness)


if __name__ == "__main__":
    run_ags(pop_size=30, generations=100, p_c=0.85, p_m=0.05)


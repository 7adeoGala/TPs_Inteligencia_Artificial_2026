import random

# Definición de los estados y sus adyacencias
ESTADOS = [
    "Schleswig-Holstein", "Hamburg", "Bremen", "Niedersachsen", 
    "Mecklenburg-Vorpommern", "Brandenburg", "Berlin", "Sachsen-Anhalt", 
    "Sachsen", "Thüringen", "Hessen", "Nordrhein-Westfalen", 
    "Rheinland-Pfalz", "Saarland", "Baden-Württemberg", "Bayern"
]

NUM_ESTADOS = len(ESTADOS)
COLORES = [0, 1, 2, 3] 

# Matriz de adyacencia
ADYACENCIAS = [
    (0, 1), (0, 3), (0, 4),                     # Schleswig-Holstein
    (1, 3),                                     # Hamburg
    (2, 3),                                     # Bremen
    (3, 4), (3, 5), (3, 7), (3, 9), (3, 10), (3, 11), # Niedersachsen
    (4, 5),                                     # Mecklenburg-Vorpommern
    (5, 6), (5, 7), (5, 8),                     # Brandenburg
    # (6) Berlin está contenido en Brandenburg (adyacencia ya definida)
    (7, 8), (7, 9),                             # Sachsen-Anhalt
    (8, 9), (8, 15),                            # Sachsen
    (9, 10), (9, 15),                           # Thüringen
    (10, 11), (10, 12), (10, 14), (10, 15),     # Hessen
    (11, 12),                                   # Nordrhein-Westfalen
    (12, 13), (12, 14),                         # Rheinland-Pfalz
    # (13) Saarland ya conectada con RP
    (14, 15)                                    # Baden-Württemberg
]

def calcular_conflictos(individuo):

#    Función de aptitud (fitness):
 
    conflictos = 0
    for estado_a, estado_b in ADYACENCIAS:
        if individuo[estado_a] == individuo[estado_b]:
            conflictos += 1
    return conflictos

def seleccion_torneo(poblacion, aptitudes, k=3):
    #Selecciona al mejor individuo de un grupo de k individuos elegidos al azar.
    
    seleccionados = random.sample(range(len(poblacion)), k)
    mejor_idx = seleccionados[0]
    for idx in seleccionados[1:]:
        if aptitudes[idx] < aptitudes[mejor_idx]:
            mejor_idx = idx
    return poblacion[mejor_idx]

def cruzar(padre1, padre2):
    #Cruzamiento de un punto (Single-point crossover).
    
    punto_corte = random.randint(1, NUM_ESTADOS - 1)
    hijo = padre1[:punto_corte] + padre2[punto_corte:]
    return hijo

def mutar(individuo, tasa_mutacion=0.1):
    #Mutación de reemplazo aleatorio.
    
    for i in range(NUM_ESTADOS):
        if random.random() < tasa_mutacion:
            individuo[i] = random.choice(COLORES)
    return individuo

def algoritmo_genetico_mapa(pop_size=100, generaciones=500, tasa_mutacion=0.1, tasa_cruzamiento=0.8):
    # Inicializar población aleatoria
    poblacion = [[random.choice(COLORES) for _ in range(NUM_ESTADOS)] for _ in range(pop_size)]
    
    mejor_solucion = None
    mejor_fitness = float('inf')

    for gen in range(generaciones):
        aptitudes = [calcular_conflictos(ind) for ind in poblacion]
        
        # Guardar el mejor de la generación actual
        mejor_idx_gen = aptitudes.index(min(aptitudes))
        if aptitudes[mejor_idx_gen] < mejor_fitness:
            mejor_fitness = aptitudes[mejor_idx_gen]
            mejor_solucion = poblacion[mejor_idx_gen][:]
            
        # Si llegamos a 0 conflictos, encontramos la solución perfecta
        if mejor_fitness == 0:
            print(f"Solución perfecta encontrada en la generación {gen}!")
            break

        nueva_poblacion = []
        # Conservamos al mejor de la generación anterior directamente
        nueva_poblacion.append(mejor_solucion[:])
        
        while len(nueva_poblacion) < pop_size:
            padre1 = seleccion_torneo(poblacion, aptitudes)
            padre2 = seleccion_torneo(poblacion, aptitudes)
            
            if random.random() < tasa_cruzamiento:
                hijo = cruzar(padre1, padre2)
            else:
                hijo = padre1[:]
                
            hijo = mutar(hijo, tasa_mutacion)
            nueva_poblacion.append(hijo)
            
        poblacion = nueva_poblacion

    return mejor_solucion, mejor_fitness

if __name__ == "__main__":
    random.seed(42)
    
    mejor_ind, min_conflictos = algoritmo_genetico_mapa(pop_size=100, generaciones=1000)
    
    print("\nResultados del Algoritmo Genético")
    print(f"Mejor individuo (colores asignados): {mejor_ind}")
    print(f"Cantidad de conflictos: {min_conflictos}")
    print("\nAsignación por estado:")
    
    nombres_colores = {0: "Rojo", 1: "Verde", 2: "Azul", 3: "Amarillo"}
    for idx, estado in enumerate(ESTADOS):
        color_idx = mejor_ind[idx]
        print(f"{estado}: {nombres_colores[color_idx]}")

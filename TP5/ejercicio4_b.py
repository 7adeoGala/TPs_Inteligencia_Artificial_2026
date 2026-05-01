import random

# Definición de los estados y sus adyacencias
ESTADOS = [
    "Schleswig-Holstein", "Hamburg", "Bremen", "Niedersachsen", 
    "Mecklenburg-Vorpommern", "Brandenburg", "Berlin", "Sachsen-Anhalt", 
    "Sachsen", "Thüringen", "Hessen", "Nordrhein-Westfalen", 
    "Rheinland-Pfalz", "Saarland", "Baden-Württemberg", "Bayern"
]

NUM_ESTADOS = len(ESTADOS)
COLORES = [0, 1, 2, 3] # 4 colores

ADYACENCIAS = [
    (0, 1), (0, 3), (0, 4),
    (1, 3),
    (2, 3),
    (3, 4), (3, 5), (3, 7), (3, 9), (3, 10), (3, 11),
    (4, 5),
    (5, 6), (5, 7), (5, 8),
    (7, 8), (7, 9),
    (8, 9), (8, 15),
    (9, 10), (9, 15),
    (10, 11), (10, 12), (10, 14), (10, 15),
    (11, 12),
    (12, 13), (12, 14),
    (14, 15)
]

def calcular_conflictos(individuo):
    conflictos = 0
    for estado_a, estado_b in ADYACENCIAS:
        if individuo[estado_a] == individuo[estado_b]:
            conflictos += 1
    return conflictos

def obtener_estados_en_conflicto(individuo):
    en_conflicto = set()
    for estado_a, estado_b in ADYACENCIAS:
        if individuo[estado_a] == individuo[estado_b]:
            en_conflicto.add(estado_a)
            en_conflicto.add(estado_b)
    return list(en_conflicto)

def busqueda_tabu(max_iteraciones=1000, tamano_tabu=10):
    # Generar solución inicial aleatoria
    solucion_actual = [random.choice(COLORES) for _ in range(NUM_ESTADOS)]
    mejor_solucion = solucion_actual[:]
    
    costo_actual = calcular_conflictos(solucion_actual)
    mejor_costo = costo_actual
    
    lista_tabu = [] # Almacena movimientos prohibidos como tuplas: (estado, color_prohibido)
    
    for iteracion in range(max_iteraciones):
        if mejor_costo == 0:
            print(f"Solución perfecta encontrada en la iteración {iteracion}!")
            break
            
        estados_conflictivos = obtener_estados_en_conflicto(solucion_actual)
        
        mejor_vecino = None
        mejor_costo_vecino = float('inf')
        movimiento_a_realizar = None # (estado, color_nuevo)
        color_anterior_reemplazado = None
        
        # Generar vecindario: cambiar el color solo de los estados en conflicto
        for estado in estados_conflictivos:
            color_viejo = solucion_actual[estado]
            for color_nuevo in COLORES:
                if color_nuevo != color_viejo:
                    # Crear el vecino y evaluarlo
                    vecino = solucion_actual[:]
                    vecino[estado] = color_nuevo
                    costo_vecino = calcular_conflictos(vecino)
                    
                    movimiento = (estado, color_nuevo)
                    es_tabu = movimiento in lista_tabu
                    
                    # si es tabú pero mejora el mejor histórico, lo permitimos
                    if es_tabu and costo_vecino < mejor_costo:
                        es_tabu = False
                        
                    if not es_tabu and costo_vecino < mejor_costo_vecino:
                        mejor_costo_vecino = costo_vecino
                        mejor_vecino = vecino
                        movimiento_a_realizar = movimiento
                        color_anterior_reemplazado = color_viejo
                        
        # Medida de rescate por si todos los vecinos generados son tabú 
        if mejor_vecino is None and estados_conflictivos:
            estado_random = random.choice(estados_conflictivos)
            color_random = random.choice([c for c in COLORES if c != solucion_actual[estado_random]])
            mejor_vecino = solucion_actual[:]
            mejor_vecino[estado_random] = color_random
            mejor_costo_vecino = calcular_conflictos(mejor_vecino)
            movimiento_a_realizar = (estado_random, color_random)
            color_anterior_reemplazado = solucion_actual[estado_random]

        # Realizar el movimiento
        solucion_actual = mejor_vecino
        costo_actual = mejor_costo_vecino
        
        # Actualizar el mejor global
        if costo_actual < mejor_costo:
            mejor_costo = costo_actual
            mejor_solucion = solucion_actual[:]
            
        # prohibimos regresar al color que le acabamos de quitar
        if color_anterior_reemplazado is not None:
            movimiento_prohibido = (movimiento_a_realizar[0], color_anterior_reemplazado)
            lista_tabu.append(movimiento_prohibido)
            
            if len(lista_tabu) > tamano_tabu:
                lista_tabu.pop(0)
                
    return mejor_solucion, mejor_costo

if __name__ == "__main__":
    random.seed(42)
    mejor_ind, min_conflictos = busqueda_tabu(max_iteraciones=1000, tamano_tabu=12)
    
    print("\nResultados de Búsqueda Tabú")
    print(f"Mejor individuo (colores asignados): {mejor_ind}")
    print(f"Cantidad de conflictos: {min_conflictos}")
    print("\nAsignación por estado:")
    
    nombres_colores = {0: "Rojo", 1: "Verde", 2: "Azul", 3: "Amarillo"}
    for idx, estado in enumerate(ESTADOS):
        color_idx = mejor_ind[idx]
        print(f"{estado}: {nombres_colores[color_idx]}")

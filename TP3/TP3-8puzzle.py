import time
from simpleai.search import (SearchProblem, breadth_first, uniform_cost,
                             limited_depth_first, iterative_limited_depth_first,
                             greedy, astar)

GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)

class EightPuzzleProblem(SearchProblem):
    def __init__(self, initial_state, heuristic_type="manhattan"):
        super().__init__(initial_state=initial_state)
        self.heuristic_type = heuristic_type

    def actions(self, state):
        empty_index = state.index(0)
        row = empty_index // 3
        col = empty_index % 3

        act = []
        if row > 0: act.append("Mover ARRIBA")
        if row < 2: act.append("Mover ABAJO")
        if col > 0: act.append("Mover IZQUIERDA")
        if col < 2: act.append("Mover DERECHA")
        return act

    def result(self, state, action):
        empty_index = state.index(0)
        state_list = list(state)

        if action == "Mover ARRIBA": swap_index = empty_index - 3
        elif action == "Mover ABAJO": swap_index = empty_index + 3
        elif action == "Mover IZQUIERDA": swap_index = empty_index - 1
        elif action == "Mover DERECHA": swap_index = empty_index + 1

        state_list[empty_index], state_list[swap_index] = state_list[swap_index], state_list[empty_index]
        return tuple(state_list)

    def is_goal(self, state):
        return state == GOAL

    def cost(self, state, action, state2):
        return 1

    def heuristic(self, state):
        if self.heuristic_type == "mal_colocadas":
            # Heurística: Cantidad de piezas fuera de su lugar (ignorando el 0)
            return sum(1 for i in range(9) if state[i] != GOAL[i] and state[i] != 0)
        else:
            # Heurística: Distancia de Manhattan
            h = 0
            for i in range(9):
                val = state[i]
                if val != 0:
                    target_idx = GOAL.index(val)
                    curr_row, curr_col = i // 3, i % 3
                    target_row, target_col = target_idx // 3, target_idx % 3
                    h += abs(curr_row - target_row) + abs(curr_col - target_col)
            return h

# --- CONFIGURACIÓN DEL LABORATORIO ---
estados_iniciales = [
    (1, 2, 3, 4, 0, 6, 7, 5, 8), # Estado 1
    (1, 0, 3, 4, 2, 5, 7, 8, 6), # Estado 2
    (1, 2, 3, 0, 4, 5, 7, 8, 6)  # Estado 3
]

algoritmos_no_informados = [
    ("Búsqueda en anchura", breadth_first, {}),
    ("Costo uniforme", uniform_cost, {}),
    ("Por profundidad limitada (5)", limited_depth_first, {"depth_limit": 5}),
    ("Por profundidad iterativa", iterative_limited_depth_first, {})
]

algoritmos_informados = [
    ("A* (Manhattan)", astar, "manhattan"),
    ("A* (Mal Colocadas)", astar, "mal_colocadas"),
    ("Voraz (Manhattan)", greedy, "manhattan"),
    ("Voraz (Mal Colocadas)", greedy, "mal_colocadas")
]

for idx, estado_ini in enumerate(estados_iniciales):
    print(f"\nEstado inicial {idx+1}: {estado_ini}")
    
    print("\nBúsquedas No Informadas:")
    for nombre, algoritmo, args in algoritmos_no_informados:
        problem = EightPuzzleProblem(initial_state=estado_ini)
        
        # Ejecutamos 3 veces para sacar un promedio
        tiempos = []
        for _ in range(3):
            t0 = time.perf_counter()
            result = algoritmo(problem, graph_search=True, **args)
            t1 = time.perf_counter()
            tiempos.append(t1 - t0)
            
        promedio_ms = (sum(tiempos) / len(tiempos)) * 1000
        pasos = len(result.path()) - 1 if result else "N/A"
        print(f"{nombre:30} -> {promedio_ms:.4f} ms | Pasos: {pasos}")

    print("\nBúsquedas Informadas:")
    for nombre, algoritmo, heuristica in algoritmos_informados:
        problem = EightPuzzleProblem(initial_state=estado_ini, heuristic_type=heuristica)
        
        tiempos = []
        for _ in range(3):
            t0 = time.perf_counter()
            result = algoritmo(problem, graph_search=True)
            t1 = time.perf_counter()
            tiempos.append(t1 - t0)
            
        promedio_ms = (sum(tiempos) / len(tiempos)) * 1000
        pasos = len(result.path()) - 1 if result else "N/A"
        print(f"{nombre:30} -> {promedio_ms:.4f} ms | Pasos: {pasos}")
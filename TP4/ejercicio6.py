import os
import time
from simpleai.search import SearchProblem, breadth_first, uniform_cost, depth_first, limited_depth_first, iterative_limited_depth_first, astar, greedy
GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)

class EightPuzzleProblem(SearchProblem):

    def __init__(self, heuristic_type='manhattan', **kwargs):
        super().__init__(**kwargs)
        self.heuristic_type = heuristic_type

    def heuristic(self, state):
        cost = 0
        if self.heuristic_type == 'misplaced':
            for i in range(9):
                if state[i] != 0 and state[i] != GOAL[i]:
                    cost += 1
        elif self.heuristic_type == 'manhattan':
            for i in range(9):
                val = state[i]
                if val != 0:
                    goal_idx = val - 1
                    curr_row, curr_col = divmod(i, 3)
                    goal_row, goal_col = divmod(goal_idx, 3)
                    cost += abs(curr_row - goal_row) + abs(curr_col - goal_col)
        return cost

    def actions(self, state):
        empty_index = state.index(0)
        row = empty_index // 3
        col = empty_index % 3
        act = []
        if row > 0:
            act.append('Mover ARRIBA')
        if row < 2:
            act.append('Mover ABAJO')
        if col > 0:
            act.append('Mover IZQUIERDA')
        if col < 2:
            act.append('Mover DERECHA')
        return act

    def result(self, state, action):
        empty_index = state.index(0)
        state_list = list(state)
        if action == 'Mover ARRIBA':
            swap_index = empty_index - 3
        elif action == 'Mover ABAJO':
            swap_index = empty_index + 3
        elif action == 'Mover IZQUIERDA':
            swap_index = empty_index - 1
        elif action == 'Mover DERECHA':
            swap_index = empty_index + 1
        state_list[empty_index], state_list[swap_index] = (state_list[swap_index], state_list[empty_index])
        return tuple(state_list)

    def is_goal(self, state):
        return state == GOAL

    def cost(self, state, action, state2):
        return 1
print('\n' + '=' * 50)
print(' RESULTADOS DE EJECUCIÓN (MODO BENCHMARK)')
print('=' * 50)
algoritmos = [('Búsqueda en anchura', breadth_first, {}, 'manhattan'), ('Costo uniforme', uniform_cost, {}, 'manhattan'), ('Por profundidad limitada (5)', limited_depth_first, {'depth_limit': 5}, 'manhattan'), ('Por profundidad iterativa', iterative_limited_depth_first, {}, 'manhattan'), ('A* con Manhattan', astar, {}, 'manhattan'), ('A* con Mal Colocadas', astar, {}, 'misplaced'), ('Avara con Manhattan', greedy, {}, 'manhattan'), ('Avara con Mal Colocadas', greedy, {}, 'misplaced')]
INITIAL_STATE = (1, 0, 3, 4, 2, 5, 7, 8, 6)
for nombre, algoritmo, args, h_type in algoritmos:
    problem = EightPuzzleProblem(initial_state=INITIAL_STATE, heuristic_type=h_type)
    tiempos = []
    for _ in range(3):
        t0 = time.perf_counter()
        result = algoritmo(problem, graph_search=True, **args)
        t1 = time.perf_counter()
        tiempos.append(t1 - t0)
    promedio_ms = sum(tiempos) / len(tiempos) * 1000
    camino = []
    if result:
        for nodo in result.path():
            estado_str = str(nodo[1]).replace(', ', ',')
            camino.append(estado_str)
        if len(camino) > 15:
            camino_str = ' -> '.join(camino[:5]) + f' ... ({len(camino) - 10} estados más) ... ' + ' -> '.join(camino[-5:])
        else:
            camino_str = ' -> '.join(camino)
    else:
        camino_str = 'No se encontró solución con este límite.'
    print(f'\n[{nombre}]')
    print(f'Tiempo promedio: {promedio_ms:.4f} ms')
    if result:
        print(f'Pasos: {len(result.path()) - 1}')
    print(f'Camino encontrado: {camino_str}')

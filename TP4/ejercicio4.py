import os
import time
from simpleai.search import SearchProblem, breadth_first, uniform_cost, depth_first, limited_depth_first, iterative_limited_depth_first
GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)

class EightPuzzleProblem(SearchProblem):

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
algoritmos = [('Búsqueda en anchura', breadth_first, {}), ('Costo uniforme', uniform_cost, {}), ('Por profundidad limitada (5)', limited_depth_first, {'depth_limit': 5}), ('Por profundidad limitada (10)', limited_depth_first, {'depth_limit': 10}), ('Por profundidad limitada (15)', limited_depth_first, {'depth_limit': 15}), ('Por profundidad iterativa', iterative_limited_depth_first, {})]
INITIAL_STATE = (1, 0, 3, 4, 2, 5, 7, 8, 6)
for nombre, algoritmo, args in algoritmos:
    problem = EightPuzzleProblem(initial_state=INITIAL_STATE)
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

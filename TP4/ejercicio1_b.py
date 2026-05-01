from simpleai.search import SearchProblem, breadth_first
from simpleai.search.viewers import WebViewer

class ProblemaArbolAnchura(SearchProblem):

    def __init__(self, initial_state=1, goal=11):
        super().__init__(initial_state=initial_state)
        self.goal = goal

    def actions(self, state):
        return ['rama_izquierda', 'rama_derecha']

    def result(self, state, action):
        if action == 'rama_izquierda':
            return 2 * state
        if action == 'rama_derecha':
            return 2 * state + 1

    def is_goal(self, state):
        return state == self.goal
if __name__ == '__main__':
    print('BÚSQUEDA PRIMERO EN ANCHURA')
    problema = ProblemaArbolAnchura(initial_state=1, goal=11)
    visor = WebViewer()
    print('Abre tu navegador en http://localhost:8000')
    resultado = breadth_first(problema, viewer=visor)
    if resultado:
        print(f'\nCamino encontrado: {resultado.path()}')
from simpleai.search import SearchProblem, limited_depth_first
from simpleai.search.viewers import WebViewer

class ProblemaArbolProfundidad(SearchProblem):

    def __init__(self, initial_state=1, goal=11):
        super().__init__(initial_state=initial_state)
        self.goal = goal

    def actions(self, state):
        return ['rama_derecha', 'rama_izquierda']

    def result(self, state, action):
        if action == 'rama_izquierda':
            return 2 * state
        if action == 'rama_derecha':
            return 2 * state + 1

    def is_goal(self, state):
        return state == self.goal
if __name__ == '__main__':
    print('BÚSQUEDA DE PROFUNDIDAD LIMITADA (LÍMITE 3)')
    problema = ProblemaArbolProfundidad(initial_state=1, goal=11)
    visor = WebViewer()
    print('Abre tu navegador en http://localhost:8000')
    resultado = limited_depth_first(problema, depth_limit=3, viewer=visor)
    if resultado:
        print(f'\nCamino encontrado: {resultado.path()}')
    else:
        print('\nNo se encontró el objetivo dentro del límite establecido.')
from simpleai.search import SearchProblem, iterative_limited_depth_first
from simpleai.search.viewers import WebViewer

class ProblemaArbolProfundidad(SearchProblem):

    def __init__(self, initial_state=1, goal=11):
        super().__init__(initial_state=initial_state)
        self.goal = goal

    def actions(self, state):
        return ['rama_derecha', 'rama_izquierda']

    def result(self, state, action):
        if action == 'rama_izquierda':
            return 2 * state
        if action == 'rama_derecha':
            return 2 * state + 1

    def is_goal(self, state):
        return state == self.goal
if __name__ == '__main__':
    print('BÚSQUEDA DE PROFUNDIDAD ITERATIVA')
    problema = ProblemaArbolProfundidad(initial_state=1, goal=11)
    visor = WebViewer()
    print('Abre tu navegador en http://localhost:8000')
    resultado = iterative_limited_depth_first(problema, viewer=visor)
    if resultado:
        print(f'\nCamino encontrado: {resultado.path()}')

def sucesores_adelante(n):
    return [2 * n, 2 * n + 1]

def sucesor_atras(n):
    return n // 2

def busqueda_bidireccional(inicio, objetivo):
    print(f'Búsqueda bidireccional: {inicio} -> {objetivo}\n')
    frontera_adelante = [inicio]
    frontera_atras = [objetivo]
    visitados_adelante = {inicio: None}
    visitados_atras = {objetivo: None}
    paso = 1
    while frontera_adelante and frontera_atras:
        print(f'Paso {paso}:')
        actual_adelante = frontera_adelante.pop(0)
        print(f'Adelante evalúa: {actual_adelante}')
        if actual_adelante in visitados_atras:
            print(f'\nLas búsquedas se cruzaron en el nodo {actual_adelante}')
            return reconstruir_camino(visitados_adelante, visitados_atras, actual_adelante)
        hijos = sucesores_adelante(actual_adelante)
        print(f'  -> Genera hacia adelante: {hijos}')
        for hijo in hijos:
            if hijo not in visitados_adelante:
                visitados_adelante[hijo] = actual_adelante
                frontera_adelante.append(hijo)
        actual_atras = frontera_atras.pop(0)
        print(f'Atrás evalúa: {actual_atras}')
        if actual_atras in visitados_adelante:
            print(f'\nLas búsquedas se cruzaron en el nodo {actual_atras}')
            return reconstruir_camino(visitados_adelante, visitados_atras, actual_atras)
        padre = sucesor_atras(actual_atras)
        if padre >= 1:
            print(f'  -> Genera hacia atrás: [{padre}]')
            if padre not in visitados_atras:
                visitados_atras[padre] = actual_atras
                frontera_atras.append(padre)
        print('')
        paso += 1

def reconstruir_camino(visitados_adelante, visitados_atras, punto_encuentro):
    camino_ida = []
    nodo = punto_encuentro
    while nodo is not None:
        camino_ida.append(nodo)
        nodo = visitados_adelante[nodo]
    camino_ida.reverse()
    camino_vuelta = []
    nodo = visitados_atras[punto_encuentro]
    while nodo is not None:
        camino_vuelta.append(nodo)
        nodo = visitados_atras[nodo]
    return camino_ida + camino_vuelta
camino_final = busqueda_bidireccional(inicio=1, objetivo=11)
print(f'Ruta final descubierta: {camino_final}')

import sys
sys.stdout.reconfigure(encoding='utf-8')

"""
Red neuronal con UNA neurona oculta.
Función discriminante: XOR
Algoritmo: Backpropagation

Arquitectura:
  - Entrada:  x0=1 (bias), x1, x2
  - Oculta:   neurona 3  -> W3j = (1.5, 1, 1)  para j=0,1,2
  - Salida:   neurona 4  -> W4j = (1, 1, -2)   para j=0,2,3
              (j=0: bias, j=2: conexión directa x2->y4, j=3: salida oculta)

Patrones:
  P1 = (x0=1, x1=0, x2=1)  ->  XOR(0,1) = 1
  P2 = (x0=1, x1=1, x2=0)  ->  XOR(1,0) = 1

Factor de aprendizaje: eta = 0.5
Función de activación: sigmoide f(net) = 1 / (1 + e^(-net))
"""

import math

# Funciones auxiliares

def sigmoid(net):
    return 1.0 / (1.0 + math.exp(-net))

def sigmoid_deriv(output):
    """Derivada expresada en función del valor de salida ya calculado."""
    return output * (1.0 - output)

def xor_target(x1, x2):
    return int(x1 != x2)

# Parámetros iniciales

eta = 0.5  # factor de aprendizaje

# Pesos neurona oculta 3 (j = 0, 1, 2)
#   W3[0] = w_30 (bias -> 3)
#   W3[1] = w_31 (x1   -> 3)
#   W3[2] = w_32 (x2   -> 3)
W3 = [1.5, 1.0, 1.0]

# Pesos neurona de salida 4 (j = 0, 2, 3)  ← nota: j=1 no existe
#   W4[0] = w_40 (bias -> 4)
#   W4[2] = w_42 (x2   -> 4)  conexión directa
#   W4[3] = w_43 (y3   -> 4)
W4 = {0: 1.0, 2: 1.0, 3: -2.0}

# Patrones: (x0, x1, x2)  — x0 siempre es 1 (bias)
patrones = [
    ([1.0, 0.0, 1.0], xor_target(0, 1)),   # P1 -> target = 1
    ([1.0, 1.0, 0.0], xor_target(1, 0)),   # P2 -> target = 1
]

# Función que realiza UNA iteración de backpropagation
def backprop_iteration(patron_name, x, target, W3, W4, eta):
    x0, x1, x2 = x

    print(f"ITERACIÓN PARA {patron_name} = {x}   target = {target}")

    # PASO 1: Forward pass
    print("\n FORWARD PASS")

    # Neurona oculta 3
    net3 = W3[0]*x0 + W3[1]*x1 + W3[2]*x2
    y3   = sigmoid(net3)

    print(f"  net3 = w30*x0 + w31*x1 + w32*x2")
    print(f"       = {W3[0]}*{x0} + {W3[1]}*{x1} + {W3[2]}*{x2} = {net3:.4f}")
    print(f"  y3   = σ(net3) = σ({net3:.4f}) = {y3:.6f}")

    # Neurona de salida 4
    net4 = W4[0]*x0 + W4[2]*x2 + W4[3]*y3
    y4   = sigmoid(net4)

    print(f"\n  net4 = w40*x0 + w42*x2 + w43*y3")
    print(f"       = {W4[0]}*{x0} + {W4[2]}*{x2} + ({W4[3]})*{y3:.6f} = {net4:.4f}")
    print(f"  y4   = σ(net4) = σ({net4:.4f}) = {y4:.6f}")

    #Cálculo del error 
    error = target - y4
    print(f"\n  Error = target - y4 = {target} - {y4:.6f} = {error:.6f}")

    # PASO 3: Backward pass — deltas 
    print("\n BACKWARD PASS")

    # Delta neurona de salida 4
    delta4 = error * sigmoid_deriv(y4)
    print(f"  δ4 = (target - y4) * y4*(1-y4)")
    print(f"     = {error:.6f} * {y4:.6f}*(1-{y4:.6f})")
    print(f"     = {error:.6f} * {sigmoid_deriv(y4):.6f} = {delta4:.6f}")

    # Delta neurona oculta 3
    delta3 = delta4 * W4[3] * sigmoid_deriv(y3)
    print(f"\n  δ3 = δ4 * w43 * y3*(1-y3)")
    print(f"     = {delta4:.6f} * ({W4[3]}) * {y3:.6f}*(1-{y3:.6f})")
    print(f"     = {delta4:.6f} * ({W4[3]}) * {sigmoid_deriv(y3):.6f} = {delta3:.6f}")

    # Actualización de pesos 
    print("\n ACTUALIZACIÓN DE PESOS")

    # Pesos de salida
    dW40 = eta * delta4 * x0
    dW42 = eta * delta4 * x2
    dW43 = eta * delta4 * y3

    print(f"  Δw40 = η * δ4 * x0 = {eta} * {delta4:.6f} * {x0} = {dW40:.6f}")
    print(f"  Δw42 = η * δ4 * x2 = {eta} * {delta4:.6f} * {x2} = {dW42:.6f}")
    print(f"  Δw43 = η * δ4 * y3 = {eta} * {delta4:.6f} * {y3:.6f} = {dW43:.6f}")

    W4_new = {
        0: W4[0] + dW40,
        2: W4[2] + dW42,
        3: W4[3] + dW43,
    }

    print(f"\n  w40: {W4[0]:.4f} + {dW40:.6f} = {W4_new[0]:.6f}")
    print(f"  w42: {W4[2]:.4f} + {dW42:.6f} = {W4_new[2]:.6f}")
    print(f"  w43: {W4[3]:.4f} + {dW43:.6f} = {W4_new[3]:.6f}")

    # Pesos ocultos
    dW30 = eta * delta3 * x0
    dW31 = eta * delta3 * x1
    dW32 = eta * delta3 * x2

    print(f"\n  Δw30 = η * δ3 * x0 = {eta} * {delta3:.6f} * {x0} = {dW30:.6f}")
    print(f"  Δw31 = η * δ3 * x1 = {eta} * {delta3:.6f} * {x1} = {dW31:.6f}")
    print(f"  Δw32 = η * δ3 * x2 = {eta} * {delta3:.6f} * {x2} = {dW32:.6f}")

    W3_new = [W3[0] + dW30, W3[1] + dW31, W3[2] + dW32]

    print(f"\n  w30: {W3[0]:.4f} + {dW30:.6f} = {W3_new[0]:.6f}")
    print(f"  w31: {W3[1]:.4f} + {dW31:.6f} = {W3_new[1]:.6f}")
    print(f"  w32: {W3[2]:.4f} + {dW32:.6f} = {W3_new[2]:.6f}")

    print(f"\n  Salida obtenida: y4 = {y4:.6f}  (target = {target})")

    return W3_new, W4_new



# Ejecución de las dos iteraciones

print(f"\nPesos iniciales:")
print(f"  W3 (neurona oculta 3):  w30={W3[0]}, w31={W3[1]}, w32={W3[2]}")
print(f"  W4 (neurona salida  4):  w40={W4[0]}, w42={W4[2]}, w43={W4[3]}")
print(f"\nFactor de aprendizaje: η = {eta}")
print(f"Función de activación: sigmoide σ(net) = 1 / (1 + e^(-net))")
print(f"Función discriminante: XOR")

# Iteración 1 — Patrón P1
x_p1, t_p1 = patrones[0]
W3, W4 = backprop_iteration("P1", x_p1, t_p1, W3, W4, eta)

# Iteración 2 — Patrón P2 (usando pesos actualizados de P1)
x_p2, t_p2 = patrones[1]
W3, W4 = backprop_iteration("P2", x_p2, t_p2, W3, W4, eta)

print(f"PESOS FINALES (tras las 2 iteraciones)")
print(f"  W3: w30={W3[0]:.6f}, w31={W3[1]:.6f}, w32={W3[2]:.6f}")
print(f"  W4: w40={W4[0]:.6f}, w42={W4[2]:.6f}, w43={W4[3]:.6f}")

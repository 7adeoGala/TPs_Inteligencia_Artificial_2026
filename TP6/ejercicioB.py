import sys
sys.stdout.reconfigure(encoding='utf-8')
import math
import random

# Funciones auxiliares

def sigmoid(net):
    return 1.0 / (1.0 + math.exp(-net))

def sigmoid_deriv(y):
    return y * (1.0 - y)

def discriminante(x1, x2):
    return 1 if (3*x1 + 2*x2) > 2 else 0

# Parámetros

eta = 0.5

# Pesos iniciales aleatorios pequeños (positivos y negativos)
random.seed(42)  # fijamos seed para reproducibilidad
W = [random.uniform(-0.5, 0.5) for _ in range(3)]   # [w0, w1, w2]

# Patrones
patrones = [
    ("P1", [1.0, 1.0, 1.0], discriminante(1, 1)),   # target = 1
    ("P2", [1.0, 1.0, 0.0], discriminante(1, 0)),   # target = 1
    ("P3", [1.0, 0.0, 1.0], discriminante(0, 1)),   # target = 0
]

def delta_iteration(nombre, x, target, W, eta):
    x0, x1, x2 = x

    print(f"ITERACIÓN {nombre} = ({x1},{x2})   target = {target}")

    # Forward pass
    net = W[0]*x0 + W[1]*x1 + W[2]*x2
    y   = sigmoid(net)

    print(f"\n  net = w0*x0 + w1*x1 + w2*x2")
    print(f"      = {W[0]}*{x0} + {W[1]}*{x1} + {W[2]}*{x2} = {net:.6f}")
    print(f"  y   = σ({net:.6f}) = {y:.6f}")

    # Error
    error = target - y
    print(f"\n  Error = target - y = {target} - {y:.6f} = {error:.6f}")

    # Delta
    delta = error * sigmoid_deriv(y)
    print(f"\n  δ = (t - y) * y*(1-y)")
    print(f"    = {error:.6f} * {y:.6f}*(1-{y:.6f})")
    print(f"    = {error:.6f} * {sigmoid_deriv(y):.6f} = {delta:.6f}")

    # Actualización de pesos
    dW = [eta * delta * xi for xi in x]

    print(f"\n  Δw0 = η * δ * x0 = {eta} * {delta:.6f} * {x0} = {dW[0]:.6f}")
    print(f"  Δw1 = η * δ * x1 = {eta} * {delta:.6f} * {x1} = {dW[1]:.6f}")
    print(f"  Δw2 = η * δ * x2 = {eta} * {delta:.6f} * {x2} = {dW[2]:.6f}")

    W_new = [W[i] + dW[i] for i in range(3)]

    print(f"\n  w0: {W[0]:.6f} + {dW[0]:.6f} = {W_new[0]:.6f}")
    print(f"  w1: {W[1]:.6f} + {dW[1]:.6f} = {W_new[1]:.6f}")
    print(f"  w2: {W[2]:.6f} + {dW[2]:.6f} = {W_new[2]:.6f}")

    print(f"\n  Salida obtenida: y = {y:.6f}  (target = {target})")

    return W_new


# Ejecución
print("Pesos iniciales (pequeños, positivos y negativos):")
print(f"  W = [w0={W[0]}, w1={W[1]}, w2={W[2]}]")
print(f"\nFactor de aprendizaje: η = {eta}")

for nombre, x, target in patrones:
    W = delta_iteration(nombre, x, target, W, eta)

print(f"\nPESOS FINALES (tras las 3 iteraciones):")
print(f"  w0 = {W[0]:.6f}")
print(f"  w1 = {W[1]:.6f}")
print(f"  w2 = {W[2]:.6f}")

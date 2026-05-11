import sys
sys.stdout.reconfigure(encoding='utf-8')
import math

# Neurona oculta X3: recibe x0, x1, x2  ->  W3 = [w30, w31, w32]
# Neurona salida X4: recibe x0, x2, x3  ->  W4 = [w40, w42, w43]
#
# Pesos iniciales 
#   W3 = [w30=0.3, w31=-0.1, w32=0.2]
#   W4 = [w40=-0.2, w42=0.3, w43=-0.3]   (indices j=0, j=2, j=3)
#
# Factor de aprendizaje: eta = 0.5
# Funcion de activacion: sigmoide
# Algoritmo: Backpropagation
# Funcion discriminante: XOR
#   P1=(x0=1, x1=0, x2=1) -> XOR(0,1) = 1
#   P2=(x0=1, x1=1, x2=0) -> XOR(1,0) = 1

def sigmoid(net):
    return 1.0 / (1.0 + math.exp(-net))

def sigmoid_deriv(y):
    return y * (1.0 - y)

def xor_target(x1, x2):
    return int(x1 != x2)

# Parametros
eta = 0.5

# Pesos iniciales
W3 = [0.3, -0.1, 0.2]    # [w30, w31, w32]
W4 = [-0.2, 0.3, -0.3]   # [w40, w42, w43]  (j=0, j=2, j=3)

patrones = [
    ("P1", [1.0, 0.0, 1.0], xor_target(0, 1)),   # target = 1
    ("P2", [1.0, 1.0, 0.0], xor_target(1, 0)),   # target = 1
]


def backprop_iteration(nombre, x, target, W3, W4, eta):
    x0, x1, x2 = x
    # W4[0]=w40 (bias->salida), W4[1]=w42 (x2->salida), W4[2]=w43 (y3->salida)
    w40, w42, w43 = W4

    print(f"ITERACION {nombre} = (x1={x1}, x2={x2})   target = {target}")

    # FORWARD PASS
    print("\n  FORWARD PASS")

    # Neurona oculta X3: recibe x0, x1, x2
    net3 = W3[0]*x0 + W3[1]*x1 + W3[2]*x2
    y3   = sigmoid(net3)
    print(f"  net3 = w30*x0 + w31*x1 + w32*x2")
    print(f"       = {W3[0]}*{x0} + {W3[1]}*{x1} + {W3[2]}*{x2} = {net3:.6f}")
    print(f"  y3   = σ({net3:.6f}) = {y3:.6f}")

    # Neurona salida X4: recibe x0, x2, y3  (NO x1)
    net4 = w40*x0 + w42*x2 + w43*y3
    y4   = sigmoid(net4)
    print(f"\n  net4 = w40*x0 + w42*x2 + w43*y3")
    print(f"       = {w40}*{x0} + {w42}*{x2} + ({w43})*{y3:.6f} = {net4:.6f}")
    print(f"  y4   = σ({net4:.6f}) = {y4:.6f}")

    # Error
    error = target - y4
    print(f"\n  Error = target - y4 = {target} - {y4:.6f} = {error:.6f}")

    # BACKWARD PASS
    print("\n  BACKWARD PASS")

    # Delta neurona de salida X4
    delta4 = error * sigmoid_deriv(y4)
    print(f"  δ4 = error * y4*(1-y4)")
    print(f"     = {error:.6f} * {sigmoid_deriv(y4):.6f} = {delta4:.6f}")

    # Delta neurona oculta X3 (el error fluye solo a traves de w43)
    delta3 = delta4 * w43 * sigmoid_deriv(y3)
    print(f"\n  δ3 = δ4 * w43 * y3*(1-y3)")
    print(f"     = {delta4:.6f} * ({w43}) * {sigmoid_deriv(y3):.6f} = {delta3:.6f}")

    # ACTUALIZACION DE PESOS
    print("\n  ACTUALIZACION DE PESOS")

    # Pesos de salida X4 (recibe x0, x2, y3)
    dW40 = eta * delta4 * x0
    dW42 = eta * delta4 * x2
    dW43 = eta * delta4 * y3

    print(f"  Δw40 = η*δ4*x0 = {eta}*{delta4:.6f}*{x0} = {dW40:.6f}")
    print(f"  Δw42 = η*δ4*x2 = {eta}*{delta4:.6f}*{x2} = {dW42:.6f}")
    print(f"  Δw43 = η*δ4*y3 = {eta}*{delta4:.6f}*{y3:.6f} = {dW43:.6f}")

    W4_new = [w40 + dW40, w42 + dW42, w43 + dW43]

    print(f"\n  w40: {w40:.6f} + {dW40:.6f} = {W4_new[0]:.6f}")
    print(f"  w42: {w42:.6f} + {dW42:.6f} = {W4_new[1]:.6f}")
    print(f"  w43: {w43:.6f} + {dW43:.6f} = {W4_new[2]:.6f}")

    # Pesos ocultos X3 (recibe x0, x1, x2)
    dW30 = eta * delta3 * x0
    dW31 = eta * delta3 * x1
    dW32 = eta * delta3 * x2

    print(f"\n  Δw30 = η*δ3*x0 = {eta}*{delta3:.6f}*{x0} = {dW30:.6f}")
    print(f"  Δw31 = η*δ3*x1 = {eta}*{delta3:.6f}*{x1} = {dW31:.6f}")
    print(f"  Δw32 = η*δ3*x2 = {eta}*{delta3:.6f}*{x2} = {dW32:.6f}")

    W3_new = [W3[0]+dW30, W3[1]+dW31, W3[2]+dW32]

    print(f"\n  w30: {W3[0]:.6f} + {dW30:.6f} = {W3_new[0]:.6f}")
    print(f"  w31: {W3[1]:.6f} + {dW31:.6f} = {W3_new[1]:.6f}")
    print(f"  w32: {W3[2]:.6f} + {dW32:.6f} = {W3_new[2]:.6f}")

    print(f"\n  Salida: y4 = {y4:.6f}  (target = {target})")

    return W3_new, W4_new


# Ejecucion
print("Pesos iniciales")
print(f"  W3 = [w30={W3[0]}, w31={W3[1]}, w32={W3[2]}]")
print(f"  W4 = [w40={W4[0]}, w42={W4[1]}, w43={W4[2]}]")
print(f"Factor de aprendizaje: eta = {eta}")
print()

for nombre, x, target in patrones:
    W3, W4 = backprop_iteration(nombre, x, target, W3, W4, eta)
    print()

print("PESOS FINALES")
print(f"  W3: w30={W3[0]:.6f}, w31={W3[1]:.6f}, w32={W3[2]:.6f}")
print(f"  W4: w40={W4[0]:.6f}, w42={W4[1]:.6f}, w43={W4[2]:.6f}")

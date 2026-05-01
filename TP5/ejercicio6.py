import numpy as np
import random
import matplotlib.pyplot as plt

# 1. Definición de la Red Neuronal (Perceptrón Multicapa)
def sigmoid(x):
    # Clip para evitar overflow en exp
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def mlp_forward(X, params):
    
    # Desempaquetar los parámetros 
    W1 = params[0:4].reshape(2, 2)
    b1 = params[4:6].reshape(2)
    W2 = params[6:10].reshape(2, 2)
    b2 = params[10:12].reshape(2)
    
    # Capa oculta
    hidden = sigmoid(np.dot(X, W1) + b1)
    # Capa de salida
    output = sigmoid(np.dot(hidden, W2) + b2)
    
    return output

# 2. Generación del Dataset
def generate_data(num_samples=200):
   
    np.random.seed(42)
    X = np.random.uniform(-10, 10, (num_samples, 2))
    Y = np.zeros((num_samples, 2))
    for i in range(num_samples):
        x, y = X[i]
        if x >= 0 and y >= 0:
            Y[i] = [0, 0] # C1
        elif x < 0 and y >= 0:
            Y[i] = [0, 1] # C2
        elif x < 0 and y < 0:
            Y[i] = [1, 0] # C3
        else:
            Y[i] = [1, 1] # C4
    return X, Y

# 3. Funciones de Evaluación 
def mse_loss(params, X, Y):
    preds = mlp_forward(X, params)
    return np.mean((preds - Y) ** 2)

def accuracy(params, X, Y):
    preds = mlp_forward(X, params)
    preds_rounded = np.round(preds) # Redondear a 0 o 1
    correct = np.all(preds_rounded == Y, axis=1)
    return np.mean(correct)

# Algoritmo Genético Simple
def ags_optimize(X, Y, pop_size=150, generations=500, mut_rate=0.2):
    
    # Inicialización de población
    pop = np.random.uniform(-5, 5, (pop_size, 12))
    
    best_loss = float('inf')
    best_params = None
    loss_history = []

    for gen in range(generations):
        # Minimizar MSE
        losses = np.array([mse_loss(ind, X, Y) for ind in pop])
        
        min_idx = np.argmin(losses)
        if losses[min_idx] < best_loss:
            best_loss = losses[min_idx]
            best_params = pop[min_idx].copy()
            
        loss_history.append(best_loss)
        
        # preservamos al mejor individuo
        new_pop = [best_params.copy()]
        
        # Generar nueva población
        while len(new_pop) < pop_size:
            # Selección por Torneo (k=2)
            i1, i2 = np.random.choice(pop_size, 2, replace=False)
            p1 = pop[i1] if losses[i1] < losses[i2] else pop[i2]
            
            i3, i4 = np.random.choice(pop_size, 2, replace=False)
            p2 = pop[i3] if losses[i3] < losses[i4] else pop[i4]
            
            # Cruzamiento Uniforme
            mask = np.random.rand(12) < 0.5
            child = np.where(mask, p1, p2)
            
            # Mutación 
            if np.random.rand() < mut_rate:
                child += np.random.normal(0, 1.0, 12)
                
            new_pop.append(child)
            
        pop = np.array(new_pop)
        
    return best_params, best_loss, loss_history

# Particle Swarm Optimization (PSO)
def pso_optimize(X, Y, num_particles=150, iterations=500, w=0.7, c1=1.5, c2=1.5):
    # Inicialización
    particles = np.random.uniform(-5, 5, (num_particles, 12))
    velocities = np.random.uniform(-1, 1, (num_particles, 12))
    
    pbests = particles.copy()
    pbest_losses = np.array([mse_loss(p, X, Y) for p in particles])
    
    gbest = pbests[np.argmin(pbest_losses)].copy()
    gbest_loss = np.min(pbest_losses)
    
    loss_history = []
    
    for _ in range(iterations):
        for j in range(num_particles):
            # Actualizar velocidad
            r1 = np.random.rand(12)
            r2 = np.random.rand(12)
            velocities[j] = (w * velocities[j] + 
                             c1 * r1 * (pbests[j] - particles[j]) + 
                             c2 * r2 * (gbest - particles[j]))
            
            # Actualizar posición
            particles[j] += velocities[j]
            
            # Evaluar aptitud
            loss = mse_loss(particles[j], X, Y)
            if loss < pbest_losses[j]:
                pbest_losses[j] = loss
                pbests[j] = particles[j].copy()
                
                # Actualizar mejor global 
                if loss < gbest_loss:
                    gbest_loss = loss
                    gbest = particles[j].copy()
                    
        loss_history.append(gbest_loss)
        
    return gbest, gbest_loss, loss_history

if __name__ == '__main__':
    # Generar datos de entrenamiento y prueba
    X_train, Y_train = generate_data(400)
    X_test, Y_test = generate_data(100)
    
    
    # Entrenamiento con Algoritmo Genético
    best_ags, loss_ags, hist_ags = ags_optimize(X_train, Y_train, pop_size=150, generations=600, mut_rate=0.2)
    acc_ags = accuracy(best_ags, X_test, Y_test)
    print(f"-> Mejor pérdida (MSE) AGS: {loss_ags:.4f}")
    print(f"-> Precisión en set de prueba AGS: {acc_ags*100:.2f}%\n")
    
    # Entrenamiento con PSO
    best_pso, loss_pso, hist_pso = pso_optimize(X_train, Y_train, num_particles=150, iterations=600)
    acc_pso = accuracy(best_pso, X_test, Y_test)
    print(f"-> Mejor pérdida (MSE) PSO: {loss_pso:.4f}")
    print(f"-> Precisión en set de prueba PSO: {acc_pso*100:.2f}%\n")
    
    # Comparación Final
    print("COMPARACIÓN FINAL")
    print(f"AGS - Precisión: {acc_ags*100:.2f}%, MSE: {loss_ags:.4f}")
    print(f"PSO - Precisión: {acc_pso*100:.2f}%, MSE: {loss_pso:.4f}")
    
    # Pruebas manuales con los puntos del enunciado
    print("\nPrueba con puntos específicos")
    puntos_prueba = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    nombres_cuadrantes = ["C1(0,0) [I Cuad]", "C2(0,1) [II Cuad]", "C3(1,0) [III Cuad]", "C4(1,1) [IV Cuad]"]
    
    predicciones = mlp_forward(puntos_prueba, best_pso)
    for i, pt in enumerate(puntos_prueba):
        pred_redondeada = np.round(predicciones[i]).astype(int)
        print(f"Punto {pt} -> Salida red: {pred_redondeada} | Esperado: {nombres_cuadrantes[i]}")
    
    # Graficar la curva de convergencia y los cuadrantes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico de Convergencia
    ax1.plot(hist_ags, label='Algoritmo Genético (AGS)', color='blue')
    ax1.plot(hist_pso, label='PSO', color='red')
    ax1.set_title('Evolución del Error (MSE) al ajustar pesos')
    ax1.set_xlabel('Generación / Iteración')
    ax1.set_ylabel('Error Cuadrático Medio (MSE)')
    ax1.legend()
    ax1.grid(True)
    
    # Gráfico de Cuadrantes y puntos

    ax2.axhline(0, color='black', linewidth=1)
    ax2.axvline(0, color='black', linewidth=1)
    
    ax2.fill_between([0, 10], 0, 10, color='lightblue', alpha=0.3, label='C1(0,0) [I Cuad]')
    ax2.fill_between([-10, 0], 0, 10, color='lightgreen', alpha=0.3, label='C2(0,1) [II Cuad]')
    ax2.fill_between([-10, 0], -10, 0, color='lightcoral', alpha=0.3, label='C3(1,0) [III Cuad]')
    ax2.fill_between([0, 10], -10, 0, color='khaki', alpha=0.3, label='C4(1,1) [IV Cuad]')
    
    puntos_x = [1, -1, -1, 1]
    puntos_y = [1, 1, -1, -1]
    colores = ['blue', 'green', 'red', 'orange']
    
    for i in range(4):
        ax2.scatter(puntos_x[i], puntos_y[i], color=colores[i], s=100, edgecolors='black', zorder=5)
        ax2.text(puntos_x[i] + 0.3, puntos_y[i] + 0.3, f"{puntos_prueba[i]}", fontsize=10, fontweight='bold')
    
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.set_title('Plano, Cuadrantes y Puntos de Prueba')
    ax2.set_xlabel('Eje X')
    ax2.set_ylabel('Eje Y')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

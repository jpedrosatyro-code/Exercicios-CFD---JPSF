import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# DADOS DE ENTRADA E PARÂMETROS DA MALHA
# =============================================================================
N = 50                  # Número de Volumes de Controle (Malha)
dX = 1.0 / N            # Tamanho do volume adimensional (L=1 na forma adimensional)
dFo = 0.001            # Passo de tempo adimensional (tau)

# Posições dos nós (centro dos volumes)
X_pos = np.linspace(dX/2, 1.0 - dX/2, N)

# =============================================================================
# CONDIÇÕES INICIAIS (t=0)
# Distribuição linear de T0 a TL. 
# Adimensionalmente: theta = (T-T0)/(TL-T0).
# Em x=0 -> theta=0. Em x=L -> theta=1.
# Logo, Theta_inicial = X
# =============================================================================
Theta = X_pos.copy()    # T inicial linear
Theta_old = Theta.copy()

# =============================================================================
# PREPARAÇÃO PARA O SOLVER (Coeficientes constantes)
# =============================================================================
# Equação Geral: ap*Tp = aw*Tw + ae*Te + b
# Termo difusivo adimensional: 1 / dX
diff = 1.0 / dX
# Termo transiente adimensional: dX / dFo
ap0 = dX / dFo

# Vetores para o TDMA
aW = np.zeros(N)
aE = np.zeros(N)
aP = np.zeros(N)
b  = np.zeros(N)

# Variável para contar o tempo adimensional
Fo_atual = 0.0

# Listas para armazenar resultados para plotagem
historico_plots = []
# Salva a condição inicial
historico_plots.append((0, Theta.copy()))

# =============================================================================
# FUNÇÃO TDMA (Solver do Sistema Linear)
# =============================================================================
def solver_tdma(aW, aP, aE, b):
    n = len(b)
    P = np.zeros(n)
    Q = np.zeros(n)
    Result = np.zeros(n)
    
    # Forward
    P[0] = aE[0] / aP[0]
    Q[0] = b[0] / aP[0]
    for i in range(1, n):
        denom = aP[i] - aW[i] * P[i-1]
        if i < n-1:
            P[i] = aE[i] / denom
        Q[i] = (b[i] + aW[i] * Q[i-1]) / denom
        
    # Backward
    Result[n-1] = Q[n-1]
    for i in range(n-2, -1, -1):
        Result[i] = P[i] * Result[i+1] + Q[i]
    return Result

# =============================================================================
# LOOP NO TEMPO (CRITÉRIO DE PARADA: Theta(x=L) < 0.5)
# =============================================================================
print("Iniciando simulação...")

# O último nó (índice -1) representa a região próxima a x=L
while Theta[-1] >= 0.5:
    Fo_atual += dFo
    
    # Montagem dos Coeficientes
    for i in range(N):
        # --- FRONTEIRA ESQUERDA (i=0) : Theta = 0 ---
        if i == 0:
            aW[i] = 0.0
            aE[i] = diff
            
            # Temperatura prescrita na face esquerda (Theta_wall = 0)
            # Fluxo difusivo na face: q = (Tp - T_wall) / (dX/2)
            # Coeficiente da fronteira: 2 / dX
            coef_bound = 2.0 / dX
            
            # Linearização da Fonte: Sp e Su
            # Como Theta_wall = 0, Su = coef * 0 = 0
            Su = 0.0
            Sp = -coef_bound 
            
        # --- FRONTEIRA DIREITA (i=N-1) : Adiabática (Isolada) ---
        elif i == N-1:
            aW[i] = diff
            aE[i] = 0.0 # Sem vizinho à direita
            
            # Isolamento térmico: Fluxo = 0, logo Sp=0 e Su=0
            Sp = 0.0
            Su = 0.0
            
        # --- NÓS INTERNOS ---
        else:
            aW[i] = diff
            aE[i] = diff
            Sp = 0.0
            Su = 0.0
            
        # Montagem do termo b e aP
        b[i]  = Su + ap0 * Theta_old[i]
        aP[i] = aW[i] + aE[i] + ap0 - Sp

    # Resolver Sistema
    Theta = solver_tdma(aW, aP, aE, b)
    
    # Atualizar passo anterior
    Theta_old = Theta.copy()
    
    # Salvar alguns perfis intermediários para o gráfico (a cada 0.05 de Fo)
    # Apenas para visualização, não afeta o cálculo
    if abs(Fo_atual % 0.005) < dFo: 
        historico_plots.append((Fo_atual, Theta.copy()))

# Salva o último perfil (quando atingiu < 0.5)
historico_plots.append((Fo_atual, Theta.copy()))

print(f"Critério atingido! Theta(x=L) = {Theta[-1]:.4f}")
print(f"Tempo adimensional final (Fo): {Fo_atual:.4f}")

# =============================================================================
# PLOTAGEM DOS RESULTADOS
# =============================================================================
plt.figure(figsize=(10, 6))

# Plotar perfis armazenados
# Selecionamos alguns perfis do histórico para não poluir o gráfico
indices_to_plot = np.linspace(0, len(historico_plots)-1, 6, dtype=int)

for idx in indices_to_plot:
    t_val, profile = historico_plots[idx]
    plt.plot(X_pos, profile, label=f'Fo = {t_val:.3f}')

# Configurações do gráfico
plt.title(r'Distribuição Adimensional de Temperatura $\theta(X, \tau)$')
plt.xlabel(r'Posição Adimensional ($X = x/L$)')
plt.ylabel(r'Tempeaura adimensional ($\theta$)')
plt.axhline(0.5, color='black', linestyle='--', alpha=0.5, label='Critério de Parada (0.5)')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.show()
###############################################################################
#  PMT07 – Transferência de Calor e Mecânica dos Fluidos Computacional        #
#                    Prof. Dr. Thiago Antonini Alves                          #
#                                2025/3                                       #
#                        ATIVIDADE COMPUTACIONAL 01                           #
#           Aluno: George Stephane Queiroz de Oliveira                        #
#                       João Pedro Satyro Florido                             #  
#                                                                             #
###############################################################################

import numpy as np
import matplotlib.pyplot as plt

# Dados de entrada do problema 
N = 50                 # Volumes de controle
dX = 1.0 / N            # Tamanho do volume
dFo = 0.001            # Passo de tempo adimensional (Fourier)

# Posições dos nós (centro dos volumes)
X_pos = np.linspace(dX/2, 1.0 - dX/2, N)

# CONDIÇÕES INICIAIS (t=0)
# Adimensionalmente: theta = (T-T0)/(TL-T0).
Theta = X_pos.copy()    # T inicial linear
Theta_old = Theta.copy()

# Equação Geral: ap*Tp = aw*Tw + ae*Te + b
# Termo difusivo adimensional: 1 / dX
diff = 1.0 / dX
# Termo transiente adimensional: dX / dFo
ap0 = dX / dFo

# Vetores para o cálculo da TDMA

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

# FUNÇÃO que resolve oTDMA 
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

# Parte transiente com critério de parada theta < 0.5

while Theta[-1] >= 0.5:
    Fo_atual += dFo #adiciona o passo de tempo adimensional
    
    # Montagem dos Coeficientes
    for i in range(N):
        # Na fronteira esquerda, Theta = 0
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
            
        # Na fronteira direita, está adiabática
        elif i == N-1:
            aW[i] = diff
            aE[i] = 0.0 # Sem vizinho à direita
            
            # Isolamento térmico: Fluxo = 0, logo Sp=0 e Su=0
            Sp = 0.0
            Su = 0.0
            
        # Para resolver os nós internos
        else:
            aW[i] = diff
            aE[i] = diff
            Sp = 0.0
            Su = 0.0
            
        # Montagem do termo fonte e aP
        b[i]  = Su + ap0 * Theta_old[i]
        aP[i] = aW[i] + aE[i] + ap0 - Sp

    # Resolver Sistema
    Theta = solver_tdma(aW, aP, aE, b)
    
    # Atualizar passo anterior
    Theta_old = Theta.copy()
    
    # Salvar alguns perfis intermediários para o gráfico (a cada 0.005 de Fo)
    if abs(Fo_atual % 0.005) < dFo: 
        historico_plots.append((Fo_atual, Theta.copy()))

# Salva o último perfil (quando atingiu < 0.5)
historico_plots.append((Fo_atual, Theta.copy()))


# Plot dos gráficos
plt.figure(figsize=(10, 6))

# Plotar perfis armazenados

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
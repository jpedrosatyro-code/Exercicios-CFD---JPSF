import numpy as np
import matplotlib.pyplot as plt

# =================================================================
# DADOS DE ENTRADA (Conforme Lista de Exercícios 6)
# =================================================================
L = 0.1             # Comprimento da barra [m]
k = 25.0            # Condutividade térmica [W/mK]
rho_cp = 10.0e6     # rho * cp [J/m3K] (10 MJ/m3K)
Ti = 150.0          # Temperatura inicial [C]
T_fronteira = 0.0   # Temperatura imposta em x=L [C]
N = 30              # Número de volumes de controle
tempo_total = 360.0 # Tempo final [s]
dt = 1.0            # Passo de tempo [s] (pode ajustar se quiser)

# Cálculos geométricos e propriedades
dx = L / N                  # Tamanho do volume de controle
alpha = k / rho_cp          # Difusividade térmica
nt = int(tempo_total / dt)  # Número de passos de tempo

# Posição dos nós (no centro dos volumes)
x = np.linspace(dx/2, L - dx/2, N)

# =================================================================
# FUNÇÃO TDMA (Algoritmo de Thomas)
# =================================================================
def resolver_tdma(aW, aP, aE, b):
    n = len(b)
    P = np.zeros(n)
    Q = np.zeros(n)
    T = np.zeros(n)

    # Eliminação para frente
    P[0] = aE[0] / aP[0]
    Q[0] = b[0] / aP[0]
    for i in range(1, n):
        denominador = aP[i] - aW[i] * P[i-1]
        if i < n-1: # aE só existe até o penúltimo
            P[i] = aE[i] / denominador
        Q[i] = (b[i] + aW[i] * Q[i-1]) / denominador

    # Substituição para trás
    T[n-1] = Q[n-1]
    for i in range(n-2, -1, -1):
        T[i] = P[i] * T[i+1] + Q[i]
    
    return T

# =================================================================
# LOOP PRINCIPAL (MVF - Esquema Totalmente Implícito)
# =================================================================

# Condição Inicial
T = np.ones(N) * Ti
T_anterior = T.copy()

# Coeficientes que são constantes (regime permanente da discretização)
# Fluxo difusivo: D = k*A/dx. Aqui A=1 unitária.
D = k / dx 
# Termo transiente: a0 = rho*cp*dx/dt
a0 = (rho_cp * dx) / dt

print(f"Iniciando cálculo para {nt} passos de tempo...")

# Loop no tempo
tempo_atual = 0
for t in range(nt):
    tempo_atual += dt
    
    # Vetores para a matriz tridiagonal
    aW = np.zeros(N)
    aE = np.zeros(N)
    aP = np.zeros(N)
    b = np.zeros(N) # Termo fonte + termo transiente antigo

    for i in range(N):
        # Vizinhos padrão (nós internos)
        # aW[i] = D
        # aE[i] = D
        
        # --- FRONTEIRA ESQUERDA (i=0) ---
        if i == 0:
            aW[i] = 0.0 # Isolamento térmico (fluxo nulo)
            aE[i] = D
            # O termo fonte Su e Sp são zero pois é adiabático
            Sp = 0
            Su = 0
        
        # --- FRONTEIRA DIREITA (i=N-1) ---
        elif i == N-1:
            aW[i] = D
            aE[i] = 0.0 # Não tem nó depois da fronteira
            
            # Temperatura prescrita na face: T_face = 0
            # Fluxo sai do nó P e vai pra parede: q = k*(TP - T_parede) / (dx/2)
            # Isso gera um termo fonte linearizado: S = Su + Sp*Tp
            # Coeficiente da troca com a parede (distância é dx/2)
            coef_parede = (k) / (dx/2) # = 2*D
            
            Sp = -coef_parede
            Su = coef_parede * T_fronteira
            
        # --- NÓS INTERNOS ---
        else:
            aW[i] = D
            aE[i] = D
            Sp = 0
            Su = 0

        # Montagem da equação discretizada: aP*TP = aW*TW + aE*TE + b
        # aP = aW + aE + a0 - Sp
        aP[i] = aW[i] + aE[i] + a0 - Sp
        
        # O vetor b carrega o passo de tempo anterior e o termo constante da fonte
        b[i] = a0 * T_anterior[i] + Su

    # Resolve o sistema linear
    T = resolver_tdma(aW, aP, aE, b)
    
    # Atualiza a temperatura para o próximo passo
    T_anterior = T.copy()

print("Cálculo numérico finalizado.")

# =================================================================
# SOLUÇÃO ANALÍTICA
# =================================================================
# T(x,t) = Ti * (4/pi) * somatório ...
def calc_analitica(x_vals, t_final):
    if t_final == 0: return np.ones_like(x_vals)*Ti
    
    soma = 0
    # Usando 100 termos para garantir precisão
    for n in range(1, 101):
        lambda_n = (2*n - 1) * np.pi / (2*L)
        termo = ((-1)**(n+1) / (2*n - 1)) * \
                np.exp(-alpha * (lambda_n**2) * t_final) * \
                np.cos(lambda_n * x_vals)
        soma += termo
        
    return Ti * (4.0 / np.pi) * soma

x_analitico = np.linspace(0, L, 100)
T_analitico = calc_analitica(x_analitico, tempo_total)

# =================================================================
# PLOTAGEM DOS RESULTADOS
# =================================================================
plt.figure(figsize=(8, 6))

# Plot Numérico (nós discretos)
plt.plot(x, T, 'ro', label='Numérico (MVF)', markersize=6)

# Plot Analítico (linha contínua)
plt.plot(x_analitico, T_analitico, 'b-', label='Analítica (Série Exata)', linewidth=2)

# Configurações do gráfico
plt.title(f'Perfil de Temperatura em t = {tempo_total}s (Aço)')
plt.xlabel('Posição x [m]')
plt.ylabel('Temperatura [°C]')
plt.legend()
plt.grid(True)
plt.xlim(0, L)
plt.ylim(0, Ti + 10)

# Salvar e mostrar
plt.savefig('exercicio6_resultado.png')
print("Gráfico salvo como 'exercicio6_resultado.png'")
plt.show()
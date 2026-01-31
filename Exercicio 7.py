import numpy as np
import matplotlib.pyplot as plt

# --- Parâmetros do Problema  ---
L = 0.1          # Comprimento [m]
k = 0.595        # Condutividade térmica [W/mK]
rho = 997.0      # Densidade [kg/m^3]
T0 = 150.0       # Temp em x=0 [°C]
TL = 50.0        # Temp em x=L [°C]

# Solução Analítica
def solucao_analitica(u, x_vals):
    
    if u == 0:
        return T0 + (TL - T0) * (x_vals / L)
    
    # Termo P = (rho * u) / k
    P_term = (rho * u) / k
    
    # Fórmula fornecida no exercício: (T(x) - T0) / (TL - T0) = (e^(P*x) - 1) / (e^(P*L) - 1)
    # Isolando T(x):
    numerator = np.exp(P_term * x_vals) - 1
    denominator = np.exp(P_term * L) - 1
    T_x = T0 + (TL - T0) * (numerator / denominator)
    return T_x

# MVF
def solve_fvm(u, N, scheme='CDS'):
    dx = L / N
    x_nodes = np.linspace(dx/2, L - dx/2, N) # Centros dos volumes
    
    # Coeficientes
    F = rho * u
    D = k / dx
    
    # Inicializa Matriz A e vetor b (Sistema AT = b)
    A = np.zeros((N, N))
    b = np.zeros(N)
    
    for i in range(N):
        # Coeficientes vizinhos
        if scheme == 'CDS': # Diferenças Centrais
            a_E = D - F/2
            a_W = D + F/2
        elif scheme == 'UDS': # Upwind
            a_E = D        
            a_W = D + F 
        
        a_P = a_E + a_W # Continuidade base
        
        # --- Condições de Contorno ---
        
        # Fronteira Esquerda (x=0)
        if i == 0:
            # Fluxo na face oeste (borda)
            # Aproximação do gradiente na parede
            
            if scheme == 'CDS':
                # Fluxo conv na face w: F * (T0 + Tp)/2
                # Fluxo dif na face w: 2*D * (Tp - T0)
        
                flux_w_coeff = (2*D + F) # Termo que multiplica T0
                a_P += (2*D + F) # Correção devido à proximidade da borda
                b[i] += (2*D + F) * T0

                a_E_final = a_E
                a_W_final = 0 # Não há vizinho à esquerda na matriz
                a_P_final = a_E_final + (2*D + F/2) if scheme=='CDS' else 0 
                
                
                a_P = 3*D + F/2
                a_E_term = D - F/2  # Passa para o lado direito da eq como positivo no vizinho
                b[i] = T0 * (2*D + F) 
                A[i, i] = a_P
                if N > 1: A[i, i+1] = -a_E_term
                
            elif scheme == 'UDS':
                # Face leste: F*Tp - D*(Te-Tp)
                # Face oeste: F*T0 - 2*D*(Tp-T0)
                # Eq: F*Tp - D*Te + D*Tp - F*T0 + 2*D*Tp - 2*D*T0 = 0
                # Tp*(F + 3*D) - D*Te = T0*(F + 2*D)
                
                a_P = 3*D + F
                a_E_term = D
                b[i] = T0 * (2*D + F)
                
                A[i, i] = a_P
                if N > 1: A[i, i+1] = -a_E_term

        # Fronteira Direita (x=L)
        elif i == N - 1:
             
             if scheme == 'CDS':
            
                 # Simplificação padrão BC Dirichlet direita:
                 a_E_bound = 2*D - F # Coef que multiplicaria a borda
                 a_P = 3*D - F/2
                 a_W_term = D + F/2
                 b[i] = TL * (2*D - F)
                 
                 A[i, i] = a_P
                 A[i, i-1] = -a_W_term
                 
             elif scheme == 'UDS':
                                
                 a_P = F + 3*D
                 a_W_term = D + F
                 b[i] = TL * (2*D)
                 
                 A[i, i] = a_P
                 A[i, i-1] = -a_W_term

        # Nós Internos
        else:
            A[i, i] = a_P
            A[i, i+1] = -a_E
            A[i, i-1] = -a_W

    # Resolver sistema linear
    T_res = np.linalg.solve(A, b)
    return x_nodes, T_res

# --- Definição dos Casos ---
cases = [
    {"name": "Caso A", "u": 0.01, "N": 5},   # [cite: 7]
    {"name": "Caso B", "u": 0.10, "N": 5},   # [cite: 8]
    {"name": "Caso C", "u": 0.10, "N": 20}   # [cite: 8]
]

# --- Plotagem ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, case in enumerate(cases):
    u_c = case["u"]
    N_c = case["N"]
    
    # 1. Solução Analítica (alta resolução para linha suave)
    x_ana = np.linspace(0, L, 100)
    T_ana = solucao_analitica(u_c, x_ana)
    
    # 2. Solução CDS
    x_cds, T_cds = solve_fvm(u_c, N_c, scheme='CDS')
    # Adicionar pontos de contorno para o gráfico ficar completo visualmente
    x_cds_plot = np.concatenate(([0], x_cds, [L]))
    T_cds_plot = np.concatenate(([T0], T_cds, [TL]))

    # 3. Solução UDS
    x_uds, T_uds = solve_fvm(u_c, N_c, scheme='UDS')
    x_uds_plot = np.concatenate(([0], x_uds, [L]))
    T_uds_plot = np.concatenate(([T0], T_uds, [TL]))
    
    # Gráfico
    ax = axes[i]
    ax.plot(x_ana, T_ana, 'k-', label='Analítica', linewidth=1.5)
    ax.plot(x_cds_plot, T_cds_plot, 'bo--', label='CDS (Central)', markersize=6)
    ax.plot(x_uds_plot, T_uds_plot, 'rs--', label='UDS (Upwind)', markersize=6)
    
    # Cálculo do Peclet para análise
    dx = L / N_c
    Pe = (rho * u_c * dx) / k
    
    ax.set_title(f'{case["name"]}: u={u_c} m/s, N={N_c}\nPe = {Pe:.2f}')
    ax.set_xlabel('Posição x [m]')
    ax.set_ylabel('Temperatura [°C]')
    ax.grid(True, linestyle=':', alpha=0.6)
    if i == 0:
        ax.legend()

plt.tight_layout()
plt.show()
###############################################################################
#  PMT07 – Transferência de Calor e Mecânica dos Fluidos Computacional        #
#                    Prof. Dr. Thiago Antonini Alves                          #
#                                2025/3                                       #
#                        ATIVIDADE COMPUTACIONAL 02                           #
#           Aluno: George Stephane Queiroz de Oliveira                        #
#                       João Pedro Satyro Florido                             #  
#                                                                             #
###############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Dados do problema

k = 1000.0           # W/(m.K)
t = 0.01             # m
Lx = 1.0             # m
Ly = 0.5             # m
q_flux = 500e3       # W/m^2  (500 kW/m^2)
T_norte = 100.0      # °C

# Malha da barra (10x5, 20x10 e 30x15 Volumes)
Nx_vol = 10
Ny_vol = 5

# número de nós
N = Nx_vol + 1   # nós em x
M = Ny_vol + 1   # nós em y

#tamanho do volume de controle
dx = Lx / Nx_vol 
dy = Ly / Ny_vol

x = np.linspace(0.0, Lx, N)
y = np.linspace(0.0, Ly, M)

# Áreas das faces com a placa de espessura t
Aw = t * dy
Ae = t * dy
As = t * dx
An = t * dx

# Distâncias para faces com meia célula no contorno)
dxw = np.full(N, dx)
dxe = np.full(N, dx)
dys = np.full(M, dy)
dyn = np.full(M, dy)

# meia célula próxima ao contorno
dxw[1] = dx / 2.0
dxe[N-2] = dx / 2.0
dys[1] = dy / 2.0
dyn[M-2] = dy / 2.0

# Variáveis do problema com a discretização tendo o vetor T[i,j]

T  = np.zeros((N, M))  
aW = np.zeros((N, M))
aE = np.zeros((N, M))
aS = np.zeros((N, M))
aN = np.zeros((N, M))
aP = np.zeros((N, M))
sC = np.zeros((N, M))
sP = np.zeros((N, M))


# Condição de contorno 

def apply_boundary_conditions():
    # Norte: T = 100
    T[:, M-1] = T_norte

    # Oeste: fluxo imposto
    for j in range(0, M-1):
        T[0, j] = (q_flux * dxw[1] / k) + T[1, j]

    # Leste: isolada
    for j in range(0, M-1):
        T[N-1, j] = T[N-2, j]

    # Sul: isolada
    T[:, 0] = T[:, 1]

# Coeficientes internos

def internal_coefficients():
    aW.fill(0.0); aE.fill(0.0); aS.fill(0.0); aN.fill(0.0)
    sC.fill(0.0); sP.fill(0.0)

    for i in range(1, N-1):
        for j in range(1, M-1):
            aW[i,j] = k * Aw / dxw[i]
            aE[i,j] = k * Ae / dxe[i]
            aS[i,j] = k * As / dys[j]
            aN[i,j] = k * An / dyn[j]

    # Como não tem geração de calor o sC,sP são zero

# Perto do contorno)

def boundary_coefficients():
    # Oeste e Leste
    for j in range(1, M-1):
        # Oeste 
        sC[1, j] += q_flux * Aw
        aW[1, j] = 0.0

        # Leste: isolada
        aE[N-2, j] = 0.0

    # Sul e Norte
    for i in range(1, N-1):
        # Norte: entra como fonte com Tnorte
        sC[i, M-2] += aN[i, M-2] * T_norte
        sP[i, M-2] -= aN[i, M-2]
        aN[i, M-2] = 0.0

        # Sul: isolada
        aS[i, 1] = 0.0

# Coeficiente aP

def compute_aP():
    for i in range(1, N-1):
        for j in range(1, M-1):
            aP[i,j] = aW[i,j] + aE[i,j] + aS[i,j] + aN[i,j] - sP[i,j]


# Algoritmo TDMA

def tdma(a, d, c, b):
    # a: subdiagonal, d: diagonal, c: superdiagonal, b: RHS
    n = len(d)
    cp = np.zeros(n)
    bp = np.zeros(n)

    cp[0] = c[0] / d[0]
    bp[0] = b[0] / d[0]

    for i in range(1, n):
        denom = d[i] - a[i] * cp[i-1]
        cp[i] = c[i] / denom if i < n-1 else 0.0
        bp[i] = (b[i] - a[i] * bp[i-1]) / denom

    xsol = np.zeros(n)
    xsol[-1] = bp[-1]
    for i in range(n-2, -1, -1):
        xsol[i] = bp[i] - cp[i] * xsol[i+1]
    return xsol


# Varredura horizontal

def sweep_horizontal():
    # resolve linha a linha (j fixo)
    for j in range(1, M-1):
        ni = N - 2  # número de incógnitas na linha
        a = np.zeros(ni)
        d = np.zeros(ni)
        c = np.zeros(ni)
        b = np.zeros(ni)

        for ii in range(ni):
            i = ii + 1
            a[ii] = -aW[i,j]
            d[ii] =  aP[i,j]
            c[ii] = -aE[i,j]
            b[ii] = sC[i,j] + aS[i,j]*T[i,j-1] + aN[i,j]*T[i,j+1]

        phi = tdma(a, d, c, b)

        for ii in range(ni):
            i = ii + 1
            T[i,j] = phi[ii]

# Varredura vertical

def sweep_vertical():
    # resolve coluna a coluna (i fixo)
    for i in range(1, N-1):
        nj = M - 2
        a = np.zeros(nj)
        d = np.zeros(nj)
        c = np.zeros(nj)
        b = np.zeros(nj)

        for jj in range(nj):
            j = jj + 1
            a[jj] = -aS[i,j]
            d[jj] =  aP[i,j]
            c[jj] = -aN[i,j]
            b[jj] = sC[i,j] + aW[i,j]*T[i-1,j] + aE[i,j]*T[i+1,j]

        phi = tdma(a, d, c, b)

        for jj in range(nj):
            j = jj + 1
            T[i,j] = phi[jj]


# Demonstra os resíduo L2 

def compute_residual_L2():
    r2 = 0.0
    for i in range(1, N-1):
        for j in range(1, M-1):
            lhs = aP[i,j]*T[i,j]
            rhs = (aE[i,j]*T[i+1,j] + aW[i,j]*T[i-1,j] +
                   aN[i,j]*T[i,j+1] + aS[i,j]*T[i,j-1] + sC[i,j])
            r = lhs - rhs
            r2 += r*r
    return np.sqrt(r2)


#  Programa principal semelhante ao do (Apêndice A)

itermax = 1000
tol = 1e-10

# geração de malha já foi feita acima (x,y,dxw,dxe,dys,dyn)

# coeficientes
apply_boundary_conditions()
internal_coefficients()
boundary_coefficients()
compute_aP()

res_hist = []
for it in range(1, itermax+1):
    sweep_horizontal()
    sweep_vertical()
    apply_boundary_conditions()

    res = compute_residual_L2()
    res_hist.append(res)

    if res < tol:
        print(f"Convergiu em {it} iterações | resíduo L2 = {res:.3e}")
        break
else:
    print(f"Não convergiu em {itermax} iterações | resíduo L2 = {res_hist[-1]:.3e}")

print(f"Tmin = {T.min():.6f} °C | Tmax = {T.max():.6f} °C")

# Aqui tem a interação das Tabelas e Gráficos

# Tabela nodal (i,j,x,y,T)
rows = []
for j in range(M):
    for i in range(N):
        rows.append([i+1, j+1, x[i], y[j], T[i,j]])

df = pd.DataFrame(rows, columns=["i", "j", "x (m)", "y (m)", "T (°C)"])
print("\nAmostra da tabela:")
print(df.head(12))

# Exporta CSV completo
out_csv = "temperatura_placa2D_MVF_10x5.csv"
df.to_csv(out_csv, index=False)
print(f"\nCSV salvo: {out_csv}")

# Campo 2D
X, Y = np.meshgrid(x, y, indexing="ij")

plt.figure()
plt.contourf(X, Y, T, levels=25)
plt.colorbar(label="T (°C)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Campo de temperatura ")
plt.show()

# Perfis T(x) em alturas selecionadas
plt.figure()
for y_target in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    j_idx = int(round(y_target / dy))
    j_idx = max(0, min(M-1, j_idx))
    plt.plot(x, T[:, j_idx], marker="o", label=f"y = {y[j_idx]:.2f} m")
plt.xlabel("x (m)")
plt.ylabel("T (°C)")
plt.title("Perfis T(x) em alturas selecionadas")
plt.legend()
plt.show()

# Histórico do resíduo
plt.figure()
plt.semilogy(np.arange(1, len(res_hist)+1), res_hist)
plt.xlabel("Iteração")
plt.ylabel("Resíduo L2")
plt.title("Histórico de convergência (L2)")
plt.grid(True, which="both")
plt.show()

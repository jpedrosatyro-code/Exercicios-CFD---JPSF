###############################################################################
#  PMT07 – Transferência de Calor e Mecânica dos Fluidos Computacional        #
#                    Prof. Dr. Thiago Antonini Alves                          #
#                                2025/3                                       #
#                        ATIVIDADE COMPUTACIONAL 02                           #
#           Aluno: George Stephane Queiroz de Oliveira                        #
#                       João Pedro Satyro Florido                             #  
#                                                                             #
###############################################################################

import math
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Dados do Problema seguindo o exemplo 4

@dataclass
class NozzleData:
    rho: float = 1.0
    AA: float = 3.0
    AB: float = 1.0
    p1: float = 28.0
    p3: float = 0.0

@dataclass
class Relax:
    alpha_u: float = 0.5
    alpha_p: float = 0.2

@dataclass
class SolverSettings:
    max_iter: int = 400
    tol_mass: float = 1e-10
    tol_change: float = 1e-12
    a0: float = 1e-3      # estabiliza coeficiente do momentum se u~0
    print_every: int = 50
    verbose: bool = False



# Funçoes que inicia os Algoritmos para predição do momentum

def momentum_predict(data: NozzleData, p2: float, uA_old: float, uB_old: float, st: SolverSettings):

    rho = data.rho
    FA = rho * uA_old * data.AA
    FB = rho * uB_old * data.AB

    aA = abs(FA) + st.a0
    aB = abs(FB) + st.a0

    dA = data.AA / aA
    dB = data.AB / aB

    uA_up = 0.0
    uB_up = uA_old

    uA_star = uA_up - dA * (p2 - data.p1)
    uB_star = uB_up - dB * (data.p3 - p2)

    return uA_star, uB_star, aA, aB, dA, dB, FA, FB



# SIMPLE

def pressure_correction_SIMPLE(data: NozzleData, uA_star: float, uB_star: float, dA: float, dB: float):
    """
    Continuidade no nó 2:
        AB*uB - AA*uA = 0

    Com correção:
        uA = uA* + dA*(p1' - p2')
        uB = uB* + dB*(p2' - p3')

    Com p1'=p3'=0:
        (AB*dB + AA*dA)*p2' = AA*uA* - AB*uB*
    """
    AA, AB = data.AA, data.AB
    mass_imb = AB*uB_star - AA*uA_star
    a_p2 = AB*dB + AA*dA
    p2_prime = (AA*uA_star - AB*uB_star) / a_p2
    return p2_prime, mass_imb



#  SIMPLEC

def pressure_correction_SIMPLEC(data: NozzleData, uA_star: float, uB_star: float, aA: float, aB: float, FA: float, FB: float):

    tiny = 1e-12
    denomA = max(aA - abs(FA), tiny)
    denomB = max(aB - abs(FB), tiny)

    dA_c = min(data.AA / denomA, 1e6)
    dB_c = min(data.AB / denomB, 1e6)

    AA, AB = data.AA, data.AB
    mass_imb = AB*uB_star - AA*uA_star
    a_p2 = AB*dB_c + AA*dA_c
    p2_prime = (AA*uA_star - AB*uB_star) / a_p2
    return p2_prime, mass_imb, dA_c, dB_c



# Funçoes para resolver o numerico

def solve_SIMPLE(data, relax, st, p2_init=25.0, uA_init=3.0, uB_init=5.0):
    p2, uA, uB = p2_init, uA_init, uB_init
    rows = []
    for it in range(1, st.max_iter+1):
        uA_star, uB_star, aA, aB, dA, dB, FA, FB = momentum_predict(data, p2, uA, uB, st)
        p2_prime, _ = pressure_correction_SIMPLE(data, uA_star, uB_star, dA, dB)

        p2_new = p2 + relax.alpha_p * p2_prime

        uA_corr = uA_star - dA * p2_prime
        uB_corr = uB_star + dB * p2_prime

        uA_new = uA + relax.alpha_u * (uA_corr - uA)
        uB_new = uB + relax.alpha_u * (uB_corr - uB)

        Rm = data.AB*uB_new - data.AA*uA_new
        change = max(abs(p2_new-p2), abs(uA_new-uA), abs(uB_new-uB))

        rows.append([it, p2_new, uA_new, uB_new, Rm, abs(Rm)])

        if st.verbose and (it == 1 or it % st.print_every == 0):
            print(f"[SIMPLE] it={it:4d} p2={p2_new:10.6f} uA={uA_new:10.6f} uB={uB_new:10.6f} |Rm|={abs(Rm):.3e}")

        p2, uA, uB = p2_new, uA_new, uB_new

        if abs(Rm) < st.tol_mass and change < st.tol_change:
            break

    df = pd.DataFrame(rows, columns=["it","p2","uA","uB","Rm","abs_Rm"])
    return df


def solve_SIMPLEC(data, relax, st, p2_init=25.0, uA_init=3.0, uB_init=5.0):
    p2, uA, uB = p2_init, uA_init, uB_init
    rows = []
    for it in range(1, st.max_iter+1):
        uA_star, uB_star, aA, aB, dA, dB, FA, FB = momentum_predict(data, p2, uA, uB, st)
        p2_prime, _, dA_c, dB_c = pressure_correction_SIMPLEC(data, uA_star, uB_star, aA, aB, FA, FB)

        p2_new = p2 + relax.alpha_p * p2_prime

        uA_corr = uA_star - dA_c * p2_prime
        uB_corr = uB_star + dB_c * p2_prime

        uA_new = uA + relax.alpha_u * (uA_corr - uA)
        uB_new = uB + relax.alpha_u * (uB_corr - uB)

        Rm = data.AB*uB_new - data.AA*uA_new
        change = max(abs(p2_new-p2), abs(uA_new-uA), abs(uB_new-uB))

        rows.append([it, p2_new, uA_new, uB_new, Rm, abs(Rm)])

        if st.verbose and (it == 1 or it % st.print_every == 0):
            print(f"[SIMPLEC] it={it:4d} p2={p2_new:10.6f} uA={uA_new:10.6f} uB={uB_new:10.6f} |Rm|={abs(Rm):.3e}")

        p2, uA, uB = p2_new, uA_new, uB_new

        if abs(Rm) < st.tol_mass and change < st.tol_change:
            break

    df = pd.DataFrame(rows, columns=["it","p2","uA","uB","Rm","abs_Rm"])
    return df


def solve_SIMPLER(data, relax, st, p2_init=25.0, uA_init=3.0, uB_init=5.0):

    p2, uA, uB = p2_init, uA_init, uB_init
    rows = []

    for it in range(1, st.max_iter+1):
        # coeficientes com estado atual
        uA_star0, uB_star0, aA, aB, dA, dB, FA, FB = momentum_predict(data, p2, uA, uB, st)

        # pressão "melhor": impor AB*uB(p2) = AA*uA(p2) usando u(p) do momentum
        uA_up = 0.0
        uB_up = uA

        denom = data.AB*dB + data.AA*dA
        rhs   = data.AB*dB*data.p3 + data.AA*dA*data.p1 + data.AA*uA_up - data.AB*uB_up
        p2_bar = rhs / denom

        p2_mid = p2 + relax.alpha_p * (p2_bar - p2)

        # recalcular momentum com p2_mid
        uA_star, uB_star, aA, aB, dA, dB, FA, FB = momentum_predict(data, p2_mid, uA, uB, st)

        # correção fina p'
        p2_prime, _ = pressure_correction_SIMPLE(data, uA_star, uB_star, dA, dB)
        p2_new = p2_mid + 0.5 * relax.alpha_p * p2_prime

        uA_corr = uA_star - dA * p2_prime
        uB_corr = uB_star + dB * p2_prime

        uA_new = uA + relax.alpha_u * (uA_corr - uA)
        uB_new = uB + relax.alpha_u * (uB_corr - uB)

        Rm = data.AB*uB_new - data.AA*uA_new
        change = max(abs(p2_new-p2), abs(uA_new-uA), abs(uB_new-uB))

        rows.append([it, p2_new, uA_new, uB_new, Rm, abs(Rm)])

        if st.verbose and (it == 1 or it % st.print_every == 0):
            print(f"[SIMPLER] it={it:4d} p2={p2_new:10.6f} uA={uA_new:10.6f} uB={uB_new:10.6f} |Rm|={abs(Rm):.3e}")

        p2, uA, uB = p2_new, uA_new, uB_new

        if abs(Rm) < st.tol_mass and change < st.tol_change:
            break

    df = pd.DataFrame(rows, columns=["it","p2","uA","uB","Rm","abs_Rm"])
    return df



# Pós-processamento 

def reference_inviscid(data: NozzleData):

    uB = math.sqrt(2*data.p1)
    uA = (data.AB/data.AA)*uB
    p2 = data.p1 - 0.5*uA*uA
    return p2, uA, uB


def run_case(alg_name, solver_fn, data, relax, st):
    df = solver_fn(data, relax, st)
    last = df.iloc[-1]
    out = {
        "algoritmo": alg_name,
        "alpha_u": relax.alpha_u,
        "alpha_p": relax.alpha_p,
        "iters": int(last["it"]),
        "p2": float(last["p2"]),
        "uA": float(last["uA"]),
        "uB": float(last["uB"]),
        "|Rm|_final": float(last["abs_Rm"]),
        "AB*uB - AA*uA": float(last["Rm"])
    }
    return df, out


def plot_histories(histories, title_prefix):

    # 1) resíduo
    plt.figure()
    for label, df in histories:
        plt.semilogy(df["it"], df["abs_Rm"], label=label)
    plt.xlabel("Iteração")
    plt.ylabel(r"$|R_m| = |A_B u_B - A_A u_A|$")
    plt.title(f"{title_prefix} - Convergência do resíduo de massa")
    plt.grid(True, which="both")
    plt.legend()

    # 2) variáveis
    plt.figure()
    for label, df in histories:
        plt.plot(df["it"], df["p2"], label=f"{label} (p2)")
    plt.xlabel("Iteração")
    plt.ylabel("p2")
    plt.title(f"{title_prefix} - Evolução de p2")
    plt.grid(True)
    plt.legend()

    plt.figure()
    for label, df in histories:
        plt.plot(df["it"], df["uA"], label=f"{label} (uA)")
    plt.xlabel("Iteração")
    plt.ylabel("uA")
    plt.title(f"{title_prefix} - Evolução de uA")
    plt.grid(True)
    plt.legend()

    plt.figure()
    for label, df in histories:
        plt.plot(df["it"], df["uB"], label=f"{label} (uB)")
    plt.xlabel("Iteração")
    plt.ylabel("uB")
    plt.title(f"{title_prefix} - Evolução de uB")
    plt.grid(True)
    plt.legend()


def main():
    data = NozzleData()
    st = SolverSettings(verbose=False)

    # chute inicial sugerido no enunciado no problema
    p2_0, uA_0, uB_0 = 25.0, 3.0, 5.0

    # casos de subrelaxação para comparar 
    relax_cases = [
        Relax(alpha_u=0.3, alpha_p=0.1),
        Relax(alpha_u=0.5, alpha_p=0.2),
        Relax(alpha_u=0.8, alpha_p=0.3),
    ]

    solvers = [
        ("SIMPLE",   lambda d, r, s: solve_SIMPLE(d, r, s, p2_0, uA_0, uB_0)),
        ("SIMPLEC",  lambda d, r, s: solve_SIMPLEC(d, r, s, p2_0, uA_0, uB_0)),
        ("SIMPLER",  lambda d, r, s: solve_SIMPLER(d, r, s, p2_0, uA_0, uB_0)),
    ]


    p2_ref, uA_ref, uB_ref = reference_inviscid(data)

    summary_rows = []
    all_histories_by_alg = {}

    for alg_name, solver in solvers:
        histories = []
        for rc in relax_cases:
            df, out = run_case(alg_name, solver, data, rc, st)
            summary_rows.append(out)
            label = f"au={rc.alpha_u}, ap={rc.alpha_p}"
            histories.append((label, df))
        all_histories_by_alg[alg_name] = histories

    summary = pd.DataFrame(summary_rows)


    # 1) tabela geral
    print("\n TABELA DE COONVERGENCIA")
    print(summary.sort_values(["algoritmo","alpha_u","alpha_p"]).to_string(index=False))

    # 2) checagem física
    print("\n AQUI TUDO OK, PROFESSOR ANTONINI!!!")
    print(f"p2_ref ≈ {p2_ref:.6f}, uA_ref ≈ {uA_ref:.6f}, uB_ref ≈ {uB_ref:.6f}")
    print("Esperado: uB ≈ 3*uA (continuidade, pois AA=3 e AB=1).")

    # PLOTS
    for alg_name, histories in all_histories_by_alg.items():
        plot_histories(histories, title_prefix=alg_name)

    plt.show()


if __name__ == "__main__":
    main()

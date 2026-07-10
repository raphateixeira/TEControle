#!/usr/bin/env python3
"""Gera as figuras de dados das Notas Rápidas. Saída em imgs/plots/*.png.

Cabeçalho de paleta/rcParams e save() copiados de
ControleClassico/scripts/gerar_figuras.py, para manter consistência
visual com o resto do site.

Requer: numpy, matplotlib, control (pip install numpy matplotlib control)
Uso:    python3 scripts/gerar_figuras.py
"""
import os

import control as ct
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "imgs", "plots")
os.makedirs(OUT, exist_ok=True)

# --- Paleta categórica fixa (ordem fixa, validada para daltonismo) ---
BLUE, AQUA, YELLOW, GREEN, VIOLET, RED, MAGENTA, ORANGE = (
    "#2a78d6", "#1baf7a", "#eda100", "#008300",
    "#4a3aa7", "#e34948", "#e87ba4", "#eb6834",
)
SERIES = [BLUE, RED, AQUA, ORANGE, VIOLET, YELLOW]
GRID = "#e1e0d9"
MUTED = "#898781"
INK = "#0b0b0b"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2.0,
    "legend.frameon": False,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def save(fig, name):
    path = os.path.join(OUT, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    print("gerado:", path)


# =====================================================================
# Nota 01 — Sintonia de PID via LGR
# =====================================================================

# Planta (mesma de ControleClassico/Notas/Aula09PID.qmd)
PLANTA_POLOS = [0, -1, -2]
ZETA, WN = 0.6, 2.0
SD = -ZETA * WN + 1j * WN * np.sqrt(1 - ZETA ** 2)
ZC = 0.6977  # zero do compensador PD, via condição de ângulo
K = 3.44  # ganho, via condição de módulo


def fig_pid_lgr_compensado():
    s = ct.tf('s')
    L = (s + ZC) / (s * (s + 1) * (s + 2))

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    rlist = ct.root_locus_map(L).loci
    for i in range(rlist.shape[1]):
        ax.plot(rlist[:, i].real, rlist[:, i].imag, color=BLUE, lw=1.6)

    ax.plot(PLANTA_POLOS, [0, 0, 0], "x", color=RED, ms=12, mew=2.5,
            label="polos de malha aberta")
    ax.plot([-ZC], [0], "o", color=GREEN, ms=9, mfc="none", mew=2.2,
            label="zero do compensador PD")
    ax.plot([SD.real], [SD.imag], "*", color=MAGENTA, ms=16,
            label=r"$s_d$ (projeto)")
    ax.plot([SD.real], [-SD.imag], "*", color=MAGENTA, ms=16)

    # Linha de zeta constante (raio pela origem)
    beta = np.arccos(ZETA)
    r = 3.2
    ax.plot([0, -r * np.cos(beta)], [0, r * np.sin(beta)], color=MUTED, lw=1, ls="--")
    ax.plot([0, -r * np.cos(beta)], [0, -r * np.sin(beta)], color=MUTED, lw=1, ls="--")

    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel(r"Re$(s)$")
    ax.set_ylabel(r"Im$(s)$")
    ax.set_title(r"LGR compensado — $C(s)G(s)=\dfrac{K(s+z_c)}{s(s+1)(s+2)}$")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(-3.2, 1.0)
    ax.set_ylim(-3.0, 3.0)
    ax.set_aspect("equal")
    save(fig, "fig_pid_lgr_compensado")


def fig_pid_lgr_resposta():
    s = ct.tf('s')
    G = 1 / (s * (s + 1) * (s + 2))

    malha_base = ct.feedback(1 * G, 1)  # proporcional puro, Kp=1, sem zero
    malha_comp = ct.feedback(K * (s + ZC) * G, 1)  # PD projetado via LGR

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    t = np.linspace(0, 14, 1400)
    for lbl, sys, color in [
        ("Sem compensação ($K_p=1$)", malha_base, MUTED),
        (r"PD via LGR ($z_c\approx 0{,}70$, $K\approx 3{,}44$)", malha_comp, BLUE),
    ]:
        tt, y = ct.step_response(sys, T=t)
        ax.plot(tt, y, color=color, label=lbl)

    ax.axhline(1.0, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("tempo $t$ [s]")
    ax.set_ylabel("$y(t)$")
    ax.set_title("Resposta ao degrau — antes e depois da compensação")
    ax.legend(fontsize=9, loc="lower right")
    save(fig, "fig_pid_lgr_resposta")


# =====================================================================
# Nota 02 — Sintonia de PID via Bode (compensador de avanço de fase)
# =====================================================================

# Planta (mesma de ControleClassico/Notas/Aula08RespostaFrequencia.qmd)
ALPHA, T_LEAD = 0.4120, 0.9828  # parâmetros do compensador de avanço


def fig_bode_lead_comparacao():
    s = ct.tf('s')
    G = 10 / (s * (s + 1) * (s + 5))
    C = (1 + T_LEAD * s) / (1 + ALPHA * T_LEAD * s)
    L = C * G

    w = np.logspace(-2, 2, 800)
    fig, axs = plt.subplots(2, 1, figsize=(7.4, 6.2), sharex=True)

    for sys, lbl, color in [(G, "$G(s)$ (sem compensação)", MUTED),
                             (L, "$C(s)G(s)$ (avanço de fase)", BLUE)]:
        mag, phase, omega = ct.frequency_response(sys, w)
        mag_db = 20 * np.log10(mag)
        phase_deg = np.degrees(np.unwrap(phase))
        axs[0].semilogx(omega, mag_db, color=color, label=lbl)
        axs[1].semilogx(omega, phase_deg, color=color, label=lbl)

        gm, pm, wg, wp = ct.margin(sys)
        if wp is not None and np.isfinite(wp):
            axs[1].plot([wp], [pm - 180], "o", color=color, ms=6)
            axs[1].annotate(rf"MF$\approx${pm:.0f}°", (wp, pm - 180),
                             textcoords="offset points", xytext=(6, -12 if color == MUTED else 8),
                             color=color, fontsize=9)

    axs[0].axhline(0, color=MUTED, lw=0.8)
    axs[0].set_ylabel("Magnitude [dB]")
    axs[0].set_title(r"Compensação por avanço de fase — $G(s)=\dfrac{10}{s(s+1)(s+5)}$")
    axs[0].legend(fontsize=9, loc="lower left")

    axs[1].axhline(-180, color=MUTED, lw=0.8)
    axs[1].set_ylabel("Fase [graus]")
    axs[1].set_xlabel(r"$\omega$ [rad/s]")

    fig.tight_layout()
    save(fig, "fig_bode_lead_comparacao")


def fig_bode_lead_resposta():
    s = ct.tf('s')
    G = 10 / (s * (s + 1) * (s + 5))
    C = (1 + T_LEAD * s) / (1 + ALPHA * T_LEAD * s)

    malha_base = ct.feedback(G, 1)
    malha_comp = ct.feedback(C * G, 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    t = np.linspace(0, 15, 1500)
    for lbl, sys, color in [
        ("Sem compensação (MF≈25°)", malha_base, MUTED),
        ("Avanço de fase (MF≈39°)", malha_comp, BLUE),
    ]:
        tt, y = ct.step_response(sys, T=t)
        ax.plot(tt, y, color=color, label=lbl)

    ax.axhline(1.0, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("tempo $t$ [s]")
    ax.set_ylabel("$y(t)$")
    ax.set_title("Resposta ao degrau — antes e depois da compensação")
    ax.legend(fontsize=9, loc="upper right")
    save(fig, "fig_bode_lead_resposta")


if __name__ == "__main__":
    fig_pid_lgr_compensado()
    fig_pid_lgr_resposta()
    fig_bode_lead_comparacao()
    fig_bode_lead_resposta()

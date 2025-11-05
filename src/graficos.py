import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

# Define os níveis e cores das zonas
ZONAS_FLUXO = {
    12.5: ('#8B0000', 'Red zone'),
    9.5: ('#FF4500', 'Orange zona'),
    5.0: ('#FFD700', 'Yellow zone'),
    4.0: ('#90EE90', 'Green zone'),
    1.6: ('#87CEEB', 'Blue zone')
}

MODELOS_ESTILO = {
    'API Standard': ('-', 'API Standard'),
    'Caetano': ('--', 'Caetano'),
    'DeRis': (':', 'De Ris')
}

def plot_flux_map_2D(Q_field: np.ndarray, x: np.ndarray, y: np.ndarray,
                     rings: dict, Q_med: float, filepath: str):
    """
    Gera um mapa de contorno 2D do fluxo térmico e sobrepõe os anéis de
    segurança calculados pelos diferentes modelos.
    """

    model_names = list(MODELOS_ESTILO.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 7),
                             sharex=True, sharey=True)
    if n_models == 1:
        axes = [axes]

    Q_min = 1e-2
    Q_field_plot = np.maximum(Q_field, Q_min)
    levels = np.logspace(np.log10(Q_min), np.log10(Q_field.max()), 50)

    legend_elements_flux = []
    flux_legend_done = {f: False for f in ZONAS_FLUXO}

    max_dist = max(rings[m][f] for m in rings for f in ZONAS_FLUXO if f in rings[m]) * 1.15

    for i, model_name in enumerate(model_names):
        ax = axes[i]
        linestyle, model_label = MODELOS_ESTILO[model_name]

        # Mapa de calor
        cf = ax.contourf(x, y, Q_field_plot, levels=levels,
                         norm=colors.LogNorm(vmin=Q_min, vmax=Q_field.max()),
                         cmap='hot')

        # Contornos de fluxo
        cs = ax.contour(x, y, Q_field_plot,
                        levels=[1.6, 4.0, 5.0, 9.5, 12.5],
                        colors='white',
                        linestyles='solid',
                        linewidths=0.5,
                        alpha=0.7)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f kW/m²')

        # Círculos/anéis de segurança
        if model_name in rings:
            for flux, (color, flux_label) in ZONAS_FLUXO.items():
                if flux not in rings[model_name]:
                    continue
                radius = rings[model_name][flux]
                circle = Circle((0, 0), radius,
                                fill=False,
                                color=color,
                                linestyle='--',
                                linewidth=2.5,
                                alpha=1.0)
                ax.add_artist(circle)
                if not flux_legend_done[flux]:
                    legend_elements_flux.append(
                        Line2D([0], [0], color=color, lw=3,
                               label=f'{flux_label} ({flux} kW/m²)')
                    )
                    flux_legend_done[flux] = True

        ax.set_xlabel('Distância X (m)', fontsize=12)
        ax.set_ylabel('Distância Y (m)', fontsize=12)
        ax.set_title(f'Modelo: {model_label}',
                     fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.5, color='white')
        ax.axhline(0, color='white', linestyle=':', alpha=0.5, lw=1)
        ax.axvline(0, color='white', linestyle=':', alpha=0.5, lw=1)
        ax.set_xlim(-max_dist, max_dist)
        ax.set_ylim(-max_dist, max_dist)

    fig.subplots_adjust(right=0.8, wspace=0.1)
    leg1 = fig.legend(handles=legend_elements_flux, title="Zonas de Fluxo (Cor)",
                      loc='upper left',
                      bbox_to_anchor=(0.81, 0.85),  # Posição (x=81%, y=85%)
                      fontsize=10, title_fontsize=11)

    cbar_ax = fig.add_axes([0.81, 0.15, 0.02, 0.45])
    cbar = fig.colorbar(cf, cax=cbar_ax)
    cbar.set_label('Fluxo Térmico Radiativo (kW/m²)', fontsize=12)

    fig.suptitle(f'Mapa de Fluxo 2D e Zonas de Segurança (Q = {Q_med:.0f} kW)',
                 fontsize=16, fontweight='bold')

    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_distance_vs_potencia(q_range: np.ndarray, dist_models: dict,
                              filepath: str):
    """
    Compara as distâncias de segurança calculadas pelos diferentes modelos
    em função da taxa de liberação de calor.
    """
    print(f"  Gerando D vs Q: {filepath}")

    zonas_nomes = [label for flux, (color, label) in ZONAS_FLUXO.items()]
    cores = [color for flux, (color, label) in ZONAS_FLUXO.items()]
    n_zones = len(zonas_nomes)

    model_names = list(dist_models.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(n_models, 1, figsize=(12, n_models * 5),
                             sharex=True, sharey=True)
    if n_models == 1:
        axes = [axes]

    for i, model_name in enumerate(model_names):
        ax = axes[i]
        data = dist_models[model_name]  # Shape (len(q_range), n_zones)

        for j in range(n_zones):
            ax.plot(q_range, data[:, j],
                    label=zonas_nomes[j],
                    color=cores[j],
                    linewidth=2.5)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('Safe zone (m)', fontsize=12)
        ax.set_title(f'Model: {MODELOS_ESTILO[model_name][1]}',
                     fontsize=13, fontweight='bold')
        ax.grid(True, which='both', linestyle=':', alpha=0.6)
        ax.legend(loc='upper left', fontsize=10)

    axes[-1].set_xlabel('Heat release rate ($Q_T$) (kW)', fontsize=12)
    fig.suptitle('Safe zone vs. Power per model',
                 fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
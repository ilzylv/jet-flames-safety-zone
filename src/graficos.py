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

def plot_flux_map_2D(Q_field: np.ndarray, x: np.ndarray, y: np.ndarray, rings: dict, Q_med: float, filepath_base: str):
    # Gera um mapa de contorno 2D do fluxo térmico e sobrepõe os anéis de segurança calculados pelos diferentes modelos.

    Q_min = 1e-2
    Q_field_plot = np.maximum(Q_field, Q_min)
    levels = np.logspace(np.log10(Q_min), np.log10(Q_field.max()), 50)

    # Elementos da legenda de zonas
    legend_elements_flux = [
        Line2D([0], [0], color=color, lw=3, label=f'{label} ({flux} kW/m²)')
        for flux, (color, label) in ZONAS_FLUXO.items()
    ]

    max_dist = max(
        rings[m][f] for m in rings for f in ZONAS_FLUXO if f in rings[m]
    ) * 1.15

    for model_name, (linestyle, model_label) in MODELOS_ESTILO.items():
        fig, ax = plt.subplots(figsize=(9, 6))

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

        # Círculos de segurança
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

        # Eixos e estilo
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_title(f'2D Flux Map – {model_label}\n(Q = {Q_med:.0f} kW)', fontsize=14, fontweight='bold', pad=10)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.4, color='white')
        ax.axhline(0, color='white', linestyle=':', alpha=0.5, lw=1)
        ax.axvline(0, color='white', linestyle=':', alpha=0.5, lw=1)
        ax.set_xlim(-max_dist, max_dist)
        ax.set_ylim(-max_dist, max_dist)

        # Barra de cores
        cbar = plt.colorbar(cf, ax=ax, pad=0.10, location='left')
        cbar.set_label('Radiative heat flux (kW/m²)', fontsize=11)

        # Legenda
        ax.legend(handles=legend_elements_flux, title="Flux zones", loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=9, title_fontsize=10, frameon=True)

        plt.tight_layout()
        filename = f"{filepath_base}_{model_name.replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

def plot_distance_vs_potencia(q_range: np.ndarray, dist_models: dict, filepath_base: str):

    # Compara as distâncias de segurança calculadas pelos diferentes modelos em função da taxa de liberação de calor.
    zonas_nomes = [label for flux, (color, label) in ZONAS_FLUXO.items()]
    cores = [color for flux, (color, label) in ZONAS_FLUXO.items()]
    n_zones = len(zonas_nomes)

    for model_name, data in dist_models.items():
        fig, ax = plt.subplots(figsize=(8, 6))

        for j in range(n_zones):
            ax.plot(q_range, data[:, j],
                    label=zonas_nomes[j],
                    color=cores[j],
                    linewidth=2.5)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Heat release rate ($Q_T$) (kW)', fontsize=12)
        ax.set_ylabel('Safe distance (m)', fontsize=12)
        ax.set_title(f'{MODELOS_ESTILO[model_name][1]}',
                     fontsize=14, fontweight='bold')
        ax.grid(True, which='both', linestyle=':', alpha=0.6)
        ax.legend(loc='upper left', fontsize=10)

        # Caminho do arquivo
        filename = f"{filepath_base}_{model_name.replace(' ', '_')}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
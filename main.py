import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from src.modelo_chama import (FlameGeometry, RadiationModels, FUELS_DATABASE)
from src.graficos import (plot_flux_map_2D, plot_distance_vs_potencia)

# Cria pasta de saída
os.makedirs('outputs', exist_ok=True)

# Parâmetros
fuel = FUELS_DATABASE['methane']

# Range de Potência (Q_T) como variável independente
# De 50 kW a 10.000 kW (10 MW)
q_range = np.logspace(np.log10(50), np.log10(10000), 80)

n_zones = 5
flux_levels = [12.5, 9.5, 5.0, 4.0, 1.6]  # kW/m²

# Matrizes de distância
potencias = q_range
comprimentos_chama = np.zeros(len(potencias))
dist_API = np.zeros((len(potencias), n_zones))
dist_Caetano = np.zeros_like(dist_API)
dist_DeRis = np.zeros_like(dist_API)

# Loop principal

for i, Q_total in enumerate(potencias):
    # Q_total é a nossa variável de entrada

    # Calcula a geometria da chama diretamente da potência
    Lf = FlameGeometry.calculate_flame_length(Q_total)
    comprimentos_chama[i] = Lf
    Wf = FlameGeometry.calculate_flame_width(Lf)
    Df = Wf # Diâmetro da chama (largura)

    # Temperatura baseada em Peng et al. (2025): 1051°C
    T_flame = 1324.0  # K

    for j, flux in enumerate(flux_levels):
        # Modelo API 521
        dist_API[i, j] = RadiationModels.api_standard_model(Q_total, K_limit=flux)

        # Modelo Caetano (com escalonamento para diferentes fluxos)
        dist_Caetano[i, j] = RadiationModels.calculate_distance_for_flux(
            Q_total, Lf, fuel, flux
        )

        # Modelo De Ris detalhado
        dist_DeRis[i, j] = RadiationModels.detailed_model_deris_distance(
            Q_total, T_flame, Lf, Df, fuel, K_limit=flux
        )

dist_models = {'API Standard': dist_API, 'Caetano': dist_Caetano, 'DeRis': dist_DeRis}

# Mapa 2D
x = np.linspace(-30, 30, 250)
y = np.linspace(-30, 30, 250)

# Condição intermediária
idx = len(potencias) // 2
Q_med = potencias[idx]
Lf_med = comprimentos_chama[idx]
Df_med = FlameGeometry.calculate_flame_width(Lf_med)
T_med = 1324.0  # K

Q_field = RadiationModels.deris_point_flux(
    Q_med, fuel.emissivity, T_med, Lf_med, Df_med, x, y,
    fr_fuel=fuel.fr_typical
)

# Anéis de segurança
rings = {
    'API Standard': {},
    'DeRis': {},
    'Caetano': {}
}

for f in flux_levels:
    rings['API Standard'][f] = RadiationModels.api_standard_model(Q_med, K_limit=f)
    rings['DeRis'][f] = RadiationModels.detailed_model_deris_distance(
        Q_med, T_med, Lf_med, Df_med, fuel, K_limit=f
    )
    rings['Caetano'][f] = RadiationModels.calculate_distance_for_flux(
        Q_med, Lf_med, fuel, f
    )

# Passa Q_med em vez de Re_med
plot_flux_map_2D(Q_field, x, y, rings, Q_med, 'outputs/mapa_2D_fluxo.png')

# DxQ
plot_distance_vs_potencia(potencias, dist_models, 'outputs/distancia_vs_potencia.png')

# Resultados
zonas_nomes = ['Red zone (12.5)', 'Orange zone (9.5)', 'Yellow zone (5.0)',
               'Green zone (4.0)', 'Blue zone (1.6)']

# Distâncias para condições selecionadas
indices_selecionados = [0, len(potencias) // 4, len(potencias) // 2, 3 * len(potencias) // 4, -1]

tabela_distancias = []
for idx in indices_selecionados:
    Q = potencias[idx]
    Lf = comprimentos_chama[idx]

    for j, zona in enumerate(zonas_nomes):
        linha = {
            'Power (kW)': f'{Q:.1f}',
            'Lf (m)': f'{Lf:.2f}',
            'Zone': zona.split('(')[0].strip(),
            'Flux (kW/m²)': zona.split('(')[1].replace(')', ''),
            'D_API_Standard (m)': f'{dist_API[idx, j]:.2f}',
            'D_Caetano (m)': f'{dist_Caetano[idx, j]:.2f}',
            'D_DeRis (m)': f'{dist_DeRis[idx, j]:.2f}'
        }
        tabela_distancias.append(linha)

df_distancias = pd.DataFrame(tabela_distancias)
df_distancias.to_csv('outputs/tabela_distancias_detalhada.csv', index=False)

# Resumo por zona (médias)
tabela_resumo = []
for j, zona in enumerate(zonas_nomes):
    linha = {
        'Safety Zone': zona.split('(')[0].strip(),
        'Flux (kW/m²)': zona.split('(')[1].replace(')', ''),
        'API Mean (m)': f'{np.mean(dist_API[:, j]):.2f}',
        'API Min-Max (m)': f'{np.min(dist_API[:, j]):.2f} - {np.max(dist_API[:, j]):.2f}',
        'Caetano Mean (m)': f'{np.mean(dist_Caetano[:, j]):.2f}',
        'Caetano Min-Max (m)': f'{np.min(dist_Caetano[:, j]):.2f} - {np.max(dist_Caetano[:, j]):.2f}',
        'DeRis Mean (m)': f'{np.mean(dist_DeRis[:, j]):.2f}',
        'DeRis Min-Max (m)': f'{np.min(dist_DeRis[:, j]):.2f} - {np.max(dist_DeRis[:, j]):.2f}'
    }
    tabela_resumo.append(linha)

df_resumo = pd.DataFrame(tabela_resumo)
df_resumo.to_csv('outputs/tabela_resumo_zonas.csv', index=False)

# Fatores de correção

# DeRis para Caetano
fator_deris_caetano = dist_Caetano / (dist_DeRis + 1e-10)
fator_api_caetano = dist_Caetano / (dist_API + 1e-10)

tabela_fatores = []
for j, zona in enumerate(zonas_nomes):
    fc_deris = fator_deris_caetano[:, j]
    fc_api = fator_api_caetano[:, j]

    linha = {
        'Zone': zona.split('(')[0].strip(),
        'Flux (kW/m²)': zona.split('(')[1].replace(')', ''),
        'FC DeRis -> Caetano (mean)': f'{np.mean(fc_deris):.3f}',
        'FC DeRis -> Caetano (min)': f'{np.min(fc_deris):.3f}',
        'FC DeRis -> Caetano (max)': f'{np.max(fc_deris):.3f}',
        'FC DeRis -> Caetano (std)': f'{np.std(fc_deris):.3f}',
        'FC API -> Caetano (mean)': f'{np.mean(fc_api):.3f}',
        'FC API -> Caetano (min)': f'{np.min(fc_api):.3f}',
        'FC API -> Caetano (max)': f'{np.max(fc_api):.3f}',
        'FC API -> Caetano (std)': f'{np.std(fc_api):.3f}'
    }
    tabela_fatores.append(linha)

df_fatores = pd.DataFrame(tabela_fatores)
df_fatores.to_csv('outputs/tabela_fatores_correcao.csv', index=False)

# Gráfico dos fatores de correção
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

colors = ['#8B0000', '#FF4500', '#FFD700', '#90EE90', '#87CEEB']

# FC DeRis -> Caetano
for j, (zona, color) in enumerate(zip(zonas_nomes, colors)):
    ax1.plot(potencias, fator_deris_caetano[:, j],
             label=zona.split('(')[0].strip(), linewidth=2, color=color)

min_val_1 = np.min(fator_deris_caetano)
max_val_1 = np.max(fator_deris_caetano)
padding_1 = (max_val_1 - min_val_1) * 0.1
padding_1 = max(padding_1, 0.05)
ax1.set_ylim(min(min_val_1, 1.0) - padding_1, max(max_val_1, 1.0) + padding_1)

ax1.set_xscale('log')
ax1.set_xlabel('Power $Q_T$ (kW)', fontsize=11)
ax1.set_ylabel('Correction factor: D_Caetano / D_DeRis', fontsize=11)
ax1.set_title('Correction factor: De Ris -> Caetano', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Parity')

# FC API -> Caetano
for j, (zona, color) in enumerate(zip(zonas_nomes, colors)):
    ax2.plot(potencias, fator_api_caetano[:, j],
             label=zona.split('(')[0].strip(), linewidth=2, color=color)

min_val_2 = np.min(fator_api_caetano)
max_val_2 = np.max(fator_api_caetano)
padding_2 = (max_val_2 - min_val_2) * 0.1
padding_2 = max(padding_2, 0.05)
ax2.set_ylim(min(min_val_2, 1.0) - padding_2, max(max_val_2, 1.0) + padding_2)

ax2.set_xscale('log')
ax2.set_xlabel('Power $Q_T$ (kW)', fontsize=11)
ax2.set_ylabel('Correction factor: D_Caetano / D_API', fontsize=11)
ax2.set_title('Correction factor: API -> Caetano', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Parity')

plt.tight_layout()
plt.savefig('outputs/fatores_correcao.png', dpi=300, bbox_inches='tight')
plt.close()

# Prints finais
print("Final results")
print(f"\nFuel: {fuel.name}")
print(f"Power range: {potencias[0]:.1f} - {potencias[-1]:.1f} kW")
print(f"Flame length range: {comprimentos_chama[0]:.2f} - {comprimentos_chama[-1]:.2f} m")

print("\nOverall model comparison")
print(f"API mean distance: {np.mean(dist_API):.2f} m (±{np.std(dist_API):.2f})")
print(f"Caetano mean distance: {np.mean(dist_Caetano):.2f} m (±{np.std(dist_Caetano):.2f})")
print(f"De Ris mean distance: {np.mean(dist_DeRis):.2f} m (±{np.std(dist_DeRis):.2f})")

diff_api = 100 * (np.mean(dist_API) / np.mean(dist_Caetano) - 1)
diff_deris = 100 * (np.mean(dist_DeRis) / np.mean(dist_Caetano) - 1)

print(f"\nDifference API vs Caetano: {diff_api:+.1f}%")
print(f"Difference DeRis vs Caetano: {diff_deris:+.1f}%")

print("Recommended correction factors")

for j, zona in enumerate(zonas_nomes):
    fc_deris_medio = np.mean(fator_deris_caetano[:, j])
    fc_api_medio = np.mean(fator_api_caetano[:, j])
    flux = zona.split('(')[1].replace(')', '')

    print(f"{zona}:")
    print(f"  D_Caetano ≈ {fc_deris_medio:.3f} * D_DeRis  (std: ±{np.std(fator_deris_caetano[:, j]):.3f})")
    print(f"  D_Caetano ≈ {fc_api_medio:.3f} * D_API     (std: ±{np.std(fator_api_caetano[:, j]):.3f})")
    print()
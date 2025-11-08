import numpy as np
from dataclasses import dataclass

# Propriedades do metano
@dataclass
class FuelProperties:
    name: str
    molecular_mass: float
    lower_heating_value: float  # MJ/kg
    # Constantes do modelo Caetano et al. (2020)
    C1: float  # Coeficiente linear (Slope, C2 no paper)
    C2: float  # Termo independente (Intercept, C1 no paper)
    # Emissividade típica
    emissivity: float
    # Fração radiativa característica
    fr_typical: float

FUELS_DATABASE = {
    'methane': FuelProperties(
        name='Metano (CH₄)',
        molecular_mass=16.0,
        lower_heating_value=50.0,
        # Valores da Tabela 1 [2] (Methane, d=8mm)
        # C1 no paper (intercept) = 0.57
        # C2 no paper (slope) = 0.52
        C1=0.52,  # Slope
        C2=0.57,  # Intercept
        emissivity=0.25,
        fr_typical=0.17  # Chamas turbulentas de metano (baixa fuligem)
    ),
}

# Geometria da chama
class FlameGeometry:
    @staticmethod
    def calculate_flame_length(Q_total_kW: float) -> float:
        """
        Lf = 2.8893 * Q_T^0.3728
        Correlação empírica validada experimentalmente [2, 5].
        """
        # Converte Q_T de kW para MW
        Q_total_MW = Q_total_kW / 1000.0

        Lf = 2.8893 * (Q_total_MW ** 0.3728)
        return Lf

    @staticmethod
    def calculate_flame_width(Lf: float, is_turbulent: bool = True) -> float:
        """
        Largura da chama proporcional ao comprimento.
        Wf ≈ 0.17*Lf (turbulento) ou 0.10*Lf (laminar)
        Baseado em Caetano et al. (2020) [2] e observações experimentais.
        """
        return 0.17 * Lf if is_turbulent else 0.10 * Lf

# Modelos de radiação térmica
class RadiationModels:
    @staticmethod
    def api_standard_model(Q_total_kW: float, K_limit: float = 1.58, transmissivity: float = 0.98) -> float:
        """
        Modelo API 521 (2014) para chamas de jato turbulento.
        """
        Fr = 0.25 * (1.0 - np.exp(-0.0036 * Q_total_kW))
        Q_rad = Fr * Q_total_kW
        D = np.sqrt((transmissivity * Q_rad) / (4.0 * np.pi * K_limit))
        return float(D)

    @staticmethod
    def get_caetano_D_ref(Lf: float, fuel: FuelProperties) -> float:
        """
        Calcula a distância de referência (D_ref) para K=1.58 kW/m²
        D_ref = (SLOPE * Lf) + INTERCEPT
        Referência: Caetano et al. (2020), Tabela 1 [2]
        """
        D_ref = fuel.C1 * Lf + fuel.C2
        return D_ref

    @staticmethod
    def detailed_model_deris_distance(Q_total_kW: float, flame_temp: float, Lf: float, Df: float, fuel: FuelProperties, K_limit: float = 1.58) -> float:
        """
        Modelo radiativo de De Ris (2000) adaptado.
        """
        Fr = fuel.fr_typical
        Q_rad = Fr * Q_total_kW
        tau = 0.98
        D = np.sqrt((tau * Q_rad) / (4.0 * np.pi * K_limit))
        return float(D)

    @staticmethod
    def deris_point_flux(Q_total_kW: float, emissivity: float, flame_temp: float, Lf: float, Df: float, x: np.ndarray, y: np.ndarray, z_receptor: float = 1.5, fr_fuel: float = 0.20) -> np.ndarray:
        """
        Campo de fluxo radiativo (kW/m²) no plano z = z_receptor.
        """
        X, Y = np.meshgrid(x, y)
        zc = Lf / 2
        R = np.sqrt(X ** 2 + Y ** 2 + (z_receptor - zc) ** 2)
        R = np.maximum(R, 0.1)
        tau = 0.98
        Q_rad = fr_fuel * Q_total_kW
        q_r = (tau * Q_rad) / (4.0 * np.pi * R ** 2)
        q_max = 50.0
        q_r = np.minimum(q_r, q_max)
        return q_r

    @staticmethod
    def calculate_distance_for_flux(Q_total_kW: float, Lf: float, fuel: FuelProperties, K_target: float) -> float:
        """
        Calcula distância de segurança para um fluxo térmico específico.
        Escalona o resultado do modelo Caetano (K=1.58) [2] para um K_target.
        """
        K_ref = 1.58  # kW/m²
        D_ref = RadiationModels.get_caetano_D_ref(Lf, fuel)
        D_target = D_ref * np.sqrt(K_ref / K_target)
        return float(D_target)
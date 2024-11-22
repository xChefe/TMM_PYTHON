import numpy as np
import matplotlib.pyplot as plt

# Função para criar a matriz de interface
def matriz_interface(k_i, k_j):
    return np.array([
        [(k_i + k_j) / (2 * k_i), (k_i - k_j) / (2 * k_i)],
        [(k_i - k_j) / (2 * k_i), (k_i + k_j) / (2 * k_i)]
    ])

# Função para criar a matriz de propagação
def matriz_propagacao(k, delta_z):
    return np.array([
        [np.exp(1j * k * delta_z), 0],
        [0, np.exp(-1j * k * delta_z)]
    ])

# Função para calcular os números de onda em cada ponto
def calcular_numeros_de_onda(E, potencial, m, hbar):
    k_values = []
    for U in potencial:
        if E > U:
            k = np.sqrt(2 * m * (E - U)) / hbar
        else:
            k = 1e-10  # Evitar divisão por zero em regiões onde E <= U
        k_values.append(k)
    return np.array(k_values)

# Função para definir o potencial de dupla barreira
def potencial_dupla_barreira(y_values, V0, L, d):
    """
    Define o potencial de duas barreiras:
    - Cada barreira tem altura V0 e largura L
    - As barreiras estão separadas por uma distância d
    
    Args:
        y_values (np.ndarray): Array de valores em y onde o potencial será calculado.
        V0 (float): Altura da barreira (em eV).
        L (float): Largura de cada barreira (em nm).
        d (float): Distância entre os centros das duas barreiras (em nm).

    Returns:
        np.ndarray: Potencial calculado em cada valor de y.
    """
    # Define o centro de cada barreira
    center1 = -d / 2
    center2 = d / 2

    # Potencial da primeira barreira
    barrier1 = np.where(np.abs(y_values - center1) <= L / 2, V0, 0)

    # Potencial da segunda barreira
    barrier2 = np.where(np.abs(y_values - center2) <= L / 2, V0, 0)

    # Soma dos potenciais das duas barreiras
    return barrier1 + barrier2

# Função para discretizar o potencial em retângulos
def discretizar_potencial(y_values, potencial, n_retas):
    y_discreto = np.linspace(y_values.min(), y_values.max(), n_retas)
    delta_z = y_discreto[1] - y_discreto[0]  # Largura de cada região
    potencial_discreto = np.interp(y_discreto, y_values, potencial)  # Aproximação
    return y_discreto, potencial_discreto, delta_z

# Parâmetros físicos ajustados para nm e eV
m = 9.11e-31  # Massa do elétron em kg
hbar = 1.055e-34  # Constante de Planck reduzida em J·s
eV_to_J = 1.6e-19  # Conversão de eV para Joules
nm_to_m = 1e-9  # Conversão de nanômetros para metros

# Parâmetros do potencial
V0 = 1  # Altura das barreiras em eV
L = 2.5  # Largura das barreiras em nm
d = 5  # Distância entre os centros das barreiras em nm

# Espaço contínuo para y (em nm)
y_values = np.linspace(-10, 10, 1000)  # Coordenadas y em nm
potencial = potencial_dupla_barreira(y_values, V0, L, d)  # Potencial contínuo em eV

# Discretizar o potencial
n_retas = 50  # Número de retângulos
y_discreto, potencial_discreto, delta_z = discretizar_potencial(y_values, potencial, n_retas)
delta_z = delta_z * nm_to_m  # Convertendo para metros

# Faixa de energias para o gráfico (em eV)
E_values = np.linspace(1.003, 1.075, 500)  # Energia variando de 0.01 eV a 10 eV
T_values = []  # Lista para armazenar os coeficientes de transmissão

# Cálculo do coeficiente de transmissão para cada energia
for E in E_values:
    E_J = E * eV_to_J  # Converter energia para Joules
    k_values = calcular_numeros_de_onda(E_J, potencial_discreto * eV_to_J, m, hbar)

    # Construção da matriz de transferência total
    M_total = np.eye(2)  # Matriz identidade inicial
    for i in range(len(potencial_discreto) - 1):
        M_interface = matriz_interface(k_values[i], k_values[i + 1])
        M_propag = matriz_propagacao(k_values[i + 1], delta_z)
        M_total = M_propag @ M_interface @ M_total

    # Coeficiente de transmissão
    T = 1 / np.abs(M_total[1, 1])**2
    T_values.append(T)

plt.figure(figsize=(8, 5))
plt.plot(E_values, np.clip(T_values, 0, 1), label='Coeficiente de Transmissão (T)')
plt.xlabel('Energia (eV)')
plt.ylabel('Coeficiente de Transmissão (T)')
plt.title('Transmissão vs Energia')
plt.grid(True)
plt.legend()
plt.show()

# Plotagem do potencial contínuo e discretizado
plt.figure(figsize=(8, 5))
plt.plot(y_values, potencial, label='Potencial Contínuo (eV)', lw=2)
plt.bar(y_discreto, potencial_discreto, width=delta_z / nm_to_m, alpha=0.5, label='Potencial Discretizado (eV)')
plt.xlabel('Posição (nm)')
plt.ylabel('Potencial (eV)')
plt.title('Potencial Contínuo e Discretizado')
plt.grid(True)
plt.legend()
plt.show()

import numpy as np

def get_air_properties(T_film: float) -> tuple:
    """Interpolates kinematic viscosity (nu) and thermal conductivity (k_f) of air."""
    T_ref = np.array([250, 300, 350, 400, 450, 500])
    nu_ref = np.array([1.14e-5, 1.58e-5, 2.09e-5, 2.64e-5, 3.23e-5, 3.86e-5])
    k_ref = np.array([0.0223, 0.0263, 0.0300, 0.0338, 0.0373, 0.0407])
    
    nu = np.interp(T_film, T_ref, nu_ref)
    k_f = np.interp(T_film, T_ref, k_ref)
    return nu, k_f

def calculate_teertstra_reynolds(U_app: float, b: float, t: float, nu: float) -> float:
    """Calculates channel Reynolds number based on Teertstra (2000)."""
    U_ch = U_app * ((2*b + 3*t) / (2*b))
    return (U_ch * b) / nu

def calculate_teertstra_nusselt(Q_total: float, b: float, A_wetted: float, k_f: float, T_wall_avg: float, T_in: float) -> float:
    """Calculates the Nusselt number based on Teertstra (2000)."""
    delta_T = T_wall_avg - T_in
    return (Q_total * b) / (A_wetted * k_f * delta_T)
import numpy as np

def calculate_crlb(anchors, target_pos, ple, sigma, samples):
    """Calculates the Cramer-Rao Lower Bound (CRLB) in meters."""
    if sigma <= 0:
        return 0.0
        
    effective_sigma = sigma / np.sqrt(samples)
    K = (10 * ple) / (effective_sigma * np.log(10))
    
    FIM = np.zeros((2, 2))
    x, y = target_pos
    
    for ax, ay in anchors:
        dx = x - ax
        dy = y - ay
        d_sq = max(dx**2 + dy**2, 1e-12) 
        
        coeff = (K / d_sq)**2
        
        FIM[0, 0] += coeff * (dx**2)
        FIM[1, 1] += coeff * (dy**2)
        FIM[0, 1] += coeff * (dx * dy)
        FIM[1, 0] += coeff * (dx * dy)
        
    try:
        crlb_matrix = np.linalg.inv(FIM)
        return np.sqrt(max(np.trace(crlb_matrix), 0.0))
    except np.linalg.LinAlgError:
        return float('inf')
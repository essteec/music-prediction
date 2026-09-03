"""
Shared utilities for similarity ablation studies and impact classification.
Single source of truth for decision rules across Audio, Lyric, and Metadata LOGO ablations.
"""

def classify_impact(ov10: float, v_delta: float, a_delta: float) -> str:
    """
    Rigorously classify impact of removing a feature group:
    - v_delta > 0: removing feature increased mood error (worse vibe match -> feature was helpful).
    - a_delta < 0: removing feature decreased artist agreement (worse signature match -> feature was helpful).
    """
    # 1. Essential / High Value Signal (Keep)
    if v_delta >= 0.005 or a_delta <= -1.0:
        return "Essential Signal (Keep - High Value)"
    
    # 2. Beneficial Signal (Keep)
    elif (v_delta >= 0.001 or a_delta < -0.2) and ov10 < 90.0:
        return "Beneficial Signal (Keep - Quality Drop)"
    
    # 3. Moderate Signal (Keep)
    elif (v_delta > 0.0002 or a_delta < -0.1) and ov10 < 95.0:
        return "Moderate Signal (Keep)"
    
    # 4. Redundant Signal (Negligible Unique Value)
    elif ov10 >= 95.0 and abs(v_delta) < 0.001 and abs(a_delta) < 0.2:
        return "Redundant (Negligible Unique Value)"
    
    # 5. Drift without Improvement (Drop Candidate)
    elif v_delta <= 0.0002 and a_delta >= -0.05:
        if ov10 < 80.0:
            return "Distinct but Divergent / Noisy (Drop)"
        else:
            return "Marginal / Neutral (Drop Candidate)"
    
    # 6. Fallback
    else:
        return "Marginal / Minor Signal"

"""
Unified Post-processing Configuration
Ensures consistent post-processing across Chemprop and Attention models for fair comparison
"""

# Organ thresholds for converting Biodistribution predictions to Delivery_target classifications
# These thresholds represent the minimum biodistribution percentage to consider an organ as "targeted"
# Values should be tuned based on validation data or domain knowledge
ORGAN_THRESHOLDS_DEFAULT = {
    'liver': 0.30,        # Liver: 30% or above biodistribution
    'lung': 0.25,         # Lung: 25% or above
    'spleen': 0.20,       # Spleen: 20% or above
    'muscle': 0.10,       # Muscle: 10% or above (often lower due to injection site)
    'lymph_nodes': 0.20,  # Lymph nodes: 20% or above
    'heart': 0.15,        # Heart: 15% or above
    'kidney': 0.15,       # Kidney: 15% or above
}

# Alternative threshold strategies for experimentation
THRESHOLD_STRATEGIES = {
    'default': ORGAN_THRESHOLDS_DEFAULT,
    
    'conservative': {  # Higher thresholds - fewer organs classified as "targeted"
        'liver': 0.35,
        'lung': 0.30,
        'spleen': 0.25,
        'muscle': 0.15,
        'lymph_nodes': 0.25,
        'heart': 0.20,
        'kidney': 0.20,
    },
    
    'permissive': {  # Lower thresholds - more organs classified as "targeted"
        'liver': 0.25,
        'lung': 0.20,
        'spleen': 0.15,
        'muscle': 0.08,
        'lymph_nodes': 0.15,
        'heart': 0.10,
        'kidney': 0.10,
    },
}


def get_organ_thresholds(strategy='default'):
    """
    Get organ threshold configuration for Biodistribution->Delivery_target conversion
    
    Args:
        strategy (str): Threshold strategy name
            - 'default': Balanced thresholds
            - 'conservative': Higher thresholds (stricter targeting criteria)
            - 'permissive': Lower thresholds (more lenient targeting criteria)
    
    Returns:
        dict: Mapping of organ names to threshold values
    
    Raises:
        ValueError: If strategy name is not recognized
    """
    if strategy not in THRESHOLD_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Available: {list(THRESHOLD_STRATEGIES.keys())}"
        )
    
    return THRESHOLD_STRATEGIES[strategy].copy()


def get_available_strategies():
    """
    Get list of available threshold strategies
    
    Returns:
        list: Strategy names
    """
    return list(THRESHOLD_STRATEGIES.keys())


def print_threshold_info(strategy='default'):
    """
    Print threshold information for a given strategy
    
    Args:
        strategy (str): Threshold strategy name
    """
    thresholds = get_organ_thresholds(strategy)
    
    print(f"\n{'='*60}")
    print(f"Threshold Strategy: {strategy}")
    print(f"{'='*60}")
    
    for organ, threshold in sorted(thresholds.items()):
        print(f"  {organ:15s}: {threshold:.2%}")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # Demo: Print all strategies
    for strategy in get_available_strategies():
        print_threshold_info(strategy)
"""Constraint definitions and loading utilities."""

from typing import List, Tuple, Optional
import json

def load_constraints_from_json(filepath: str) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Load must-sit and avoid pairs from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    must_sit = [tuple(pair) for pair in data.get('must_sit_pairs', [])]
    avoid = [tuple(pair) for pair in data.get('avoid_pairs', [])]
    
    return must_sit, avoid

def generate_realistic_constraints(num_guests: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Generate realistic constraints for testing."""
    # Families sit together
    families = [
        [(0,1,2), (3,4)],  # Family A: parents + child, Family B: couple
        [(5,6)],            # Couple
    ]
    
    must_sit = []
    for family in families:
        for i in range(len(family)):
            for j in range(i+1, len(family)):
                must_sit.append((family[i], family[j]))
    
    # Conflicts (people who don't get along)
    avoid = [(0, 5), (1, 6), (2, 3)]
    
    return must_sit, avoid
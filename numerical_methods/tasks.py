import numpy as np
from math import pi, exp

TASKS = {
    '4': {
        'a': 0,
        'b': pi,
        'f': lambda x: np.sin(x),
        'u_exact': lambda x: np.sin(x),
        'order': 4,
        'boundary_conditions': '4th'
    },
    '5': {
        'a': 0,
        'b': 1,
        'f': lambda x: np.exp(x),
        'u_exact': lambda x: np.exp(x) - 1 - x - x**2/2,
        'order': 5,
        'boundary_conditions': '5th'
    },
    '6': {
        'a': 0,
        'b': 1,
        'f': lambda x: 0*x,
        'u_exact': lambda x: 0*x,
        'order': 6,
        'boundary_conditions': '6th'
    }
} 
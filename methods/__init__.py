"""
MCDM Methods Module
Contains implementations of various Multi-Criteria Decision Making methods.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import methods.ahp as ahp
import methods.bwm as bwm
import methods.entropy as entropy
import methods.critic as critic
import methods.wsm as wsm
import methods.wpm as wpm
import methods.waspas as waspas
import methods.topsis as topsis
import methods.vikor as vikor
import methods.promethee as promethee
import methods.electre as electre
import methods.fuzzy_topsis as fuzzy_topsis

__all__ = [
    'ahp',
    'bwm',
    'entropy',
    'critic',
    'wsm',
    'wpm',
    'waspas',
    'topsis',
    'vikor',
    'promethee',
    'electre',
    'fuzzy_topsis'
]

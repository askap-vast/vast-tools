'''
-------------------
VAST tools module.
-------------------

This module is a collection of useful scripts,
functions and classes for use within the VAST project.

'''

__author__ = 'Dougal Dobie, Adam Stewart'
__version__ = "2.0.0-dev"


RELEASED_EPOCHS = {
    "0": "00",  # RACS, needs check that it exists, not part of VAST release
    "1": "01",
    "2": "02",
    "3x": "03x",
    "4x": "04x",
    "5x": "05x",
    "6x": "06x",
    "7x": "07x",
    "8": "08",
    "9": "09",
    "10x": "10x",
    "11x": "11x",
    "12": "12",
    "13": "13",
}

OBSERVED_EPOCHS = {
    "14": "14",
    "17": "17",
    "18": "18",
    "19": "19"
}


ALLOWED_PLANETS = [
    'mercury',
    'venus',
    'mars',
    'jupiter',
    'saturn',
    'uranus',
    'neptune',
    'sun',
    'moon'
]

'''
-------------------
VAST tools module.
-------------------

This module is a collection of useful scripts,
functions and classes for use within the VAST project.

'''

__author__ = 'Dougal Dobie, Adam Stewart'
__version__ = "3.1.1-dev"


RELEASED_EPOCHS = {
    "0": "00",  # RACS-low, needs check that it exists, not part of VAST release
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
    "14": "14",  # RACS-mid, needs check that it exists, not part of VAST release
    "17": "17",
    "18": "18",
    "19": "19",
    "20": "20",
    "21": "21",
    "22": "22",
    "23": "23",
    "24": "24",
    "25": "25",
    "26": "26",
    "27": "27",
    "28": "28",  # RACS-high, needs check that it exists, not part of VAST release
    "29": "29",  # RACS-low-2, needs check that it exists, not part of VAST release
    "30": "30",
    "31": "31",
    "32": "32",
    "33": "33",
    "34": "34",
    "35": "35",
    "36": "36",
    "37": "37",
    "38": "38",
    "39": "39",
    "40": "40",
    "41": "41",
    "42": "42",
    "43": "43",
    "44": "44",
    "45": "45",
    "46": "46",
    "47": "47",
    "48": "48",
    "49": "49",
    "50": "50",
    "51": "51",
    "52": "52",
    "53": "53",
    "54": "54",
    "55": "55",
    "56": "56",
    "57": "57",
    "58": "58",
    "59": "59",
    "60": "60",
    "61": "61",
    "62": "62",
    "63": "63",
    "64": "64",
    "65": "65",
    "66": "66",
    "67": "67",
    "68": "68",
    "69": "69"
}

OBSERVED_EPOCHS = {
}

BASE_EPOCHS = {
    "RACS": ['0', '14', '28'],
    "VAST": ['1', '18', '24', '40']
}

RACS_EPOCHS = [
    '0',
    '14',
    '28',
    '29'

]

P1_EPOCHS = [
    '1',
    '2',
    '3x',
    '4x',
    '5x',
    '6x',
    '7x',
    '8',
    '9',
    '10x',
    '11x',
    '12',
    '13'
]

P2_EPOCHS = [
    '17',
    '18',
    '19',
    '20',
    '21'
]

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

FREQ_CONVERSION = {
    864.5: 887.5,
    1272.5: 1367.5,
    1655.5: 1655.5
}

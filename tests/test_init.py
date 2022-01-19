import pytest
from vasttools import RELEASED_EPOCHS, OBSERVED_EPOCHS


def test_epoch_dicts():
    """
    Check that there is no overlapy between the OBSERVED_EPOCHS
    and RELEASED_EPOCHS dictionaries

    Args:
        None

    Returns:
        None
    """

    intersection = []

    for epoch in RELEASED_EPOCHS.keys():
        if epoch in OBSERVED_EPOCHS.keys():
            intersection.append(epoch)

    assert intersection == []

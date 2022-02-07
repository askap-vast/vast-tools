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

    # check intersection result is an empty set.
    assert RELEASED_EPOCHS.keys() & OBSERVED_EPOCHS.keys() == set()

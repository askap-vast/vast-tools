import pathlib
import pandas as pd
import pytest

from pytest_mock import mocker  # noqa: F401
from typing import Union

from vasttools.moc import VASTMOCS


@pytest.fixture
def vast_tools_moc() -> VASTMOCS:
    """
    Returns an VASTMOCS instance.

    Returns:
        VASTMOCS instance.
    """
    vast_mocs = VASTMOCS()
    return vast_mocs


@pytest.fixture
def epoch_moc_filename() -> str:
    """
    Obtains the assumed epoch MOC file name.

    Returns
        Epoch MOC file name from internal _get_values.
    """
    def _get_values(epoch: str) -> str:
        """
        Workhorse to get the actual filename.

        Args:
            epoch: The epoch to include in the string.

        Returns:
            The expected filename.
        """
        if epoch.endswith('x'):
            value = int(epoch[:-1])
            zero_padded_epoch = f'{value:02d}x'
        else:
            value = int(epoch)
            zero_padded_epoch = f'{value:02d}x'

        assumed_filename = f'VAST_PILOT_EPOCH{zero_padded_epoch}.moc.fits'

        return assumed_filename

    return _get_values


@pytest.fixture
def tile_moc_filename() -> str:
    """
    Obtains the assumed tile MOC file name.

    Returns
        Tile MOC file name from internal _get_values.
    """
    def _get_values(field: str) -> str:
        """
        Workhorse to get the actual filename.

        Args:
            field: The field to include in the filename.

        Returns:
            The expected filename.
        """
        assumed_filename = f'{field}.EPOCH01.I.moc.fits'

        return assumed_filename

    return _get_values


@pytest.fixture
def field_moc_filename() -> str:
    """
    Obtains the assumed field MOC file name.

    Returns
        Field MOC file name from internal _get_values.
    """
    def _get_values(field: Union[str, int]) -> str:
        """
        Workhorse to get the actual filename.

        Args:
            field: The field to include in the filename.

        Returns:
            The expected filename.
        """
        assumed_filename = f'VAST_PILOT_FIELD_{field}.fits'

        return assumed_filename

    return _get_values


def test_moc_load_pilot_stmoc(vast_tools_moc: VASTMOCS, mocker) -> None:
    """
    Test the loading of the STMOC

    Args:
        vast_tools_moc: Pytest fixture of VASTMOCS instance.
        mocker: The pytest_mock mocker object.

    Returns:
        None
    """
    filename = 'VAST_PILOT.stmoc.fits'
    mock_path = f'/path/to/{filename}'

    stmoc_mocker = mocker.patch('mocpy.STMOC.from_fits', return_value=True)
    importlib_mocker = mocker.patch(
        'importlib.resources.path',
        return_value=pathlib.Path(mock_path)
    )

    result = vast_tools_moc.load_pilot_stmoc()

    importlib_mocker.assert_called_once_with('vasttools.data.mocs', filename)
    stmoc_mocker.assert_called_once_with(importlib_mocker.return_value)
    assert result is True


def test_moc_load_pilot_epoch_moc_str(
    vast_tools_moc: VASTMOCS,
    epoch_moc_filename: str,
    mocker
) -> None:
    """
    Test the loading of the pilot epoch MOC.

    Args:
        vast_tools_moc: Pytest fixture of VASTMOCS instance.
        epoch_moc_filename: Pytest fixture that returns the string filename.
        mocker: The pytest_mock mocker object.

    Returns:
        None
    """
    epoch = '5x'
    filename = epoch_moc_filename(epoch)
    mock_path = f'/path/to/{filename}'

    moc_mocker = mocker.patch('mocpy.MOC.from_fits', return_value=-99)
    importlib_mocker = mocker.patch(
        'importlib.resources.path',
        return_value=pathlib.Path(mock_path)
    )

    result = vast_tools_moc.load_pilot_epoch_moc(epoch)

    importlib_mocker.assert_called_once_with('vasttools.data.mocs', filename)
    moc_mocker.assert_called_once_with(importlib_mocker.return_value)
    assert result == -99


def test_moc_load_pilot_epoch_moc_fail(vast_tools_moc: VASTMOCS) -> None:
    """
    Test the failure of loading the epoch MOC.

    Args:
        vast_tools_moc: Pytest fixture of VASTMOCS instance.

    Returns:
        None
    """
    epoch = '99z'

    with pytest.raises(Exception) as excinfo:
        vast_tools_moc.load_pilot_epoch_moc(epoch)

    assert str(excinfo.value) == f"EPOCH {epoch} not recognised"


@pytest.mark.parametrize("field", ['1', 1])
def test_moc_load_pilot_field_moc(
    field: Union[str, int],
    vast_tools_moc: VASTMOCS,
    field_moc_filename: str,
    mocker
) -> None:
    """
    Test the loading of the pilot field MOC. Tests field entered as a
    string and int.

    Args:
        field: Entered field from the parametrize.
        vast_tools_moc: Pytest fixture of VASTMOCS instance.
        field_moc_filename: Pytest fixture that returns the string filename.
        mocker: The pytest_mock mocker object.

    Returns:
        None
    """
    filename = field_moc_filename(field)
    mock_path = f'/path/to/{filename}'

    moc_mocker = mocker.patch('mocpy.MOC.from_fits', return_value=-99)
    importlib_mocker = mocker.patch(
        'importlib.resources.path',
        return_value=pathlib.Path(mock_path)
    )

    result = vast_tools_moc.load_pilot_field_moc(field)

    importlib_mocker.assert_called_once_with('vasttools.data.mocs', filename)
    moc_mocker.assert_called_once_with(importlib_mocker.return_value)
    assert result == -99


def test_moc_load_pilot_field_moc_fail(vast_tools_moc: VASTMOCS) -> None:
    """
    Test the failure of loading field MOC

    Args:
        vast_tools_moc: Pytest fixture of VASTMOCS instance.

    Returns:
        None
    """
    field = '99'

    with pytest.raises(ValueError) as excinfo:
        vast_tools_moc.load_pilot_field_moc(field)


@pytest.mark.parametrize("itype", ['COMBINED', 'TILES'])
def test_moc_load_pilot_tile_moc(
    itype: str,
    vast_tools_moc: VASTMOCS,
    tile_moc_filename: str,
    mocker
) -> None:
    """
    Test the loading of the pilot tile MOC. Tests both COMBINED and TILE
    tiles.

    Args:
        itype: COMBINED or TILES.
        vast_tools_moc: Pytest fixture of VASTMOCS instance.
        tile_moc_filename: Pytest fixture that returns the string filename.
        mocker: The pytest_mock mocker object.

    Returns:
        None
    """
    field = 'VAST_2053+00A'
    filename = tile_moc_filename(field)
    mock_path = f'/path/to/{itype}/{filename}'

    moc_mocker = mocker.patch('mocpy.MOC.from_fits', return_value=-99)
    importlib_mocker = mocker.patch(
        'importlib.resources.path',
        return_value=pathlib.Path(mock_path)
    )
    # need to also mock the loading of the field_centres
    fields_mocker = mocker.patch(
        'vasttools.moc.load_field_centres',
        return_value=pd.DataFrame(
            data={'field': [field]}
        )
    )

    result = vast_tools_moc.load_pilot_tile_moc(field, itype)

    importlib_mocker.assert_called_once_with(
        f'vasttools.data.mocs.{itype}',
        filename
    )
    moc_mocker.assert_called_once_with(importlib_mocker.return_value)
    assert result == -99


def test_moc_load_pilot_tile_moc_type_fail(vast_tools_moc: VASTMOCS) -> None:
    """
    Test the failure of loading of the tile MOC by providing a wrong type.

    Args:
        vast_tools_moc: Pytest fixture of VASTMOCS instance.

    Returns:
        None
    """
    field = 'VAST_2053+00A'
    itype = 'NOTATYPE'
    with pytest.raises(Exception) as excinfo:
        vast_tools_moc.load_pilot_tile_moc(field, itype, )

    assert str(excinfo.value).startswith(
        "Image type not recognised. Valid entries are:"
    )


def test_moc_load_pilot_tile_moc_field_fail(
    vast_tools_moc: VASTMOCS,
    mocker
) -> None:
    """
    Test the failure of loading of the tile MOC by providing a wrong field.

    Args:
        vast_tools_moc: Pytest fixture of VASTMOCS instance.
        mocker: The pytest_mock mocker object.

    Returns:
        None
    """
    field = 'VAST_9999+99A'

    # need to also mock the loading of the field_centres
    fields_mocker = mocker.patch(
        'vasttools.moc.load_field_centres',
        return_value=pd.DataFrame(
            data={'field': ['VAST_2053+00A']}
        )
    )

    with pytest.raises(Exception) as excinfo:
        vast_tools_moc.load_pilot_tile_moc(field, )

    assert str(excinfo.value) == f"Field {field} not recognised"


def test_moc_query_vizier_vast_pilot(
    vast_tools_moc: VASTMOCS,
    mocker
) -> None:
    """
    Test the vizier MOC query. No call to vizier is actually made.

    Args:
        vast_tools_moc: Pytest fixture of VASTMOCS instance.
        mocker: The pytest_mock mocker object.

    Returns:
        None
    """
    pilot_moc_mocker = mocker.patch(
        'vasttools.moc.VASTMOCS.load_pilot_epoch_moc',
    )
    pilot_moc_mocker.return_value.query_vizier_table.return_value = -99

    importlib_mocker = mocker.patch(
        'importlib.resources.path',
        return_value=pathlib.Path('/a/path.fits')
    )

    vizier_table_id = '1111'
    maxrows = 1000

    result = vast_tools_moc.query_vizier_vast_pilot(
        vizier_table_id, max_rows=maxrows
    )

    pilot_moc_mocker.return_value.query_vizier_table.assert_called_once_with(
        vizier_table_id,
        max_rows=maxrows
    )
    assert result == -99

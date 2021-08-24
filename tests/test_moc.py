import os
import pathlib
import pandas as pd
import pytest

from pytest_mock import mocker

from vasttools.moc import VASTMOCS


@pytest.fixture
def vast_tools_moc():
    vast_mocs = VASTMOCS()
    return vast_mocs


@pytest.fixture
def epoch_moc_filename():
    def _get_values(epoch: str):
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
def tile_moc_filename():
    def _get_values(field: str, itype: str):

        assumed_filename = f'{field}.EPOCH01.I.moc.fits'

        return assumed_filename

    return _get_values


@pytest.fixture
def field_moc_filename():
    def _get_values(field):
        assumed_filename = f'VAST_PILOT_FIELD_{field}.fits'

        return assumed_filename

    return _get_values


def test_moc_load_pilot_stmoc(vast_tools_moc, mocker):
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
    assert result == True


def test_moc_load_pilot_epoch_moc_str(
    vast_tools_moc, epoch_moc_filename, mocker
):
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


def test_moc_load_pilot_epoch_moc_fail(vast_tools_moc):
    epoch = '99z'

    with pytest.raises(Exception) as excinfo:
        vast_tools_moc.load_pilot_epoch_moc(epoch)

    assert str(excinfo.value) == f"EPOCH {epoch} not recognised"


def test_moc_load_pilot_field_moc_str(
    vast_tools_moc, field_moc_filename, mocker
):
    field = '1'
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


def test_moc_load_pilot_field_moc_int(
    vast_tools_moc, field_moc_filename, mocker
):
    field = 1
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


def test_moc_load_pilot_field_moc_fail(vast_tools_moc):
    field = '99'

    with pytest.raises(ValueError) as excinfo:
        vast_tools_moc.load_pilot_field_moc(field)


def test_moc_load_pilot_tile_moc_combined(
    vast_tools_moc, tile_moc_filename, mocker
):
    field = 'VAST_2053+00A'
    itype = 'COMBINED'
    filename = tile_moc_filename(field, itype)
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
            data={'field': [field,]}
        )
    )

    result = vast_tools_moc.load_pilot_tile_moc(field, itype)

    importlib_mocker.assert_called_once_with(
        f'vasttools.data.mocs.{itype}',
        filename
    )
    moc_mocker.assert_called_once_with(importlib_mocker.return_value)
    assert result == -99


def test_moc_load_pilot_tile_moc_tiles(
    vast_tools_moc, tile_moc_filename, mocker
):
    field = 'VAST_2053+00A'
    itype = 'TILES'
    filename = tile_moc_filename(field, itype)
    mock_path = f'/path/to/{itype}/{filename}'

    moc_mocker = mocker.patch('mocpy.MOC.from_fits', return_value=True)
    importlib_mocker = mocker.patch(
        'importlib.resources.path',
        return_value=pathlib.Path(mock_path)
    )
    # need to also mock the loading of the field_centres
    fields_mocker = mocker.patch(
        'vasttools.moc.load_field_centres',
        return_value=pd.DataFrame(
            data={'field': [field,]}
        )
    )

    result = vast_tools_moc.load_pilot_tile_moc(field, itype)

    importlib_mocker.assert_called_once_with(
        f'vasttools.data.mocs.{itype}',
        filename
    )
    moc_mocker.assert_called_once_with(importlib_mocker.return_value)
    assert result == True


def test_moc_load_pilot_tile_moc_type_fail(vast_tools_moc):
    field = 'VAST_2053+00A'
    itype = 'NOTATYPE'
    with pytest.raises(Exception) as excinfo:
        vast_tools_moc.load_pilot_tile_moc(field, itype, )

    assert str(excinfo.value).startswith(
        "Image type not recognised. Valid entries are:"
    )


def test_moc_load_pilot_tile_moc_field_fail(vast_tools_moc, mocker):
    field = 'VAST_9999+99A'

    # need to also mock the loading of the field_centres
    fields_mocker = mocker.patch(
        'vasttools.moc.load_field_centres',
        return_value=pd.DataFrame(
            data={'field': ['VAST_2053+00A',]}
        )
    )

    with pytest.raises(Exception) as excinfo:
        vast_tools_moc.load_pilot_tile_moc(field, )

    assert str(excinfo.value) == f"Field {field} not recognised"


def test_moc_query_vizier_vast_pilot(vast_tools_moc, mocker):
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

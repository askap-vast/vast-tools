"""Simple class to interface and load the VAST MOCS.

Attributes:
    BASE_MOC_PATH (str): The base MOC path (str) in relation to the package.

"""

import pkg_resources
import os
from mocpy import MOC, STMOC, World2ScreenMPL
from vasttools.survey import RELEASED_EPOCHS, FIELD_CENTRES
from astropy.time import Time
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table


BASE_MOC_PATH = pkg_resources.resource_filename(
    __name__, "./data/mocs/"
)


class VASTMOCS(object):
    """
    Class to interface with VAST MOC files included in the package.
    """

    def __init__(self) -> None:
        '''
        Constructor method.

        Returns:
            None
        '''
        super(VASTMOCS, self).__init__()

    def load_pilot_stmoc(self) -> STMOC:
        """
        Load spacetime MOC

        Returns:
            STMOC of the VAST Pilot survey.
        """

        stmoc_path = os.path.join(
            BASE_MOC_PATH, 'VAST_PILOT.stmoc.fits'
        )

        stmoc = STMOC.from_fits(stmoc_path)

        return stmoc

    def load_pilot_epoch_moc(self, epoch: str = '1') -> MOC:
        """
        Load MOC corresponding to one epoch of the pilot survey.

        Args:
            epoch: Epoch to load.

        Returns:
            MOC of the requested epoch.

        Raises:
            Exception: Entered epoch is not recognised.
        """

        if epoch not in RELEASED_EPOCHS:
            raise Exception(
                "EPOCH {} not recognised".format(epoch)
            )

        epoch_str = RELEASED_EPOCHS[epoch]

        moc_path = os.path.join(
            BASE_MOC_PATH, 'VAST_PILOT_EPOCH{}.moc.fits'.format(
                epoch_str
            )
        )

        moc = MOC.from_fits(moc_path)

        return moc

    def load_pilot_field_moc(self, field: str) -> MOC:
        """
        Load MOCs corresponding to the VAST Pilot 'field', which is a
        collection of tiles.

        Args:
            field: Name of the VAST Pilot field requested.

        Returns:
            The field MOC.

        Raises:
            Exception: VAST Pilot field is not valid (1 - 6).
        """
        if isinstance(field, int):
            field = str(field)
        # While this could be an int it's left as string to be consistent
        # with the other loads.
        if field not in ['1', '2', '3', '4', '5', '6']:
            raise Exception(
                f"VAST Pilot field #{field} is not valid - valid fields"
                " are numbered 1 - 6."
            )

        moc_path = os.path.join(
            BASE_MOC_PATH, 'VAST_PILOT_FIELD_{}.fits'.format(field))

        moc = MOC.from_fits(moc_path)

        return moc

    def load_pilot_tile_moc(self, field: str, itype: str = 'COMBINED') -> MOC:
        """
        Load MOCs corresponding to pilot tile field.

        Args:
            field: The name of field requested.
            itype: Image type (COMBINED or TILES), defaults to 'COMBINED'.

        Returns:
            Tile MOC.

        Raises:
            Exception: Entered image type is not recognised
                ('COMBINED' or 'TILES').
            Exception: Entered field is not found.
        """

        types = ["COMBINED", "TILES"]

        itype = itype.upper()

        if itype not in types:
            raise Exception(
                "Image type not recognised. Valid entries are"
                " 'COMBINED' or 'TILES'."
            )

        if field not in FIELD_CENTRES.field.values:
            raise Exception(
                "Field {} not recognised".format(field)
            )

        moc_path = os.path.join(
            BASE_MOC_PATH, itype, '{}.EPOCH01.I.moc.fits'.format(
                field
            )
        )

        moc = MOC.from_fits(moc_path)

        return moc

    def query_vizier_vast_pilot(
        self, table_id: str, max_rows: int = 10000
    ) -> Table:
        """
        Query the Vizier service for sources within Pilot footprint.

        Args:
            table_id: Vizier ID of table to query.
            max_rows: Maximum rows to return, defaults to 10000.

        Returns:
            Astropy table of Vizier results.
        """

        moc = self.load_pilot_epoch_moc('1')

        viz_table = moc.query_vizier_table(table_id, max_rows=max_rows)

        return viz_table

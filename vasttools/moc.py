"""Simple class to interface and load the VAST MOCS.

Attributes:
    BASE_MOC_PATH (str): The base MOC path (str) in relation to the package.

"""
import importlib.resources

from astropy.table import Table
from mocpy import MOC, STMOC
from typing import Union

from vasttools import RELEASED_EPOCHS
from vasttools.survey import load_field_centres


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
        Load space and time MOC of all VAST Pilot observations.

        Returns:
            STMOC of the VAST Pilot survey. A `mocpy.STMOC` object.
        """
        stmoc_name = 'VAST_PILOT.stmoc.fits'

        with importlib.resources.path(
            "vasttools.data.mocs",
            stmoc_name
        ) as stmoc_path:
            stmoc_path = stmoc_path.resolve()

        stmoc = STMOC.from_fits(stmoc_path)

        return stmoc

    def load_pilot_epoch_moc(self, epoch: str = '1') -> MOC:
        """
        Load MOC corresponding to one epoch of the pilot survey.

        Args:
            epoch: Epoch to load as a string with no zero padding.
                E.g. '3x'.

        Returns:
            MOC of the requested epoch. A `mocpy.MOC` object.

        Raises:
            Exception: Entered epoch is not recognised.
        """

        if epoch not in RELEASED_EPOCHS:
            raise Exception(
                "EPOCH {} not recognised".format(epoch)
            )

        epoch_str = RELEASED_EPOCHS[epoch]

        moc_name = 'VAST_PILOT_EPOCH{}.moc.fits'.format(epoch_str)

        with importlib.resources.path(
            "vasttools.data.mocs",
            moc_name
        ) as moc_path:
            moc_path = moc_path.resolve()

        moc = MOC.from_fits(moc_path)

        return moc

    def load_pilot_field_moc(self, field: Union[str, int]) -> MOC:
        """
        Load MOCs corresponding to the VAST Pilot 'field', which is a
        collection of tiles.

        Enter as a string ranging from fields 1 – 6.

        Args:
            field: Name of the VAST Pilot field requested.

        Returns:
            The field MOC. A `mocpy.MOC` object.

        Raises:
            Exception: VAST Pilot field is not valid (1 - 6).
        """
        # While this could be an int it's left as string to be consistent
        # with the other loads.
        fields = ['1', '2', '3', '4', '5', '6']

        if isinstance(field, int):
            field = str(field)

        if field not in fields:
            raise ValueError(
                f"VAST Pilot field #{field} is not valid - valid fields"
                " are numbered {}.".format(", ".join(fields))
            )

        moc_name = f'VAST_PILOT_FIELD_{field}.fits'

        with importlib.resources.path(
            "vasttools.data.mocs",
            moc_name
        ) as moc_path:
            moc_path = moc_path.resolve()

        moc = MOC.from_fits(moc_path)

        return moc

    def load_pilot_tile_moc(self, field: str, itype: str = 'COMBINED') -> MOC:
        """
        Load the MOC corresponding to the requested pilot tile field.

        Args:
            field: The name of field requested. For example, 'VAST_0012-06A'.
            itype: Image type (COMBINED or TILES), defaults to 'COMBINED'.

        Returns:
            Tile MOC. A `mocpy.MOC` object.

        Raises:
            Exception: Entered image type is not recognised
                ('COMBINED' or 'TILES').
            Exception: Entered field is not found.
        """
        types = ["COMBINED", "TILES"]

        itype = itype.upper()

        if itype not in types:
            raise Exception(
                "Image type not recognised. Valid entries are:"
                " {}.".format(", ".join(types))
            )

        field_centres = load_field_centres()
        if field not in field_centres['field'].to_numpy():
            raise Exception(
                "Field {} not recognised".format(field)
            )

        moc_name = f'{field}.EPOCH01.I.moc.fits'

        with importlib.resources.path(
            f"vasttools.data.mocs.{itype}",
            moc_name
        ) as moc_path:
            moc_path = moc_path.resolve()

        moc = MOC.from_fits(moc_path)

        return moc

    def _load_pilot_footprint(self, order: int = 10) -> MOC:
        """
        Load the complete footprint of the pilot survey

        Args:
            order: MOC order to use (10 corresponds to a spatial res of 3.5')

        Returns:
            MOC containing the pilot survey footprint
        """
        # There are 6 unique 'fields' in the pilot survey hence these are looped over in turn to load
        for i in range(5):
            moc = self.load_pilot_field_moc(i + 1)
            if i == 0:
                pilot_moc = moc
            else:
                pilot_moc = pilot_moc.union(moc)

        return pilot_moc.degrade_to_order(order)

    def _load_full_survey_footprint(self, order: int = 10) -> MOC:
        """
        Load the complete footprint of the full survey

        Args:
            order: MOC order to use (10 corresponds to a spatial res of 3.5')

        Returns:
            MOC containing the full survey footprint
        """

        for i, subsurvey in enumerate(['EQUATORIAL', 'HIGHDEC', 'GALACTIC']):
            moc_name = f'VAST_{subsurvey}.moc.fits'

            with importlib.resources.path(
                "vasttools.data.mocs",
                moc_name
            ) as moc_path:
                moc_path = moc_path.resolve()

            moc = MOC.from_fits(moc_path)

            if i == 0:
                survey_moc = moc
            else:
                survey_moc = survey_moc.union(moc)

        return survey_moc.degrade_to_order(order)

    def load_survey_footprint(self, survey, order: int = 10) -> MOC:
        """
        Load the footprint of either the pilot or full VAST surveys

        Args:
            survey: Survey requested (can be "pilot or "full")
            order: MOC order to use (10 corresponds to a spatial res of 3.5')

        Returns:
            Survey footprint in MOC format
        """

        if survey not in ['pilot', 'full']:
            raise Exception(
                f"Survey must be either 'pilot' or 'full', not {survey}"
            )
        if survey == 'pilot':
            return self._load_pilot_footprint(order=order)
        elif survey == 'full':
            return self._load_full_survey_footprint(order=order)

    def query_vizier_vast_pilot(
        self,
        table_id: str,
        max_rows: int = 10000
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

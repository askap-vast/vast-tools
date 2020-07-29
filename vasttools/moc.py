# Simple class to interface and load the VAST mocs

import pkg_resources
import os
from mocpy import MOC, STMOC, World2ScreenMPL
from vasttools.survey import RELEASED_EPOCHS, FIELD_CENTRES
from astropy.time import Time
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord


BASE_MOC_PATH = pkg_resources.resource_filename(
        __name__, "./data/mocs/"
)


class VASTMOCS(object):
    """docstring for VastMocs"""
    def __init__(self):
        super(VASTMOCS, self).__init__()

    def load_pilot_stmoc(self):

        stmoc_path = os.path.join(
            BASE_MOC_PATH, 'VAST_PILOT.stmoc.fits'
        )

        stmoc = STMOC.from_fits(stmoc_path)

        return stmoc

    def load_pilot_epoch_moc(self, epoch):

        if epoch not in RELEASED_EPOCHS:
            raise Exception(
                "EPOCH {} not recongised".format(epoch)
            )

        epoch_str = RELEASED_EPOCHS[epoch]

        moc_path = os.path.join(
            BASE_MOC_PATH, 'VAST_PILOT_EPOCH{}.moc.fits'.format(
                epoch_str
            )
        )

        moc = MOC.from_fits(moc_path)

        return moc

    def load_pilot_field_moc(self, field, itype='COMBINED'):

        types = ["COMBINED", "TILES"]

        itype = itype.upper()

        if itype not in types:
            raise Exception(
                "Image type not recongised. Valid entries are"
                " 'COMBINED' or 'TILE'."
            )

        if field not in FIELD_CENTRES.field.values:
            raise Exception(
                "Field {} not recongised".format(field)
            )

        moc_path = os.path.join(
            BASE_MOC_PATH, itype, '{}.EPOCH01.I.moc.fits'.format(
                field
            )
        )

        moc = MOC.from_fits(moc_path)

        return moc

    def query_vizier_vast_pilot(self, table_id, max_rows=10000):

        moc = self.load_pilot_field_moc('1')

        viz_table = moc.query_vizier_table(table_id, max_rows=max_rows)

        return viz_table

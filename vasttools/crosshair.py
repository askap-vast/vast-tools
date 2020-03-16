"""This module adds a true crosshair marker to matplotlib.

============================== ===============================================
marker                         description
============================== ===============================================
`"c"`                          crosshair

Usage
-----

import matplotlib.pyplot as plt
import crosshair
plt.scatter(0,0, marker='c', s=100)
plt.show()

Notes
-----

I tried to stay as close to the style of `matplotlib/lib/markers.py`, so it can
easily implemented in mpl after further testing.

How to implement this in matplotlib via a module was inspired by:
https://stackoverflow.com/a/16655800/5064815

Be aware that for small sizes the crosshair looks like four dots or even a
circle.  This is due to the fact that in this case the linewidth is larger then
the length of the 'hairs' of the crosshair.  This is know and similar behaviour
is seen for other markers at small sizes.

Author
------
L. A. Boogaard (13/07/2017)

"""

__author__ = "L. A. Boogaard"
__version__ = '0.1'

from matplotlib.transforms import Affine2D
from matplotlib.path import Path
import matplotlib.markers
import matplotlib.lines

_crosshair_path = Path([(0.0, -0.5),  # center, bottom
                        (0.0, -0.25),  # center, q_bot
                        (-0.5, 0.0),  # left, center
                        (-0.25, 0.0),  # q_left, center
                        (0.0, 0.25),  # center, q_top
                        (0.0, 0.5),  # center, top
                        (0.25, 0.0),  # q_right, center
                        (0.5, 0.0)],  # right, center
                       [Path.MOVETO,
                        Path.LINETO,
                        Path.MOVETO,
                        Path.LINETO,
                        Path.MOVETO,
                        Path.LINETO,
                        Path.MOVETO,
                        Path.LINETO])


def _set_crosshair(self):
    self._transform = Affine2D().scale(1.0)
    self._snap_threshold = 1.0
    self._filled = False
    self._path = _crosshair_path


matplotlib.markers.MarkerStyle._set_crosshair = _set_crosshair
matplotlib.markers.MarkerStyle.markers['c'] = 'crosshair'
matplotlib.lines.Line2D.markers = matplotlib.markers.MarkerStyle.markers

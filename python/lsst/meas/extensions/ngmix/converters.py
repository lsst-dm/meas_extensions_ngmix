#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np

from ngmix import GMix

from lsst.geom import Point2D
from lsst.afw.geom.ellipses import Quadrupole, Ellipse
from lsst.shapelet import ShapeletFunction, MultiShapeletFunction, HERMITE


__all__ = ["convertGMixToMultiShapelet", "convertMultiShapeletToGMix"]


def convertGMixToMultiShapelet(gmix):
    """Create an LSST MultiShapeletFunction from an ngmix.GMix.

    Parameters
    ----------
    gmix : `ngmix.gmix`
        Input Gaussian mixture.

    Returns
    -------
    msf : `lsst.shapelet.MultiShapeletFunction`
        Output multi-shapelet function.
    """
    reshaped = gmix.get_full_pars().reshape(-1, 6)
    components = []
    for flux, y, x, iyy, ixy, ixx in reshaped:
        quad = Quadrupole(ixx, iyy, ixy)
        ellipse = Ellipse(quad, Point2D(x, y))
        # create a 0th order (gaussian) shapelet function.
        component = ShapeletFunction(0, HERMITE, ellipse)
        component.getCoefficients()[0] = flux/ShapeletFunction.FLUX_FACTOR
        components.append(component)
    return MultiShapeletFunction(components)


def convertMultiShapeletToGMix(msf, ignoreHighOrder=False):
    """Create an ngmix.GMix from an LSST MultiShapeletFunction.

    Parameters
    ----------
    msf : `lsst.shapelet.MultiShapeletFunction`
        Input multi-shapelet object.
    ignoreHighOrder : `bool`
        If True, ignore shapelet terms beyond the zeroth (Gaussian) term.

    Returns
    -------
    gmix : `ndarray.gmix`
        Output Gaussian mixture.

    Raises
    ------
    ValueError:
        Raised if ``msf`` includes components with order > 0 and
        ``ignoreHighOrder`` is `False`.
    """
    components = msf.getComponents()
    shaped = np.zeros((len(components), 6), dtype=np.float64)
    for i, component in enumerate(components):
        if not ignoreHighOrder and component.getOrder() != 0:
            raise ValueError("Component %d has higher-order shapelet terms" % i)
        shaped[i][0] = component.getCoefficients()[0]*ShapeletFunction.FLUX_FACTOR
        ellipse = component.getEllipse()
        center = ellipse.getCenter()
        shaped[i][1] = center.getY()
        shaped[i][2] = center.getX()
        quad = ellipse.getCore()
        shaped[i][3] = quad.getIyy()
        shaped[i][4] = quad.getIxy()
        shaped[i][5] = quad.getIxx()
    return GMix(pars=shaped.reshape(shaped.size))

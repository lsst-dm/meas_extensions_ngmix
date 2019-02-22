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
import unittest

import lsst.utils.tests

from lsst.geom import Point2I, Box2I
from lsst.afw.image import Image
from lsst.shapelet.tests import ShapeletTestCase
from lsst.shapelet import MultiShapeletFunction, ShapeletFunction, HERMITE

from lsst.meas.extensions.ngmix import convertMultiShapeletToGMix, convertGMixToMultiShapelet


class ConverterTestCase(ShapeletTestCase):
    """Tests for conversions between similar ngmix and LSST objects.
    """

    def testGMixToMultiShapelet(self):
        np.random.seed(5)  # value doesn't matter, but we want deterministic numbers
        msf1 = MultiShapeletFunction([self.makeRandomShapeletFunction(order=0) for n in range(4)])
        gmix1 = convertMultiShapeletToGMix(msf1)
        msf2 = convertGMixToMultiShapelet(gmix1)

        # Test that we've round-tripped through GMix
        self.compareMultiShapeletFunctions(msf1, msf2, simplify=False)

        # Test that LSST and ngmix make images that agree
        bbox = Box2I(Point2I(-20, -15), Point2I(20, 15))
        image1 = Image(bbox, dtype=np.float64)
        msf1.evaluate().addToImage(image1)
        image2 = Image(bbox, dtype=np.float64)
        image2.array[:, :] = gmix1.make_image(image2.array.shape)
        self.assertImagesAlmostEqual(image1, image2)

        # Should reject ShapeletFunctions with order > 0, unless ignoreHighOrder==True
        with self.assertRaises(ValueError):
            problematic = MultiShapeletFunction([self.makeRandomShapeletFunction(order=2) for n in range(4)])
            convertMultiShapeletToGMix(problematic)
        ok = convertMultiShapeletToGMix(problematic, ignoreHighOrder=True)
        truncated1 = convertGMixToMultiShapelet(ok)
        truncated2 = MultiShapeletFunction([ShapeletFunction(0, HERMITE, p.getEllipse(),
                                                             p.getCoefficients()[0:1])
                                            for p in problematic.getComponents()])
        self.compareMultiShapeletFunctions(truncated1, truncated2)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

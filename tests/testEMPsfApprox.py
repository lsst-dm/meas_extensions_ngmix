from builtins import range
# LSST Data Management System
# Copyright 2008-2015 AURA/LSST
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
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
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

import os
import numpy as np
import unittest
import itertools

import lsst.pex.exceptions as pexExceptions
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.detection as afwDetection
import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable
import lsst.meas.algorithms as measAlg
import lsst.meas.base as measBase
from lsst.meas.base.tests import AlgorithmTestCase

import lsst.meas.extensions.ngmix.EMPsfApprox

#   Create an array of size x size containing a 2D circular Gaussian of size sigma.  Normalized to 1.0


def makeGaussianArray(size, sigma, xc=None, yc=None):
    if xc == None:
        xc = (size-1)/2.0
    if yc == None:
        yc = (size-1)/2.0
    image = afwImage.ImageD(afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Point2I(size, size)))
    array = np.ndarray(shape=(size, size), dtype=np.float64)
    for yi, yv in enumerate(range(0, size)):
        for xi, xv in enumerate(range(0, size)):
            array[yi, xi] = np.exp(-0.5*((xv - xc)**2 + (yv - yc)**2)/sigma**2)
    array /= array.sum()
    return array

#   Run a measurement task which has previously been initialized on a single source


def runMeasure(task, schema, exposure):
    cat = afwTable.SourceCatalog(schema)
    source = cat.addNew()
    dettask = measAlg.SourceDetectionTask()

    # Suppress non-essential task output.
    dettask.log.setThreshold(dettask.log.WARN)

    # We are expecting this task to log an error. Suppress it, so that it
    # doesn't appear on the console or in logs, and incorrectly cause the user
    # to assume a failure.
    task.log.setThreshold(task.log.FATAL)
    footprints = dettask.detectFootprints(exposure, sigma=4.0).positive.getFootprints()
    source.setFootprint(footprints[0])
    task.run(exposure, cat)
    return source

#   make a Gaussian with one or two components.  Always square of dimensions size x size


def makePsf(size, sigma1, mult1, sigma2, mult2):
    array0 = makeGaussianArray(size, sigma1)
    array0 *= mult1
    array1 = makeGaussianArray(size, sigma2)
    array1 *= mult2
    kernel = lsst.afw.math.FixedKernel(lsst.afw.image.ImageD(array0 + array1))
    return measAlg.KernelPsf(kernel)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class testEMTestCase(AlgorithmTestCase, lsst.utils.tests.TestCase):
    """A test case for shape measurement"""

    def setUp(self):
        self.dataDir = os.path.join(lsst.utils.getPackageDir("meas_extensions_ngmix"), "tests", "data")
        self.algName = "meas_extensions_ngmix_EMPsfApprox"

    def makeConfig(self):
        config = measBase.SingleFrameMeasurementConfig()
        config.plugins = [self.algName]
        config.slots.centroid = None
        config.slots.apFlux = None
        config.slots.calibFlux = None
        config.slots.instFlux = None
        config.slots.modelFlux = None
        config.slots.psfFlux = None
        config.slots.shape = None
        return config

    def testUnexpectedError(self):
        msConfig = self.makeConfig()
        msConfig.plugins[self.algName].nGauss = 2
        #   This generates an unexpected exception in Erin's code
        msConfig.plugins[self.algName].nTries = 0
        schema = afwTable.SourceTable.makeMinimalSchema()
        task = measBase.SingleFrameMeasurementTask(schema=schema, config=msConfig)
        exposure = afwImage.ExposureF(os.path.join(self.dataDir, "exp.fits"))
        psf = makePsf(67, 4.0, .7, 12.0, .3)
        exposure.setPsf(psf)
        source = runMeasure(task, schema, exposure)
        self.assertEqual(source.get(self.algName + "_flag"), True)
        self.assertEqual(source.get(self.algName + "_flag_rangeError"), False)
        self.assertEqual(source.get(self.algName + "_flag_maxIters"), False)
        self.assertEqual(source.get(self.algName + "_flag_noPsf"), False)

    #   Test to be sure that we can catch the maximum iterations error
    def testMaxIter(self):
        msConfig = self.makeConfig()
        msConfig.plugins[self.algName].nGauss = 2
        msConfig.plugins[self.algName].tolerance = 1e-10
        #   we know the code can't fit this in one iteration
        msConfig.plugins[self.algName].maxIters = 1
        schema = afwTable.SourceTable.makeMinimalSchema()
        task = measBase.SingleFrameMeasurementTask(schema=schema, config=msConfig)
        exposure = afwImage.ExposureF(os.path.join(self.dataDir, "exp.fits"))
        psf = makePsf(67, 4.0, .7, 12.0, .3)
        exposure.setPsf(psf)
        source = runMeasure(task, schema, exposure)
        self.assertEqual(source.get(self.algName + "_flag"), True)
        self.assertEqual(source.get(self.algName + "_flag_rangeError"), False)
        self.assertEqual(source.get(self.algName + "_flag_maxIters"), True)
        self.assertEqual(source.get(self.algName + "_flag_noPsf"), False)

    #   Test to be sure that we can catch the missing psf error
    def testMissingPsf(self):
        msConfig = self.makeConfig()
        schema = afwTable.SourceTable.makeMinimalSchema()
        task = measBase.SingleFrameMeasurementTask(schema=schema, config=msConfig)
        exposure = afwImage.ExposureF(os.path.join(self.dataDir, "exp.fits"))
        #   Strip the psf
        exposure.setPsf(None)
        source = runMeasure(task, schema, exposure)
        self.assertEqual(source.get(self.algName + "_flag"), True)
        self.assertEqual(source.get(self.algName + "_flag_rangeError"), False)
        self.assertEqual(source.get(self.algName + "_flag_maxIters"), False)
        self.assertEqual(source.get(self.algName + "_flag_noPsf"), True)

    #   Test that the EmPsfApprox plugin produces a more or less reasonable result
    def testEMPlugin(self):
        msConfig = self.makeConfig()
        msConfig.plugins[self.algName].nGauss = 2
        schema = afwTable.SourceTable.makeMinimalSchema()
        task = measBase.SingleFrameMeasurementTask(schema=schema, config=msConfig)
        exposure = afwImage.ExposureF(os.path.join(self.dataDir, "exp.fits"))

        #   this is a double gaussian with a .7/.3 ratio of inner to outer
        #   we expect ixx = iyy = sigma*sigma
        psf = makePsf(67, 4.0, .7, sigma2=10.0, mult2=.3)
        exposure.setPsf(psf)
        source = runMeasure(task, schema, exposure)

        #   Be sure there were no failures
        self.assertEqual(source.get(self.algName + "_flag"), False)
        self.assertEqual(source.get(self.algName + "_flag_rangeError"), False)
        self.assertEqual(source.get(self.algName + "_flag_maxIters"), False)
        self.assertEqual(source.get(self.algName + "_flag_noPsf"), False)

        self.msfKey = lsst.shapelet.MultiShapeletFunctionKey(schema[self.algName],
                                                             lsst.shapelet.HERMITE)

        #   check the two component result to be sure it is close to the input PSF
        #   we don't control the order of EmPsfApprox, so order by size.
        msf = source.get(self.msfKey)
        components = msf.getComponents()
        self.assertEqual(len(components), 2)
        comp0 = components[0]
        comp1 = components[1]
        flux0 = comp0.getCoefficients()[0]
        flux1 = comp1.getCoefficients()[0]
        if flux0 < flux1:
            temp = comp1
            comp1 = comp0
            comp0 = temp
        #  We are not looking for really close matches in this unit test, which is why
        #  the tolerances are set rather large.  Really just a check that we are getting
        #  some kind of reasonable value for the fit.  A more quantitative test may be needed.
        self.assertClose(flux0/flux1, 7.0/3.0, rtol=.05)
        self.assertClose(comp0.getEllipse().getCore().getIxx(), 16.0, rtol=.05)
        self.assertClose(comp0.getEllipse().getCore().getIyy(), 16.0, rtol=.05)
        self.assertClose(comp0.getEllipse().getCore().getIxy(), 0.0, atol=.1)
        self.assertClose(comp1.getEllipse().getCore().getIxx(), 100.0, rtol=.05)
        self.assertClose(comp1.getEllipse().getCore().getIyy(), 100.0, rtol=.05)
        self.assertClose(comp1.getEllipse().getCore().getIxy(), 0.0, atol=.1)

class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

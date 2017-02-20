#!/usr/bin/env python
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
import numpy
import os
import unittest
import lsst.afw.image
import lsst.afw.table
import lsst.meas.base
from lsst.meas.base.tests import AlgorithmTestCase
import lsst.utils.tests as utilsTests

import lsst.meas.extensions.ngmix

#   See if modelfit can be imported
try:
    import lsst.meas.modelfit
    modelfit = True
except ImportError:
    modelfit = False


class testShapeTestCase(AlgorithmTestCase, lsst.utils.tests.TestCase):
    """A test case for shape measurement"""

    def setUp(self):
        self.dataDir = os.path.join(lsst.utils.getPackageDir("meas_extensions_ngmix"), "tests", "data")
        self.exposure = lsst.afw.image.ExposureF(os.path.join(self.dataDir, "exp.fits"))
        psfArray = self.exposure.getPsf().computeKernelImage().getArray()
        #   Add random noise to our noiseless psfs and instructed by Erin Sheldon
        noise = (numpy.random.random_sample(size=psfArray.shape) - .5) * .01 * psfArray.sum()
        kernel = lsst.afw.math.FixedKernel(lsst.afw.image.ImageD(psfArray + noise))
        self.exposure.setPsf(lsst.meas.algorithms.KernelPsf(kernel))
        self.algName = "meas_extensions_ngmix_LMSimpleShape"

    def tearDown(self):
        del self.dataDir
        del self.exposure

    #  Create a config to run shape measurer (algName) using psf approx (psfName)
    #  if a flux or centroid algorithm is needed, that can also be run
    def makeConfig(self, algName, psfName=None, psfModel=None, nGauss=None, centroid=None, psfFlux=None):
        config = lsst.meas.base.SingleFrameMeasurementConfig()
        plugins = [algName]
        if centroid is not None:
            plugins.append(centroid)
        if psfFlux is not None:
            plugins.append(psfFlux)
        if psfName is not None:
            plugins.append(psfName)
        config.plugins.names = plugins
        if psfName is not None:
            config.plugins[algName].psfName = psfName
        if psfModel is not None:
            config.plugins[psfName].sequence = [psfModel]
            config.plugins[algName].psfName = psfName + '_' + psfModel
        if nGauss is not None:
            config.plugins[psfName].nGauss = nGauss
        config.slots.centroid = centroid
        config.slots.apFlux = None
        config.slots.calibFlux = None
        config.slots.instFlux = None
        config.slots.modelFlux = None
        config.slots.psfFlux = psfFlux
        config.slots.shape = None
        return config

    #   Run algName (shape measurer) using the specified config
    def runShape(self, config, algName, exposure, model=None):
        if model is not None:
            config.plugins[algName].model = model
        schema = lsst.afw.table.SourceTable.makeMinimalSchema()
        task = lsst.meas.base.SingleFrameMeasurementTask(schema=schema, config=config)
        cat = lsst.afw.table.SourceCatalog(schema)
        # Run detection on the single-source image
        threshold = lsst.afw.detection.Threshold(5.0, lsst.afw.detection.Threshold.STDEV)
        fpSet = lsst.afw.detection.FootprintSet(exposure.getMaskedImage().getImage(), threshold)
        if len(fpSet.getFootprints()) > 1:
            raise RuntimeError("Threshold value results in multiple Footprints for a single object")
        if len(fpSet.getFootprints()) == 0:
            raise RuntimeError("Threshold value results in zero Footprints for object")
        source = cat.addNew()
        source.setFootprint(fpSet.getFootprints()[0])

        task.run(cat, exposure)
        if source.get(self.algName + "_flag"):
            return source, None
        self.msfKey = lsst.shapelet.MultiShapeletFunctionKey(source.getSchema()[algName],
                                                             lsst.shapelet.HERMITE)
        msf = source.get(self.msfKey)
        return source, msf

    #   This code runs the shape plugin allowing it to do its own psf estimation
    #   We use this for reference, as we don't have anything better.  It is good enough
    #   To test the plugin against different psfApprox algorithms.
    def runReferenceShape(self, exposure):
        config = self.makeConfig(self.algName)
        return self.runShape(config, self.algName, self.exposure, model='gauss')

    #   Test to be sure that we can catch the missing psf error
    def testMissingPsf(self):
        self.exposure.setPsf(None)
        config = self.makeConfig(self.algName)
        source, msf = self.runShape(config, self.algName, self.exposure, model='gauss')
        self.assertEqual(source.get(self.algName + "_flag"), True)
        self.assertEqual(source.get(self.algName + "_flag_noPsf"), True)

    #   Test to be sure that we can catch the max fev error
    def testMaxFev(self):
        config = self.makeConfig(self.algName)
        config.plugins[self.algName].maxfev = 10
        source, msf = self.runShape(config, self.algName, self.exposure, model='dev')
        self.assertEqual(source.get(self.algName + "_flag"), True)
        self.assertEqual(source.get(self.algName + "_flag_maxFev"), True)
        config.plugins["meas_extensions_ngmix_LMSimpleShape"].maxfev = 4000

    #   Run only if modelfit can be imported.
    #   Test the use of the Shape algorithm using the GeneralShapeletPsfApprox Gaussians
    #   Compare against the default moments which are produced by the default
    #   Can be fairly far off -- just want to be sure algorithms work together
    @unittest.skipUnless(modelfit, "Test using ShapeletPsfApprox was not run.")
    def testLMSimpleShapeSingleGaussian(self):
        source, msf1 = self.runReferenceShape(self.exposure)
        psfName = 'modelfit_GeneralShapeletPsfApprox'
        psfModel = 'SingleGaussian'
        config = self.makeConfig(self.algName, psfName=psfName, psfModel=psfModel,
                                 centroid='base_GaussianCentroid')
        source, msf2 = self.runShape(config, self.algName, self.exposure)
        moments1 = msf1.evaluate().computeMoments()
        moments2 = msf2.evaluate().computeMoments()
        self.assertEqual(source.get(self.algName + "_flag"), False)
        self.assertFloatsAlmostEqual(moments1.getCore().getIxx(), moments2.getCore().getIxx(), atol=.30)
        self.assertFloatsAlmostEqual(moments1.getCore().getIxy(), moments2.getCore().getIxy(), atol=.30)
        self.assertFloatsAlmostEqual(moments1.getCore().getIyy(), moments2.getCore().getIyy(), atol=.30)
        self.assertFloatsAlmostEqual(moments1.getCore().getIxx(), moments2.getCore().getIxx(), atol=.1)
        self.assertFloatsAlmostEqual(moments1.getCore().getIxy(), moments2.getCore().getIxy(), atol=.1)

    @unittest.skipUnless(modelfit, "Test using ShapeletPsfApprox was not run.")
    def testLMSimpleShapeDoubleGaussian(self):
        source, msf1 = self.runReferenceShape(self.exposure)
        psfName = 'modelfit_GeneralShapeletPsfApprox'
        psfModel = 'DoubleGaussian'
        config = self.makeConfig(self.algName, psfName=psfName, psfModel=psfModel,
                                 centroid='base_GaussianCentroid')
        source, msf2 = self.runShape(config, self.algName, self.exposure)
        moments1 = msf1.evaluate().computeMoments()
        moments2 = msf2.evaluate().computeMoments()
        self.assertEqual(source.get(self.algName + "_flag"), False)
        self.assertFloatsAlmostEqual(moments1.getCore().getIxx(), moments2.getCore().getIxx(), atol=.30)
        self.assertFloatsAlmostEqual(moments1.getCore().getIxy(), moments2.getCore().getIxy(), atol=.30)
        self.assertFloatsAlmostEqual(moments1.getCore().getIyy(), moments2.getCore().getIyy(), atol=.30)
        self.assertFloatsAlmostEqual(moments1.getCenter().getX(), moments2.getCenter().getX(), atol=.1)
        self.assertFloatsAlmostEqual(moments1.getCenter().getY(), moments2.getCenter().getY(), atol=.1)

    #   Just test to be sure this runs without errors
    def testLMSimpleShapeNGauss1(self):
        config = self.makeConfig(self.algName, psfName='meas_extensions_ngmix_EMPsfApprox', nGauss=2)
        source, msf = self.runShape(config, self.algName, self.exposure, model='gauss')
        self.assertEqual(source.get(self.algName + "_flag"), False)

    #   Just test to be sure this runs without errors
    def testLMSimpleShapeNGauss2(self):
        config = self.makeConfig(self.algName, psfName='meas_extensions_ngmix_EMPsfApprox', nGauss=3)
        source, msf = self.runShape(config, self.algName, self.exposure, model='gauss')
        self.assertEqual(source.get(self.algName + "_flag"), False)

    #   Just test to be sure this runs without errors
    def testLMSimpleShapeDev(self):
        config = self.makeConfig(self.algName)
        source, msf = self.runShape(config, self.algName, self.exposure, model='dev')
        self.assertEqual(source.get(self.algName + "_flag"), False)

    #   Just test to be sure this runs without errors
    def testLMSimpleShapeExp(self):
        config = self.makeConfig(self.algName)
        source, msf = self.runShape(config, self.algName, self.exposure, model='exp')
        self.assertEqual(source.get(self.algName + "_flag"), False)

class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

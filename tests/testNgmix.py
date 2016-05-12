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

import re, os, sys
import glob
import math
import numpy as np
import unittest
import itertools
import lsst.pex.exceptions as pexExceptions
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.meas.base as measBase
import lsst.meas.algorithms as measAlg
from lsst.meas.base.tests import AlgorithmTestCase
import lsst.meas.algorithms as algorithms
import lsst.utils.tests as utilsTests
import lsst.afw.detection as afwDetection
import lsst.afw.table as afwTable
import lsst.afw.geom as afwGeom
import lsst.afw.geom.ellipses as afwEll
import lsst.afw.coord as afwCoord
import lsst.afw.display.ds9 as ds9

import lsst.meas.extensions.ngmix.emPsfApprox

def makeGaussianImage(bbox, sigma, xc=0.0, yc=0.0):
    image = afwImage.ImageD(bbox)
    array = image.getArray()
    for yi, yv in enumerate(xrange(bbox.getBeginY(), bbox.getEndY())):
        for xi, xv in enumerate(xrange(bbox.getBeginX(), bbox.getEndX())):
            array[yi, xi] = numpy.exp(-0.5*((xv - xc)**2 + (yv - yc)**2)/sigma**2)
    array /= array.sum()
    return image

def makeGaussianArray(size, sigma, xc=0.0, yc=0.0):
    image = afwImage.ImageD(afwGeom.Box2I(afwGeom.Point2I(0,0), afwGeom.Point2I(size,size)))
    array = np.ndarray(shape = (size, size), dtype = np.float64)
    for yi, yv in enumerate(xrange(0, size)):
        for xi, xv in enumerate(xrange(0, size)):
            array[yi, xi] = np.exp(-0.5*((xv - xc)**2 + (yv - yc)**2)/sigma**2)
    array /= array.sum()
    return array

def makePluginAndCat(alg, name, control=None, metadata=False, centroid=None):
    if control == None:
        control=alg.ConfigClass()
    schema = afwTable.SourceTable.makeMinimalSchema()
    if centroid:
        schema.addField(centroid + "_x", type=float)
        schema.addField(centroid + "_y", type=float)
        schema.addField(centroid + "_flag", type='Flag')
        schema.getAliasMap().set("slot_Centroid", centroid)
    if metadata:
        plugin = alg(control, name, schema, dafBase.PropertySet())
    else:
        plugin = alg(control, name, schema)
    cat = afwTable.SourceCatalog(schema)
    if centroid:
        cat.defineCentroid(centroid)
    return plugin, cat

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class ShapeTestCase(AlgorithmTestCase):
    """A test case for shape measurement"""


    def testEMPlugin(self):
        """Test that we can instantiate and play with a measureShape"""

        algorithmName = "meas_extensions_ngmix_emPsfApprox"
        # perform the shape measurement
        msConfig = measBase.SingleFrameMeasurementConfig()
        msConfig.plugins = [algorithmName]
        msConfig.plugins["meas_extensions_ngmix_emPsfApprox"].nGauss = 2
        msConfig.plugins["meas_extensions_ngmix_emPsfApprox"].tolerance = 1e-4
        msConfig.plugins["meas_extensions_ngmix_emPsfApprox"].maxIter = 10000
        schema = afwTable.SourceTable.makeMinimalSchema()
        schema.addField("truth" + "_x", type=float)
        schema.addField("truth" + "_y", type=float)
        schema.addField("truth" + "_flag", type='Flag')
        msConfig.slots.centroid = None
        msConfig.slots.apFlux = None
        msConfig.slots.calibFlux = None
        msConfig.slots.instFlux = None
        msConfig.slots.modelFlux = None
        msConfig.slots.psfFlux = None
        msConfig.slots.shape = None
        task = measBase.SingleFrameMeasurementTask(schema=schema, config=msConfig)
        self.dataDir = os.path.join(os.getenv('MEAS_EXTENSIONS_NGMIX_DIR'), "tests", "data")
        exposure = afwImage.ExposureF(os.path.join(self.dataDir, "exp.fits"))
        array0 = makeGaussianArray(67, 4.0, 33, 33)
        array0 *= .7
        array1 = makeGaussianArray(67, 12.0, 33, 33)
        array1 *= .3
        kernel = lsst.afw.math.FixedKernel(lsst.afw.image.ImageD(array0 + array1))
        psf = lsst.meas.algorithms.KernelPsf(kernel)
        exposure.setPsf(psf)

        cat = afwTable.SourceCatalog(schema)
        #source.setFootprint(afwDetection.Footprint(afwGeom.Point2I(23, 34), width))
        source = cat.addNew()
        plugin = task.plugins['meas_extensions_ngmix_emPsfApprox']
        plugin.measure(source, exposure)
        self.msfKey = lsst.shapelet.MultiShapeletFunctionKey(schema["meas_extensions_ngmix_emPsfApprox"],
                                                             lsst.shapelet.HERMITE)
        msf = source.get(self.msfKey)
        eval0 = source.get(lsst.shapelet.ShapeletFunctionKey(schema["meas_extensions_ngmix_emPsfApprox_0"])).evaluate()
        eval1 = source.get(lsst.shapelet.ShapeletFunctionKey(schema["meas_extensions_ngmix_emPsfApprox_1"])).evaluate()
        if eval0.integrate() < eval1.integrate():
            temp = eval1
            eval1 = eval0
            eval0 = temp
        core0 = eval0.computeMoments().getCore()
        core1 = eval1.computeMoments().getCore()
        self.assertClose(eval0.integrate(), .7, rtol = .3)
        self.assertClose(core0.getIxx() + core0.getIyy(), 32, rtol = .3)
        self.assertClose(eval1.integrate(), .3, rtol = .3)
        self.assertClose(core1.getIxx() + core1.getIyy(), 288, rtol = .3)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(ShapeTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(exit = False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)

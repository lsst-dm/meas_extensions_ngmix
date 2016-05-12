#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2008-2016 AURA/LSST.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
"""
Definitions and registration of pure-Python plugins with trivial implementations,
and automatic plugin-from-algorithm calls for those implemented in C++.
"""
import numpy
import ngmix
from ngmix.bootstrap import EMRunner

import lsst.pex.exceptions
import lsst.afw.detection
import lsst.afw.geom
import lsst.afw.geom as afwGeom
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
import lsst.shapelet

from lsst.meas.base.pluginRegistry import register
from lsst.meas.base.sfm import SingleFramePluginConfig, SingleFramePlugin

__all__ = (
    "SingleFrameNgmixConfig", "SingleFrameNgmixPlugin"
)

class SingleFrameNgmixConfig(SingleFramePluginConfig):
    nGauss = lsst.pex.config.Field(dtype=int, default=1, optional=False,
                                  doc="Number of gaussians")
    maxIter = lsst.pex.config.Field(dtype=int, default=10000, optional=False,
                                  doc="maximum number of iterations")
    tolerance = lsst.pex.config.Field(dtype=float, default=1e-6, optional=False,
                                  doc="tolerance")

@register("meas_extensions_ngmix_emPsfApprox")
class SingleFrameNgmixPlugin(SingleFramePlugin):
    '''
    Algorithm to calculate the position of a centroid on the focal plane
    '''

    ConfigClass = SingleFrameNgmixConfig

    @classmethod
    def getExecutionOrder(cls):
        return cls.SHAPE_ORDER

    def __init__(self, config, name, schema, metadata):
        SingleFramePlugin.__init__(self, config, name, schema, metadata)

        self.failKey = schema.addField(name + '_flag', type="Flag", doc="Set to 1 for any fatal failure")
        self.iterKey = schema.addField(name + '_iterations', type=int, doc="number of iterations run")
        self.triesKey = schema.addField(name + '_tries', type=int, doc="number of tries")
        self.fdiffKey = schema.addField(name + '_fdiff', type=float, doc="fit difference")
        self.keys = []
        for i in range(config.nGauss):
            key = lsst.shapelet.ShapeletFunctionKey.addFields(schema,
                  "%s_%d"%(name, i), "ngmix EM gaussian", "pixels", "", 1)
            self.keys.append(key)
        self.msfKey = lsst.shapelet.MultiShapeletFunctionKey(self.keys)


    def measure(self, measRecord, exposure):

        psfImage = exposure.getPsf().computeKernelImage()
        psfArray = psfImage.getArray()
        psf_obs = ngmix.observation.Observation(psfArray)
        # Simple means one of the 6 parameter models

        ngauss = self.config.nGauss
        shape = exposure.getPsf().computeShape()
        Tguess = shape.getIxx() + shape.getIyy()

        ntry = 10
        em_pars={'maxiter':self.config.maxIter, 'tol':self.config.tolerance}
        runner=EMRunner(psf_obs, Tguess, ngauss, em_pars)
        runner.go(ntry=ntry)

        fitter=runner.get_fitter()
        res=fitter.get_result()

        measRecord.set(self.iterKey, res['numiter'])
        measRecord.set(self.fdiffKey, res['fdiff'])
        measRecord.set(self.triesKey, res['ntry'])
        if res['flags'] != 0:
           self.fail(measRecord)
        fitter.get_gmix()
        psf_pars = fitter.get_gmix().get_full_pars()

        for i in range(self.config.nGauss):
            pars = psf_pars[i*6:i*6+6]
            flux = pars[0]
            x = pars[1]
            y = pars[2]
            ixx = pars[5]
            iyy = pars[3]
            ixy = pars[4]
            order = 1
            quad = lsst.afw.geom.ellipses.Quadrupole(ixx, iyy, ixy)
            ellipse = lsst.afw.geom.ellipses.Ellipse(quad, lsst.afw.geom.Point2D(x,y))
            sf = lsst.shapelet.ShapeletFunction(order, lsst.shapelet.HERMITE, ellipse)
            sf.getCoefficients()[0] = flux/lsst.shapelet.ShapeletFunction.FLUX_FACTOR
            measRecord.set(self.keys[i], sf)

#        pfitter=ngmix.fitting.LMSimple(psf_obs,'gauss')
#        psf_pars=[0.0, 0.0, -0.03, 0.02, 4.0, 1.0]
#        # for simplicity, guess pars before pixelization
#        guess=[0.0, 0.0, -0.03, 0.02, 4.0, 1.0]
#        eps = .01
#        #guess[0] += urand(low=-eps,high=eps)
#        #guess[1] += urand(low=-eps,high=eps)
#        #guess[2] += urand(low=-eps, high=eps)
#        #guess[3] += urand(low=-eps, high=eps)
#        #guess[4] *= (1.0 + urand(low=-eps, high=eps))
#        #guess[5] *= (1.0 + urand(low=-eps, high=eps))
#
#        pfitter.go(guess)
#

    def fail(self, measRecord, error=None):
        measRecord.set(failFlag, True)

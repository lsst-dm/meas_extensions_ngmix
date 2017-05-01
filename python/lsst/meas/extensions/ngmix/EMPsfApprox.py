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
from builtins import range
import numpy
import ngmix
from ngmix.bootstrap import EMRunner
from ngmix.em import EM_RANGE_ERROR, EM_MAXITER
import lsst.pex.exceptions
import lsst.afw.detection
import lsst.afw.geom
import lsst.afw.geom as afwGeom
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
import lsst.shapelet

from lsst.meas.base.pluginRegistry import register
from lsst.meas.base.sfm import SingleFramePluginConfig, SingleFramePlugin
from lsst.meas.base import MeasurementError
from lsst.meas.base import FlagDefinition, FlagDefinitionList, FlagHandler

__all__ = ("SingleFrameEmPsfApproxConfig", "SingleFrameEmPsfApproxPlugin")


class SingleFrameEmPsfApproxConfig(SingleFramePluginConfig):
    """
    nGauss = 1, 2, or 3 is the number of Gaussian used in the fit.
    nTries, maxItters, and tolerance are inputs to the ngmix EMRunner.
    """

    nGauss = lsst.pex.config.Field(dtype=int, default=3, optional=False,
                                   doc="Number of gaussians")
    nTries = lsst.pex.config.Field(dtype=int, default=10, optional=False,
                                   doc="maximum number of tries with different guesses")
    maxIters = lsst.pex.config.Field(dtype=int, default=10000, optional=False,
                                     doc="maximum number of iterations")
    tolerance = lsst.pex.config.Field(dtype=float, default=1e-6, optional=False,
                                      doc="tolerance")


@register("meas_extensions_ngmix_EMPsfApprox")
class SingleFrameEmPsfApproxPlugin(SingleFramePlugin):
    """
    Plugin to do Psf modeling using the ngmix Expectation-Maximization Algorithm.
    Calls the ngmix fitter ngmix.EMRunner with an image of the Psf.
    Returns nGauss Gaussians stored as lsst.shapelet.ShapeletFunction components.
    """
    ConfigClass = SingleFrameEmPsfApproxConfig

    gaussian_pars_len = 6

    @classmethod
    def getExecutionOrder(cls):
        return cls.SHAPE_ORDER

    def __init__(self, config, name, schema, metadata):
        SingleFramePlugin.__init__(self, config, name, schema, metadata)
        flagDefs = FlagDefinitionList()
        flagDefs.addFailureFlag()
        self.rangeError = flagDefs.add("flag_rangeError", "Iteration error in Gaussian parameters.")
        self.maxIters = flagDefs.add("flag_maxIters", "Fitter exceeded maxIters setting without converging.")
        self.noPsf = flagDefs.add("flag_noPsf", "No PSF attached to the exposure.")
        self.flagHandler = FlagHandler.addFields(schema, name, flagDefs)

        self.iterKey = schema.addField(name + '_iterations', type='I', doc="number of iterations run")
        self.fdiffKey = schema.addField(name + '_fdiff', type='D', doc="fit difference")

        # Add ShapeletFunction keys for the number of Gaussians requested
        self.keys = []
        for i in range(config.nGauss):
            key = lsst.shapelet.ShapeletFunctionKey.addFields(schema,
                                                             "%s_%d"%(name, i),
                                                             "ngmix EM gaussian", "pixels", "",
                                                              0, lsst.shapelet.HERMITE)
            self.keys.append(key)
        self.msfKey = lsst.shapelet.MultiShapeletFunctionKey(self.keys)

    def measure(self, measRecord, exposure):
        if exposure.getPsf() is None:
            raise MeasurementError(self.noPsf.doc, self.noPsf.number)
        psfArray = exposure.getPsf().computeKernelImage().getArray()
        psfObs = ngmix.observation.Observation(psfArray,
            jacobian=ngmix.UnitJacobian(row=(psfArray.shape[0]-1)/2, col=(psfArray.shape[1]-1)/2))

        #  Need a guess at the sum of the diagonal moments
        nGauss = self.config.nGauss
        shape = exposure.getPsf().computeShape()
        Tguess = shape.getIxx() + shape.getIyy()
        emPars = {'maxiter': self.config.maxIters, 'tol': self.config.tolerance}

        runner = EMRunner(psfObs, Tguess, nGauss, emPars)
        runner.go(ntry=self.config.nTries)
        fitter = runner.get_fitter()
        res = fitter.get_result()

        #   Set the results, including the fit info returned by EMRunner
        measRecord.set(self.iterKey, res['numiter'])
        measRecord.set(self.fdiffKey, res['fdiff'])

        #   We only know about two EM errors.  Anything else is thrown as "unknown".
        if res['flags'] != 0:
            if res['flags'] & EM_RANGE_ERROR:
                raise MeasurementError(self.rangeError.doc, self.rangeError.number)
            if res['flags'] & EM_MAXITER:
                raise MeasurementError(self.maxIters.doc, self.maxIters.number)
            raise RuntimeError("Unknown EM fitter exception")

        #   Convert the nGauss Gaussians to ShapeletFunction's.  Zeroth order HERMITES are Gaussians.
        #   There are always 6 parameters for each Gaussian.
        psf_pars = fitter.get_gmix().get_full_pars()
        #  Gaussian pars are 6 numbers long.  Pick the results off one at a time
        for i in range(self.config.nGauss):
            flux, y, x, iyy, ixy, ixx = psf_pars[i*self.gaussian_pars_len: (i+1)*self.gaussian_pars_len]
            quad = lsst.afw.geom.ellipses.Quadrupole(ixx, iyy, ixy)
            ellipse = lsst.afw.geom.ellipses.Ellipse(quad, lsst.afw.geom.Point2D(x, y))
            # create a 0th order (gaussian) shapelet function.
            sf = lsst.shapelet.ShapeletFunction(0, lsst.shapelet.HERMITE, ellipse)
            sf.getCoefficients()[0] = flux/lsst.shapelet.ShapeletFunction.FLUX_FACTOR
            measRecord.set(self.keys[i], sf)

    #   This routine responds to the standard failure call in baseMeasurement
    def fail(self, measRecord, error=None):
        if error is None:
            self.flagHandler.handleFailure(measRecord)
        else:
            self.flagHandler.handleFailure(measRecord, error.cpp)

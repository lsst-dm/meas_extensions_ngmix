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

import numpy as np
import ngmix
from ngmix.bootstrap import EMRunner
from ngmix.em import EM_RANGE_ERROR, EM_MAXITER

import lsst.pex.exceptions
import lsst.afw.detection
import lsst.afw.geom
import lsst.shapelet

from lsst.meas.base.pluginRegistry import register
from lsst.meas.base import SingleFramePluginConfig, SingleFramePlugin, FatalAlgorithmError

from .converters import convertGMixToMultiShapelet


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
    maxIter = lsst.pex.config.Field(dtype=int, default=10000, optional=False,
                                    doc="maximum number of iterations")
    tolerance = lsst.pex.config.Field(dtype=float, default=1e-6, optional=False,
                                      doc="tolerance")


@register("ngmix_EMPsfApprox")
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

        self.iterKey = schema.addField(
            schema.join(name, "iterations"),
            type=np.int32,
            doc="number of iterations run"
        )
        self.fdiffKey = schema.addField(
            schema.join(name, "fdiff"),
            type=float,
            doc="fit difference"
        )
        self.flagKey = schema.addField(
            schema.join(name, "flag"),
            type="Flag",
            doc="General failure flag."
        )
        self.flagRangeErrorKey = schema.addField(
            schema.join(name, "flag_rangeError"),
            type="Flag",
            doc="Invalid ellipse parameters detected during iteration."
        )
        self.flagMaxIterKey = schema.addField(
            schema.join(name, "flag_maxIter"),
            type="Flag",
            doc="Maximum number of iterations (%d) exceeded." % self.config.maxIter
        )

        # Add MultiShapeletFunction key for the actual multi-Gaussian (a
        # multi-Gaussian is a MultiShapelet with all components zeroth-order).
        self.msfKey = lsst.shapelet.MultiShapeletFunctionKey.addFields(
            schema,
            name,
            "ngmix E-M multi-Gaussian PSF approximation",
            ellipseUnit="pixel",
            coeffUnit="",
            orders=[0]*config.nGauss,
        )

    def measure(self, measRecord, exposure):
        psf = exposure.getPsf()
        if psf is None:
            raise FatalAlgorithmError("EMPsfApprox requires a PSF.")
        psfArray = psf.computeKernelImage().getArray()
        psfObs = ngmix.observation.Observation(
            psfArray,
            jacobian=ngmix.UnitJacobian(row=(psfArray.shape[0]-1)/2, col=(psfArray.shape[1]-1)/2)
        )

        #  Need a guess at the sum of the diagonal moments
        nGauss = self.config.nGauss
        shape = psf.computeShape()
        Tguess = shape.getIxx() + shape.getIyy()
        emPars = {'maxiter': self.config.maxIter, 'tol': self.config.tolerance}

        runner = EMRunner(psfObs, Tguess, nGauss, emPars)
        runner.go(ntry=self.config.nTries)
        fitter = runner.get_fitter()
        res = fitter.get_result()

        #   Set the results, including the fit info returned by EMRunner
        measRecord.set(self.iterKey, res['numiter'])
        measRecord.set(self.fdiffKey, res['fdiff'])

        #   We only know about two EM errors.
        if res['flags'] != 0:
            if res['flags'] & EM_RANGE_ERROR:
                measRecord.set(self.flagRangeErrorKey, True)
            if res['flags'] & EM_MAXITER:
                measRecord.set(self.flagMaxIterKey, True)
            if not (res['flags'] & (EM_RANGE_ERROR | EM_MAXITER)):
                raise RuntimeError("Unexpected error flag %x in ngmix results." % res['flags'])
            measRecord.set(self.flagKey, True)
            return

        measRecord.set(self.msfKey, convertGMixToMultiShapelet(fitter.get_gmix()))

    def fail(self, measRecord, error=None):
        # Should only get here if we get an unexpected exception; in that case
        # we set the general flag and rely on the measurement framework to
        # also warn.
        measRecord.set(self.flagKey, True)

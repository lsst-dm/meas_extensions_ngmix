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
import ngmix
from ngmix.fitting import LMSimple, LM_SINGULAR_MATRIX, LM_NEG_COV_EIG, LM_NEG_COV_DIAG,\
    EIG_NOTFINITE, LM_FUNC_NOTFINITE, LM_DIV_ZERO
from ngmix.observation import Observation

import lsst.pex.exceptions
import lsst.afw.detection
import lsst.afw.geom
from lsst.meas.base.pluginRegistry import register
from lsst.meas.base.sfm import SingleFramePluginConfig, SingleFramePlugin
from lsst.meas.base.baseLib import MeasurementError
import lsst.meas.base.flagDecorator
from lsst.shapelet import ShapeletFunction, ShapeletFunctionKey, MultiShapeletFunctionKey, HERMITE

__all__ = ("SingleFrameNgmixConfig", "SingleFrameNgmixPlugin")


class SingleFrameNgmixConfig(SingleFramePluginConfig):

    psfName = lsst.pex.config.Field(dtype=str, default=None, optional=False,
                                    doc = "Field name prefix of the PSF approximation plugin to use")

    model = lsst.pex.config.Field(dtype=str, default="gauss", optional=False,
                                  doc="which LMSimple model to run")
    maxfev = lsst.pex.config.Field(dtype=int, default=4000, optional=False,
                                   doc="maximum number of function evaluations")
    ftol = lsst.pex.config.Field(dtype=float, default=1e-5, optional=False,
                                 doc="tolerance in sum of squares")
    xtol = lsst.pex.config.Field(dtype=float, default=1e-5, optional=False,
                                 doc="tolerance in solution")


@register("meas_extensions_ngmix_LMSimpleShape")
@lsst.meas.base.flagDecorator.addFlagHandler(("flag", "General Failure error"),
        ("flag_noPsf", "No PSF attached to the exposure."),
        ("flag_maxFev", "Maximum function evaluations exceeded"),
        ("flag_singularMatrix", "Singular Matrix"),
        ("flag_negativeCovEig", "Negative Covariance Eig"),
        ("flag_negativeCoveDiag", "Negative Covariance Diag"),
        ("flag_eigNotFinite", "Eig not finite"),
        ("flag_funcNotFinite", "Func not finite"),
        ("flag_divideByZero", "Divide by Zero"))
class SingleFrameNgmixPlugin(SingleFramePlugin):
    '''
    Plugin to do shape measurement using ngmix.LMSimple fitter
    May be run using gauss, dev, or exp model
    '''
    ConfigClass = SingleFrameNgmixConfig

    #   Length of a single gaussian representation in ngmix
    _gaussian_pars_len = 6
    _modelInfo = {"gauss": {"nPars": 6, "nGauss": 1},
                  "dev": {"nPars": 6, "nGauss": 6},
                  "exp": {"nPars": 6, "nGauss": 6}}

    @classmethod
    def getExecutionOrder(cls):
        return cls.SHAPE_ORDER+0.5

    #   This defines the legal LMSimple routines to call
    #   This metadata will be fetched from Erin when he adds an API for it
    def getModelInfo(self, model):
        if model not in self._modelInfo.keys():
            raise lsst.pex.exceptions.RuntimeError("Illegal model: %s for LMSimple"%model)
        return self._modelInfo[model]

    def __init__(self, config, name, schema, metadata):
        SingleFramePlugin.__init__(self, config, name, schema, metadata)

        # Add ShapeletFunction keys for the number of Gaussians required
        self.keys = []
        self.nGauss = self.getModelInfo(self.config.model)["nGauss"]
        #   Add a 0th order shapelet for every Gaussian to be returned
        for i in range(self.nGauss):
            key = ShapeletFunctionKey.addFields(schema, "%s_%d"%(name, i), "ngmix Shape gaussian",
                                                "pixels", "", 0, HERMITE)
            self.keys.append(key)
        #   If the psfName is specified, it means to we can get the psfApprox from it
        self.multiShapeletFunctionKey = MultiShapeletFunctionKey(self.keys)
        if self.config.psfName is not None:
            self.psfMsfKey = MultiShapeletFunctionKey(schema[self.config.psfName], HERMITE)
        else:
            self.psfMsfKey = None
        self.name = name

    def measure(self, measRecord, exposure):
        psf = exposure.getPsf()
        if psf is None:
            raise MeasurementError(self.flagHandler.getDefinition(
                                   SingleFrameNgmixPlugin.ErrEnum.flag_noPsf).doc,
                                   SingleFrameNgmixPlugin.ErrEnum.flag_noPsf)
        # make an observation for the psf image
        psfArray = psf.computeKernelImage().getArray()
        psfJacob = ngmix.UnitJacobian(row=(psfArray.shape[0] - 1.0)/2.0, col=(psfArray.shape[1] - 1.0)/2.0)
        psfObs = Observation(psfArray, jacobian=psfJacob)

        #   Fallback code if no psf algorithm is requested.  Then just do an LM single gaussian fit
        if self.config.psfName is None:
            pfitter = LMSimple(psfObs, 'gauss')
            #   gues parameters for a simple Gaussian
            psfPars = [0.0, 0.0, -0.03, 0.02, 8.0, 1.0]
            pfitter.go(psfPars)
            psfGMixFit = pfitter.get_gmix()
            psfObs.set_gmix(psfGMixFit)
        else:
            shape = psfArray.shape
            image = lsst.afw.image.ImageD(shape[1], shape[0])
            evaluate = measRecord.get(self.psfMsfKey).evaluate()
            evaluate.addToImage(image.getArray(), lsst.afw.geom.Point2I(-(shape[0] - 1)//2, -(shape[1] - 1)//2))
            psfObs = Observation(image.getArray(), jacobian=psfJacob)
            #   Now create a gmix directly from what the PsfApprox algorithm produced
            multiShapeletFunction = measRecord.get(self.psfMsfKey)
            shapeletFunctions = multiShapeletFunction.getComponents()
            psfPars = []
            #   add pars in the order flux, y, x, yy, xy, xx
            for sf in shapeletFunctions:
                psfPars.append(sf.getCoefficients()[0])
                psfPars.append(sf.getEllipse().getCenter().getY())
                psfPars.append(sf.getEllipse().getCenter().getX())
                psfPars.append(sf.getEllipse().getCore().getIyy())
                psfPars.append(sf.getEllipse().getCore().getIxy())
                psfPars.append(sf.getEllipse().getCore().getIxx())
            gmix = ngmix.gmix.GMix(len(shapeletFunctions), psfPars)
            psfObs.set_gmix(gmix)

        #   Now create an obs for the galaxy itself, including the weight plane
        galArray = exposure.getMaskedImage().getImage().getArray()
        gal_jacob = ngmix.UnitJacobian(row=(galArray.shape[1] - 1)/2.0, col=(galArray.shape[0] - 1)/2.0)
        variance = exposure.getMaskedImage().getVariance().getArray()
        gal_weight = 1/variance
        obs = Observation(galArray, weight=gal_weight, jacobian=gal_jacob, psf=psfObs)

        #   Run detection on the source image to get a guess of the object pars
        #   This will give us a rough guess about the shape of the object and its centroid
        threshold = lsst.afw.detection.Threshold(5.0, lsst.afw.detection.Threshold.STDEV)
        fpSet = lsst.afw.detection.FootprintSet(exposure.getMaskedImage().getImage(), threshold)
        if len(fpSet.getFootprints()) > 1:
            raise RuntimeError("Threshold value results in multiple Footprints for a single object")
        if len(fpSet.getFootprints()) == 0:
            raise RuntimeError("Threshold value results in zero Footprints for object")
        fp = fpSet.getFootprints()[0]
        x = fp.getCentroid().getX() - exposure.getXY0().getX()
        y = fp.getCentroid().getY() - exposure.getXY0().getY()
        xx = fp.getShape().getIxx()
        yy = fp.getShape().getIyy()
        xy = fp.getShape().getIxy()
        #    set the guess as [cen1, cen2, g1, g2, T, flux]
        guess = (x, y, (xx-yy)/(xx+yy), 2*xy/(xx+yy), xx+yy, galArray[x, y])

        #   Now run the shape algorithm, using the config parameters
        lmPars = {'maxfev': self.config.maxfev, 'ftol': self.config.ftol, 'xtol': self.config.xtol}
        fitter = LMSimple(obs, self.config.model, lm_pars=lmPars)
        fitter.go(guess=guess)
        res = fitter.get_result()

        #   Set the results, including the fit info returned by EMRunner
        #   We only know about two EM errors.  Anything else is thrown as "unknown".
        if res['flags'] != 0:
            if res['flags'] & LM_SINGULAR_MATRIX:
                raise MeasurementError(self.flagHandler.getDefinition(
                    SingleFrameNgmixPlugin.ErrEnum.flag_singularMatrix).doc,
                    SingleFrameNgmixPlugin.ErrEnum.flag_singularMatrix)
            if res['flags'] & LM_NEG_COV_EIG:
                raise MeasurementError(
                    self.flagHandler.getDefinition(SingleFrameNgmixPlugin.ErrEnum.flag_negativeCovEig).doc,
                    SingleFrameNgmixPlugin.ErrEnum.flag_negativeCovEig)
            if res['flags'] & LM_NEG_COV_DIAG:
                raise MeasurementError(
                    self.flagHandler.getDefinition(SingleFrameNgmixPlugin.ErrEnum.flag_negativeCovDiag).doc,
                    SingleFrameNgmixPlugin.ErrEnum.flag_negativeCovDiag)
            if res['flags'] & EIG_NOTFINITE:
                raise MeasurementError(
                    self.flagHandler.getDefinition(SingleFrameNgmixPlugin.ErrEnum.flag_eigNotFinite).doc,
                    SingleFrameNgmixPlugin.ErrEnum.flag_eigNotFinite)
            if res['flags'] & LM_FUNC_NOTFINITE:
                raise MeasurementError(
                    self.flagHandler.getDefinition(SingleFrameNgmixPlugin.ErrEnum.flag_funcNotFinite).doc,
                    SingleFrameNgmixPlugin.ErrEnum.flag_funcNotFinite)
            if res['flags'] & LM_DIV_ZERO:
                raise MeasurementError(
                    self.flagHandler.getDefinition(SingleFrameNgmixPlugin.ErrEnum.flag_divZero).doc,
                    SingleFrameNgmixPlugin.ErrEnum.flag_divZero)
            if res['nfev'] >= self.config.maxfev:
                raise MeasurementError(
                    self.flagHandler.getDefinition(SingleFrameNgmixPlugin.ErrEnum.flag_maxFev).doc,
                    SingleFrameNgmixPlugin.ErrEnum.flag_maxFev)
            #   Unknown error, but there should be an errmsg set by ngmix
            raise RuntimeError(res['errmsg'])

        #   Convert the nGauss Gaussians to ShapeletFunction's. Zeroth order HERMITES are Gaussians.
        #   There are always 6 parameters for each Gaussian.
        galPars = fitter.get_gmix().get_full_pars()
        for i in range(self.nGauss):
            flux, y, x, iyy, ixy, ixx = galPars[i*self._gaussian_pars_len: (i+1)*self._gaussian_pars_len]
            quad = lsst.afw.geom.ellipses.Quadrupole(ixx, iyy, ixy)
            ellipse = lsst.afw.geom.ellipses.Ellipse(quad, lsst.afw.geom.Point2D(x, y))
            # create a 0th order (gaussian) shapelet function.
            sf = ShapeletFunction(0, HERMITE, ellipse)
            sf.getCoefficients()[0] = flux/ShapeletFunction.FLUX_FACTOR
            measRecord.set(self.keys[i], sf)

    #   This routine responds to the standard failure call in baseMeasurement
    def fail(self, measRecord, error=None):
        if error is None:
            self.flagHandler.handleFailure(measRecord)
        else:
            self.flagHandler.handleFailure(measRecord, error.cpp)

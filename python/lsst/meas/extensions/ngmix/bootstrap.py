"""
bootstrap the ngmix measurements

There are bootstrappers in the ngmix code base but they need to be refactored.
Also we don't plan to support multi-epoch fitting so these can be simplified
"""

import numpy as np
import ngmix
from ngmix.gexceptions import GMixRangeError, BootPSFFailure, BootGalFailure
from . import procflags

class BootstrapperBase(object):
    """
    bootstrap a fit on the object.
    """
    def __init__(self, obs):
        self.mbobs = ngmix.observation.get_mb_obs(obs)

    def go(self):
        """
        run the fitting
        """
        raise NotImplementedError("implement in a child class")

class MaxBootstrapper(BootstrapperBase):
    """
    bootstrap maximum likelihood fits
    """
    def __init__(self, obs, psfconf, objconf, prior, rng):
        super(MaxBootstrapper,self).__init__(obs)
        self.psfconf=psfconf
        self.objconf=objconf
        self.prior=prior
        self.rng=rng

        self._set_default_result()

    def _set_default_result(self):
        self.result = {
            'psf':{
                'flags':procflags.NO_ATTEMPT,
                'byband':[],
            },
            'psf_flux':{
                'flags':procflags.NO_ATTEMPT,
            },
            'obj': {
                'flags':procflags.NO_ATTEMPT,
            }
        }

    def fit_psfs(self):
        """
        Fit the psfs and set the gmix attribute

        side effects
        ------------
        The result dict is modified to set the fit data and set flags.  A gmix
        object set for the psf observations if the fits succeed
        """
        pres=self.result['psf']
        pres_byband=pres['byband']

        pres['flags'] = 0

        pconf = self.psfconf
        sigma_guess = pconf['fwhm_guess']/2.35
        Tguess = 2*sigma_guess**2

        for band,obslist in enumerate(self.mbobs):
            assert len(obslist)==1,"multi-epoch fitting is not supported"
            obs=obslist[0]

            psf_obs = obs.psf
            runner = self._get_psf_runner(psf_obs, Tguess)
            runner.go(ntry=pconf['max_pars']['ntry'])
            fitter=runner.get_fitter()
            tres=fitter.get_result()

            pres['flags'] |= tres['flags']


            if tres['flags'] == 0:
                gmix = fitter.get_gmix()
                psf_obs.gmix = gmix
                g1,g2,T=gmix.get_g1g2T()
                tres['g1'] = g1
                tres['g2'] = g2
                tres['T'] = T

            pres_byband.append(tres)

    def _get_psf_runner(self, obs, Tguess):
        """
        get a runner to be used for fitting the psfs
        """
        pconf=self.psfconf
        model=pconf['model']
        if 'coellip' in model:
            ngauss=ngmix.bootstrap.get_coellip_ngauss(model)
            runner=ngmix.bootstrap.PSFRunnerCoellip(
                obs,
                Tguess,
                ngauss,
                pconf['max_pars']['lm_pars'],
                rng=self.rng,
            )
        elif 'em' in model:
            raise NotImplementedError("implement EM psf fitting")
        else:
            runner=ngmix.bootstrap.PSFRunner(
                obs,
                model,
                Tguess,
                pconf['max_pars']['lm_pars'],
                rng=self.rng,
            )
        return runner

    def _get_runner(self):
        """
        get a runner to be used for fitting the object
        """
        pass



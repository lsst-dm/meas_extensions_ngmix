"""
bootstrap the ngmix measurements

There are bootstrappers in the ngmix code base but they need to be refactored.
Also we don't plan to support multi-epoch fitting so these can be simplified
"""

import lsst.log
import numpy as np
import ngmix
import galsim
from . import procflags

from copy import deepcopy

logger = lsst.log.Log.getLogger("meas.extensions.ngmix.bootstrap")

DEFAULT_RESULT = {
    # overall processing flags
    'flags': procflags.NO_ATTEMPT,

    # psf fitting related information
    'psf': {
        'flags': procflags.NO_ATTEMPT,
        'byband': [],
    },

    # psf flux fitting related information
    'psf_flux': {
        'flags': procflags.NO_ATTEMPT,
        'byband': [],
    },

    # object fitting related information
    'obj': {
        'flags': procflags.NO_ATTEMPT,
    }
}

DEFAULT_MCAL_RESULT = {
    'mcal_flags': procflags.NO_ATTEMPT,
    'noshear': {},
    '1p': {},
    '1m': {},
    '2p': {},
    '2m': {},
}

for type in DEFAULT_MCAL_RESULT:
    if type not in ['mcal_flags']:
        DEFAULT_MCAL_RESULT[type] = deepcopy(DEFAULT_RESULT)


def get_default_result():
    """
    get the default result dict for fitting
    """
    return deepcopy(DEFAULT_RESULT)


def get_default_mcal_result():
    """
    get the default result dict for metacal
    """
    return deepcopy(DEFAULT_MCAL_RESULT)


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

    def __init__(self, obs, config, prior, rng):
        super(MaxBootstrapper, self).__init__(obs)
        self.config = config
        self.prior = prior
        self.rng = rng

        self._set_default_result()

    @property
    def result(self):
        """
        get a reference to the result dictionary
        """
        return self._result

    def fit_psfs(self):
        """
        Fit the psfs and set the gmix attribute

        side effects
        ------------
        The result dict is modified to set the fit data and set flags.  A gmix
        object set for the psf observations if the fits succeed
        """
        res = self.result

        # we have now begun processing, so set to zero
        res['flags'] = 0

        pres = res['psf']
        pres_byband = pres['byband']

        pres['flags'] = 0

        pconf = self.config['psf']
        sigma_guess = pconf['fwhm_guess']/2.35
        Tguess = 2*sigma_guess**2

        for band, obslist in enumerate(self.mbobs):
            assert len(obslist) == 1, "multi-epoch fitting is not supported"
            obs = obslist[0]

            psf_obs = obs.psf
            runner = self._get_psf_runner(psf_obs, Tguess)
            runner.go(ntry=pconf['max_pars']['ntry'])
            fitter = runner.fitter
            tres = fitter.get_result()

            pres['flags'] |= tres['flags']

            if tres['flags'] == 0:
                gmix = fitter.get_gmix()
                psf_obs.gmix = gmix
                g1, g2, T = gmix.get_g1g2T()
                tres['g1'] = g1
                tres['g2'] = g2
                tres['T'] = T
            else:
                filt = self.cdict['filters'][band]
                logger.debug('psf fit failed for filter %s' % filt)

            pres_byband.append(tres)

        if pres['flags'] == 0:
            self._set_mean_psf_stats(pres)
        else:
            res['flags'] |= procflags.PSF_FIT_FAILURE

    def fit_psf_fluxes(self, normalize_psf=True):
        """
        use psf as a template, measure flux (linear)

        make normalize_psf a configurable

        side effects
        ------------
        The result dict is modified to set the fit data and set flags.
        """

        mbobs = self.mbobs

        res = self.result
        pfres = res['psf_flux']
        pfres['flags'] = 0

        res_byband = pfres['byband']

        for band, obs_list in enumerate(mbobs):

            if not obs_list[0].has_psf_gmix():
                filt = self.cdict['filters'][band]
                tres = {'flags': procflags.NO_ATTEMPT}
                logger.debug('not fitting psf flux in '
                             'filter %s due to missing PSF fit' % filt)
            else:
                fitter = ngmix.fitting.TemplateFluxFitter(
                    obs_list,
                    do_psf=True,
                    normalize_psf=normalize_psf,
                )
                fitter.go()

                tres = fitter.get_result()

            if tres['flags'] != 0:
                logger.debug('psf flux fit failed: %s' % tres['flags'])
                pfres['flags'] |= procflags.PSF_FLUX_FIT_FAILURE
                res['flags'] |= procflags.PSF_FLUX_FIT_FAILURE

            res_byband.append(tres)

    def fit_model(self):
        """
        fit the model to the object
        """

        res = self.result

        # we need psfs to be fit and psf fluxes to be fit
        pfres = res['psf_flux']
        if pfres['flags'] != 0:
            raise RuntimeError(
                'psf flux flags non zero: %d.  '
                'cannot fit object without psf fluxes' % pfres['flags']
            )

        self.result['obj']['flags'] = 0

        runner = self._get_runner()
        runner.go(ntry=self.config['obj']['max_pars']['ntry'])

        tres = runner.fitter.get_result()

        res['obj'].update(tres)
        if res['obj']['flags'] != 0:
            res['flags'] |= procflags.OBJ_FIT_FAILURE

    def _set_model_stats(self, tres):
        """
        set parameters from the modeling
        """

    def _set_mean_psf_stats(self, pres):

        byband = pres['byband']

        g1 = np.array([t['g1'] for t in byband])
        g2 = np.array([t['g2'] for t in byband])
        T = np.array([t['T'] for t in byband])

        pres['g1_mean'] = g1.mean()
        pres['g2_mean'] = g2.mean()
        pres['T_mean'] = T.mean()

    def _get_psf_runner(self, obs, Tguess):
        """
        get a runner to be used for fitting the psfs
        """
        pconf = self.config['psf']
        model = pconf['model']
        if 'coellip' in model:
            ngauss = ngmix.bootstrap.get_coellip_ngauss(model)
            runner = ngmix.bootstrap.PSFRunnerCoellip(
                obs,
                Tguess,
                ngauss,
                pconf['max_pars']['lm_pars'],
                rng=self.rng,
            )
        elif 'em' in model:
            raise NotImplementedError("implement EM psf fitting")
        else:
            runner = ngmix.bootstrap.PSFRunner(
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

        objconf = self.config['obj']

        max_pars = {}
        max_pars.update(objconf['max_pars'])
        max_pars['method'] = 'lm'

        guesser = self._get_guesser()

        if objconf['model'] in ['bd', 'bdf']:
            runner = ngmix.bootstrap.BDFRunner(
                self.mbobs,
                max_pars,
                guesser,
                prior=self.prior,
            )
        else:
            runner = ngmix.bootstrap.MaxRunner(
                self.mbobs,
                objconf['model'],
                max_pars,
                guesser,
                prior=self.prior,
            )

        return runner

    def _get_guesser(self):
        """
        get a guesser for the model parameters
        """
        pfres = self.result['psf_flux']
        if pfres['flags'] != 0:
            raise RuntimeError(
                'psf flux flags non zero: %d.  '
                'cannot fit object without psf fluxes' % pfres['flags']
            )

        Tguess = self.result['psf']['T_mean']
        flux_guesses = [t['flux'] for t in pfres['byband']]

        if self.config['obj']['model'] in ['bd', 'bdf']:
            guesser = ngmix.guessers.BDFGuesser(
                Tguess,
                flux_guesses,
                self.prior,
            )
        else:
            guesser = ngmix.guessers.TFluxAndPriorGuesser(
                Tguess,
                flux_guesses,
                self.prior,
            )

        return guesser

    def _set_default_result(self):
        self._result = get_default_result()


class MetacalMaxBootstrapper(object):
    """
    do metacal with fits using maximum likelihood
    """

    def __init__(self,
                 mbobs,
                 config,
                 prior,
                 rng):

        self.mbobs = mbobs
        self.config = config
        self.prior = prior
        self.rng = rng

        self._set_default_result()

    @property
    def result(self):
        """
        get a reference to the result dictionary
        """
        return self._result

    def go(self):
        """
        do all the processing necessary for metacal
        """

        res = self.result
        res['mcal_flags'] = 0

        config = self.config

        if config['metacal'].get('symmetrize_psf', False):
            self._do_psf_fits_for_symmetrize(self.mbobs)
            if res['mcal_flags'] != 0:
                return

        try:
            mdict = ngmix.metacal.get_all_metacal(
                self.mbobs,
                **config['metacal'],
            )
        except galsim.GalSimValueError:
            res['mcal_flags'] = procflags.IMAGE_FLAGS
            return

        for type, tmbobs in mdict.items():
            self._do_one_metacal(tmbobs, type)

    def _do_one_metacal(self, mbobs, type):
        """
        run fits on metacaled images

        the result will be set and possibly flags
        """
        res = self.result

        boot = self._get_one_bootstrapper(mbobs)
        boot.fit_psfs()
        boot.fit_psf_fluxes()

        if boot.result['psf']['flags'] != 0:
            logger.debug('skipping model fit due psf fit failure')
            res['mcal_flags'] |= procflags.METACAL_PSF_FAILURE
        elif boot.result['psf_flux']['flags'] != 0:
            logger.debug('skipping model fit due psf flux fit failure')
            res['mcal_flags'] |= procflags.METACAL_PSF_FLUX_FAILURE
        else:
            boot.fit_model()
            if boot.result['obj']['flags'] != 0:
                res['mcal_flags'] |= procflags.METACAL_OBJ_FAILURE

        res[type] = boot.result

    def _do_psf_fits_for_symmetrize(self, mbobs):
        """
        run psf fits *before* the metacal fits
        so we can do symmetrization
        """

        boot = self._get_one_bootstrapper(mbobs)
        boot.fit_psfs()

        if boot.result['psf']['flags'] != 0:
            logger.debug('cannot do symmetrize psf due to psf fitting failures')
            # TODO need finer grained flag
            raise RuntimeError("Code path not supported")
            res = dict()
            res['mcal_flags'] = procflags.METACAL_PSF_FAILURE
            return res

    def _get_one_bootstrapper(self, mbobs):
        """
        get a bootstrapper for the input observations
        """
        return MaxBootstrapper(
            mbobs,
            self.config,
            self.prior,
            self.rng,
        )

    def _set_default_result(self):
        self._result = get_default_mcal_result()

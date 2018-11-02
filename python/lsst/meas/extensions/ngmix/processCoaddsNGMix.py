"""
Some TODO items (there are many more below in the code)

    - get a real estimate of the background noise. I am faking this
      by taking the median of the weight map, which includes the
      object poisson noise.  metacal is not tested with object poisson
      noise included

    - deal properly with the mask plane.  I've got something working
      but need someone to look it over.

    - normalize psf for flux fitting?

    - save all parameters for the PSF.  Because the number of parameters can
    vary a lot, this would require either very special code or saving an array
    (preferred for ease of coding)

    - metacal
        - make sure we are doing the right thing with the weight map.
        - This means checking to see that the underlying metacal
        code is working appropriately.
        - also need to record somewhere an amount of masking that
        is in place so we can cut on that.

    - add Tasks for multi-object fitting (MOF), which will require
      making stamps for multiple objects and fitting simultaneously

    - move configuration stuff to a separate module

    - move ngmix Observation extractor stuff to a separate module

    - get docs for the classes/structures that are input

    - get proper exceptions to throw rather than asserts, if that
    is preferred


"""
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

import time
import numpy as np
from lsst.geom import Extent2D
from lsst.afw.table import SourceCatalog, SchemaMapper
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
from lsst.pex.config import Field, ListField, ConfigField, Config, ChoiceField
from lsst.pipe.base import Struct
import lsst.log


from .processCoaddsTogether import ProcessCoaddsTogetherConfig, ProcessCoaddsTogetherTask
from . import util
from .util import Namer
from . import bootstrap
from . import priors

import ngmix

import pprint

#__all__ = ("ProcessCoaddsNGMixConfig", "ProcessCoaddsNGMixTask")



class MetacalConfig(Config):
    """
    configuration of metacalibration

    we can add more options later
    """
    types = ListField(
        dtype=str,
        default=['noshear','1p','1m','2p','2m'],
        optional=True,
        doc='types of images to create',
    )
    psf = Field(
        dtype=str,
        default='fitgauss',
        optional=True,
        doc=('Use round Gaussian for the PSF, based on a '
             'fit to the PSF image'),
    )

class StampsConfig(Config):
    """
    configuration for the postage stamps
    """
    min_stamp_size = Field(
        dtype=int,
        default=32,
        doc='min allowed stamp size',
    )
    max_stamp_size = Field(
        dtype=int,
        default=256,
        doc='min allowed stamp size',
    )
    sigma_factor = Field(
        dtype=float,
        default=5.0,
        doc='make stamp radius this many sigma',
    )

    bits_to_ignore_for_weight = ListField(
        dtype=str,
        default=[],
        doc='bits to ignore when calculating the background noise',
    )


    bits_to_null = ListField(
        dtype=str,
        default=[],
        doc='bits to null in the weight map',
    )




class LeastsqConfig(Config):
    """
    configuration for the likelihood fitting using scipy.leastsq
    """
    maxfev = Field(
        dtype=int,
        doc='max allowed number of function evaluations in scipy.leastsq',
    )
    xtol = Field(
        dtype=float,
        doc='xtol paramter for scipy.leastsq',
    )
    ftol = Field(
        dtype=float,
        doc='ftol paramter for scipy.leastsq',
    )

class MaxConfig(Config):
    ntry = Field(
        dtype=int,
        doc='number of times to attempt the fit with different guesses',
    )
    lm_pars = ConfigField(
        dtype=LeastsqConfig,
        doc="parameters for scipy.leastsq",
    )

class CenPriorConfig(Config):
    """
    configuration of the prior for the center position
    """
    type = ChoiceField(
        dtype=str,
        allowed={
            "gauss2d":"2d gaussian",
        },
        doc="type of prior for center",
    )
    pars = ListField(dtype=float, doc="parameters for the center prior")

class GPriorConfig(Config):
    """
    configuration of the prior for the ellipticity g
    """
    type = ChoiceField(
        dtype=str,
        allowed={
            "ba":"See Bernstein & Armstrong",
        },
        doc="type of prior for ellipticity g",
    )
    pars = ListField(dtype=float, doc="parameters for the ellipticity prior")

class TPriorConfig(Config):
    """
    configuration of the prior for the square size T
    """
    type = ChoiceField(
        dtype=str,
        allowed={
            "two-sided-erf":"two-sided error function, smoother than a flat prior",
        },
        doc="type of prior for the square size T",
    )
    pars = ListField(dtype=float, doc="parameters for the T prior")

class FluxPriorConfig(Config):
    """
    configuration of the prior for the flux
    """
    type = ChoiceField(
        dtype=str,
        allowed={
            "two-sided-erf":"two-sided error function, smoother than a flat prior",
        },
        doc="type of prior for the flux; gets repeated for multiple bands",
    )
    pars = ListField(dtype=float, doc="parameters for the flux prior")

class FracdevPriorConfig(Config):
    """
    configuration of the prior for fracdev, the fraction of the
    light in the bulge
    """
    type = ChoiceField(
        dtype=str,
        allowed={
            "gauss":"gaussian prior on fracdev",
        },
        doc="type of prior for fracdev",
    )
    pars = ListField(
        dtype=float,
        optional=True,
        doc="parameters for the fracdev prior",
    )

class ObjectPriorsConfig(Config):
    """
    Configuration of priors for the bulge+disk model
    """
    cen = ConfigField(dtype=CenPriorConfig, doc="prior on center")
    g = ConfigField(dtype=GPriorConfig, doc="prior on g")
    T = ConfigField(dtype=TPriorConfig, doc="prior on square size T")
    flux = ConfigField(dtype=FluxPriorConfig, doc="prior on flux")

    # this is optional, only used by the bulge+disk fitter
    fracdev = ConfigField(
        dtype=FracdevPriorConfig,
        default=None,
        #optional=True,
        doc="prior on fracdev",
    )

class MaxFitConfigBase(Config):
    """
    base config for max likelihood fitting
    """
    max_pars = ConfigField(
        dtype=MaxConfig,
        doc="parameters for maximum likelihood fitting with scipy.leastsq",
    )

class PSFMaxFitConfig(MaxFitConfigBase):
    """
    PSF fitting configuration using maximum likelihood

    inherits max_pars
    """
    model = ChoiceField(
        dtype=str,
        allowed={
            "gauss":"gaussian model",
            "coellip2":"coelliptical 2 gauss model",
            "coellip3":"coelliptical 3 gauss model",
        },
        doc="The model to fit with ngmix",
    )
    fwhm_guess = Field(
        dtype=float,
        doc='rough guess for PSF FWHM',
    )


class ObjectMaxFitConfig(MaxFitConfigBase):
    """
    object fitting configuration

   inherits max_pars
    """
    model = ChoiceField(
        dtype=str,
        allowed={
            "gauss":"gaussian model",
            "exp":"exponential model",
            "dev":"dev model",
            # bd and bdf are the same
            "bd":"bulge+disk model with fixed size ratio",
            "bdf":"bulge+disk model with fixed size ratio",
        },
        doc="The model to fit with ngmix",
    )

    priors = ConfigField(
        dtype=ObjectPriorsConfig,
        doc="priors for a maximum likelihood model fit",
    )


class BasicProcessConfig(ProcessCoaddsTogetherConfig):
    """
    basic config loads filters and misc stuff
    """
    filters = ListField(dtype=str, default=[], doc="List of expected bandpass filters.")

    stamps = ConfigField(dtype=StampsConfig, doc="configuration for postage stamps")

    start_index = Field(
        dtype=int,
        default=0,
        optional=True,
        doc='optional starting index for the processing',
    )
    num_to_process = Field(
        dtype=int,
        default=None,
        optional=True,
        doc='optional number to process',
    )

    make_plots = Field(
        dtype=bool,
        default=False,
        optional=True,
        doc='write some image plots',
    )
    plot_prefix= Field(
        dtype=str,
        default=None,
        optional=True,
        doc='prefix to add to plot names',
    )



class ProcessCoaddsNGMixMaxConfig(BasicProcessConfig):
    """
    fit the object and PSF using maximum likelihood
    """
    psf = ConfigField(dtype=PSFMaxFitConfig, doc='psf fitting config')
    obj = ConfigField(dtype=ObjectMaxFitConfig,doc="object fitting config")

    def setDefaults(self):
        """
        prefix for the output file
        """
        self.output.name = "deepCoadd_ngmix"

class ProcessDeblendedCoaddsNGMixMaxConfig(ProcessCoaddsNGMixMaxConfig):
    """
    fit the object and PSF using maximum likelihood
    """

    def setDefaults(self):
        """
        prefix for the output file
        """
        self.output.name = "deepCoadd_ngmix_deblended"


class ProcessCoaddsMetacalMaxConfig(BasicProcessConfig):
    """
    perform metacal using maximum likelihood
    """
    psf = ConfigField(dtype=PSFMaxFitConfig, doc='psf fitting config')
    obj = ConfigField(dtype=ObjectMaxFitConfig,doc='object fitting config')
    metacal = ConfigField(dtype=MetacalConfig,doc='metacal config')

    def setDefaults(self):
        """
        prefix for the output file
        """
        self.output.name = "deepCoadd_mcalmax"

class ProcessDeblendedCoaddsMetacalMaxConfig(ProcessCoaddsMetacalMaxConfig):
    """
    perform metacal using maximum likelihood on deblended coadds
    """

    def setDefaults(self):
        """
        prefix for the output file
        """
        self.output.name = "deepCoadd_mcalmax_deblended"



class ProcessCoaddsNGMixBaseTask(ProcessCoaddsTogetherTask):
    """
    Base class for ngmix tasks
    """
    _DefaultName = "processCoaddsNGMixBase"
    ConfigClass = BasicProcessConfig

    @property
    def cdict(self):
        """
        get a dict version of the configuration
        """
        if not hasattr(self,'_cdict'):
            self._cdict=self.config.toDict()
        return self._cdict

    def run(self, images, ref, replacers, imageId):
        """Process coadds from all bands for a single patch.

        This method should not add or modify self.

        So far all children are u sing this exact code so leaving
        it here for now. If we specialize a lot, might make a 
        processor its own object

        Parameters
        ----------
        images : `dict` of `lsst.afw.image.ExposureF`
            Coadd images and associated metadata, keyed by filter name.
        ref : `lsst.afw.table.SourceCatalog`
            A catalog with one record for each object, containing "best"
            measurements across all bands.
        replacers : `dict` of `lsst.meas.base.NoiseReplacer`, optional
            A dictionary of `~lsst.meas.base.NoiseReplacer` objects that can
            be used to insert and remove deblended pixels for each object.
            When not `None`, all detected pixels in ``images`` will have
            *already* been replaced with noise, and this *must* be used
            to restore objects one at a time.
        imageId : `int`
            Unique ID for this unit of data.  Should be used (possibly
            indirectly) to seed random numbers.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            Struct with (at least) an `output` attribute that is a catalog
            to be written as ``self.config.output``.
        """

        tm0 = time.time()
        nproc = 0

        self.set_rng(imageId)

        config=self.cdict
        self.log.info(pprint.pformat(config))

        extractor = self._get_extractor(images)

        # Make an empty catalog
        output = SourceCatalog(self.schema)

        # Add mostly-empty rows to it, copying IDs from the ref catalog.
        output.extend(ref, mapper=self.mapper)

        index_range=self.get_index_range(output)

        for n, (outRecord, refRecord) in enumerate(zip(output, ref)):
            if n < index_range[0] or n > index_range[1]:
                continue

            self.log.info('index: %06d/%06d' % (n,index_range[1]))
            nproc += 1

            outRecord.setFootprint(None)  # copied from ref; don't need to write these again

            # Insert the deblended pixels for just this object into all images.
            if replacers is not None:
                for r in replacers.values():
                    r.insertSource(refRecord.getId())

            mbobs = extractor.get_mbobs(refRecord)

            res = self._process_observations(ref['id'][n], mbobs)
            self._copy_result(mbobs, res, outRecord)

            # Remove the deblended pixels for this object so we can process the next one.
            if replacers is not None:
                for r in replacers.values():
                    r.removeSource(refRecord.getId())

        # Restore all original pixels in the images.
        if replacers is not None:
            for r in replacers.values():
                r.end()

        tm = time.time()-tm0
        self.log.info('time: %g min' % (tm/60.0))
        self.log.info('time per: %g sec' % (tm/nproc))

        return Struct(output=output)

    def get_index_range(self, cat):
        """
        Get the range of indices to process.  If these were not set in the
        configuration, [0,huge_number] is returned
        """
        if not hasattr(self,'_index_range'):

            ntot=len(cat)
            start=self.cdict['start_index']
            num=self.cdict['num_to_process']
            if start is None:
                start=0
            else:
                if start < 0 or start > ntot-1:
                    raise ValueError(
                        'requested start index %d out '
                        'of bounds [%d,%d]' % (start,ntot-1)
                    )

            if num is None:
                num=ntot-start
            if num < 1:
                raise ValueError(
                    'requested number to process %d '
                    'less than 1' % num
                )


            self._index_range=[start,start+num-1]

        return self._index_range

    def _process_observations(self, id, mbobs):
        """
        process the input observations

        Parameters
        ----------
        id: int
            ID of this observation
        mbobs: ngmix.MultiBandObsList
            ngmix multi-band observation, or maybe list of them if
            deblending.

        Returns
        -------
        results : `dict`
            Dictionary of outputs, with keys matching the fields added in
            `defineSchema()`.
        """
        raise NotImplementedError('implement in child class')

    def _get_bootstrapper(self, mbobs):
        """
        get a bootstrapper to automate the processing
        """
        raise NotImplementedError('implement in child class')

    def _copy_result(self, mbobs, res, output):
        """
        copy the result dict to the output record
        """
        raise NotImplementedError('implement in child class')

    def _get_extractor(self, images):
        """
        load the appropriate observation extractor
        """
        return MBObsExtractor(self.cdict, images)

    @property
    def rng(self):
        """
        get a ref to the random number generator
        """
        return self._rng

    def set_rng(self, seed):
        """
        set a random number generator based on the input seed

        parameters
        ----------
        seed: int
            The seed for the random number generator
        """
        self._rng = np.random.RandomState(seed)

    def _make_plots(self, id, mbobs):
        filters=self.cdict['filters']

        imlist=[o[0].image for o in mbobs]
        titles=[f for f in filters]

        imlist += [o[0].weight for o in mbobs]
        titles += [f+' wt' for f in filters]

        imlist += [o[0].bmask for o in mbobs]
        titles += [f+' bmask' for f in filters]

        plt=make_image_plots(
            id,
            imlist,
            ncols=len(filters),
            titles=titles,
            show=False,
        )

        fname='images-%d.png' % id
        if self.cdict['plot_prefix'] is not None:
            fname = self.cdict['plot_prefix'] + fname

        self.log.info('saving figure: %s' % fname)
        plt.savefig(fname)

class ProcessCoaddsNGMixMaxTask(ProcessCoaddsNGMixBaseTask):
    """
    Class for maximum likelihood fitting with ngmix
    """
    _DefaultName = "processCoaddsNGMixMax"
    ConfigClass = ProcessCoaddsNGMixMaxConfig

    def defineSchema(self, refSchema):
        """Return the Schema for the output catalog.

        This may add or modify self.

        Parameters
        ----------
        refSchema : `lsst.afw.table.Schema`
            Schema of the input reference catalogs.

        Returns
        -------
        outputSchema : `lsst.afw.table.Schema`
            Schema of the output catalog.  Will be added as ``self.schema``
            by calling code.
        """
        self.mapper = SchemaMapper(refSchema)
        self.mapper.addMinimalSchema(SourceCatalog.Table.makeMinimalSchema(), True)
        schema = self.mapper.getOutputSchema()

        config=self.cdict

        model=config['obj']['model']
        n=self.get_namer()
        pn=self.get_psf_namer()
        mn=self.get_model_namer()

        # generic ngmix fields
        mtypes=[
            (n('flags'),'overall flags for the processing',np.int32,''),
            (n('stamp_size'),'size of postage stamp',np.int32,''),
        ]

        # psf fitting related fields
        mtypes += [
            (pn('flags'),'overall flags for the PSF processing',np.int32,''),

            # mean over filters
            (pn('g2_mean'),'mean over filters of component 2 of the PSF ellipticity',np.float64,''),
            (pn('g1_mean'),'mean over filters of component 2 of the PSF ellipticity',np.float64,''),
            (pn('T_mean'),'mean over filters <x^2> + <y^2> for the gaussian mixture',np.float64,'arcsec^2'),
        ]

        # PSF measurements by filter
        for filt in config['filters']:
            pfn=self.get_psf_namer(filt=filt)
            mtypes += [
                (pfn('flags'), 'overall flags for PSF processing in %s filter' % filt, np.int32, ''),
                (pfn('row'),'offset from canonical row position',np.float64,'arcsec'),
                (pfn('col'),'offset from canonical col position',np.float64,'arcsec'),
                (pfn('g1'), 'component 1 of the PSF ellipticity in %s filter' % filt, np.float64, ''),
                (pfn('g2'), 'component 2 of the PSF ellipticity in %s filter' % filt, np.float64, ''),
                (pfn('T'), '<x^2> + <y^2> for the PSF in %s filter' % filt, np.float64, 'arcsec^2'),
            ]

        # PSF flux measurements, on the object, by filter
        for filt in config['filters']:
            pfn=self.get_psf_flux_namer(filt)
            mtypes += [
                (pfn('flux_flags'),'flags for PSF template flux fitting in the %s filter' % filt,np.float64,''),
                (pfn('flux'),'PSF template flux in the %s filter' % filt,np.float64,''),
                (pfn('flux_err'),'error on PSF template flux in the %s filter' % filt,np.float64,''),
            ]


        # object fitting related fields
        mtypes += [
            (mn('flags'),'flags for model fit',np.int32,''),
            (mn('nfev'),'number of function evaluations during fit',np.int32,''),
            (mn('chi2per'),'chi^2 per degree of freedom',np.float64,''),
            (mn('dof'),'number of degrees of freedom',np.int32,''),

            (mn('s2n'),'S/N for the fit',np.float64,''),

            (mn('row'),'offset from canonical row position',np.float64,'arcsec'),
            (mn('row_err'),'error on offset from canonical row position',np.float64,'arcsec'),
            (mn('col'),'offset from canonical col position',np.float64,'arcsec'),
            (mn('col_err'),'error on offset from canonical col position',np.float64,'arcsec'),
            (mn('g1'),'component 1 of the ellipticity',np.float64,''),
            (mn('g1_err'),'error on component 1 of the ellipticity',np.float64,''),
            (mn('g2'),'component 2 of the ellipticity',np.float64,''),
            (mn('g2_err'),'error on component 2 of the ellipticity',np.float64,''),
            (mn('T'),'<x^2> + <y^2> for the gaussian mixture',np.float64,'arcsec^2'),
            (mn('T_err'),'error on <x^2> + <y^2> for the gaussian mixture',np.float64,'arcsec^2'),
        ]
        if model in ['bd','bdf']:
            mtypes += [
                (mn('fracdev'),'fraction of light in the bulge',np.float64,''),
                (mn('fracdev_err'),'error on fraction of light in the bulge',np.float64,''),
            ]

        for filt in config['filters']:
            mfn=self.get_model_flux_namer(filt)
            mtypes += [
                (mfn('flux'),'flux in the %s filter' % filt,np.float64,''),
                (mfn('flux_err'),'error on flux in the %s filter' % filt,np.float64,''),
            ]

        for name,doc,dtype,units in mtypes:
            schema.addField(
                name,
                type=dtype,
                doc=doc,
                units=units,
            )

        return schema

    def _process_observations(self, id, mbobs):
        """
        process the input observations

        Parameters
        ----------
        id: int
            ID of this observation
        mbobs: ngmix.MultiBandObsList
            ngmix multi-band observation.  we may loosen this to be  alist
            of them, for deblending

        Returns
        -------
        results : `dict`
            Dictionary of outputs, with keys matching the fields added in
            `defineSchema()`.
        """

        if self.cdict['make_plots']:
            self._make_plots(id, mbobs)

        boot=self._get_bootstrapper(mbobs)
        boot.fit_psfs()
        boot.fit_psf_fluxes()
        if boot.result['psf']['flags'] !=0:
            self.log.info('    skipping model fit due psf fit failure')
        elif boot.result['psf_flux']['flags']!=0:
            self.log.info('    skipping model fit due psf flux fit failure')
        else:
            boot.fit_model()

        return boot.result

    def _get_bootstrapper(self, mbobs):
        """
        get a bootstrapper to automate the processing
        """
        return bootstrap.MaxBootstrapper(
            mbobs,
            self.cdict,
            self.prior,
            self.rng,
        )

    def _copy_result(self, mbobs, res, output):
        """
        copy the result dict to the output record
        """
        n=self.get_namer()
        stamp_shape = mbobs[0][0].image.shape
        stamp_size=stamp_shape[0]

        output[n('flags')] = res['flags']
        output[n('stamp_size')] = stamp_size

        self._copy_psf_fit_result(res['psf'], output)
        self._copy_psf_fit_results_byband(res['psf'], output)
        self._copy_psf_flux_results_byband(res['psf_flux'], output)

        self._copy_model_result(res['obj'], output)

    def _copy_psf_fit_result(self, pres, output):
        """
        copy the PSF result dict to the output record.
        The statistics here are averaged over all bands
        """

        n=self.get_psf_namer()
        output[n('flags')] = pres['flags']
        if pres['flags'] == 0:
            output[n('g1_mean')] = pres['g1_mean']
            output[n('g2_mean')] = pres['g2_mean']
            output[n('T_mean')]  = pres['T_mean']

    def _copy_psf_fit_results_byband(self, pres, output):
        """
        copy the PSF result from each band to the output record.
        """

        config=self.cdict
        for ifilt,filt in enumerate(config['filters']):
            filt_res = pres['byband'][ifilt]

            if filt_res is not None:
                n=self.get_psf_namer(filt=filt)

                output[n('flags')] = filt_res['flags']
                output[n('row')] = filt_res['pars'][0]
                output[n('col')] = filt_res['pars'][1]
                if filt_res['flags']==0:
                    for name in ['g1','g2','T']:
                        output[n(name)] = filt_res[name]

    def _copy_psf_flux_results_byband(self, pres, output):
        """
        copy the PSF flux fitting results from each band to the output record.
        """

        config=self.cdict
        for ifilt,filt in enumerate(config['filters']):
            filt_res = pres['byband'][ifilt]

            if filt_res is not None:
                n=self.get_psf_flux_namer(filt=filt)

                output[n('flux_flags')] = filt_res['flags']
                if filt_res['flags']==0:
                    output[n('flux')] = filt_res['flux']
                    output[n('flux_err')] = filt_res['flux_err']


    def _copy_model_result(self, ores, output):
        """
        copy the model fitting result dict to the output record.
        """

        config=self.cdict

        mn=self.get_model_namer()
        output[mn('flags')] = ores['flags']

        if 'nfev' in ores:
            # can be there even if the fit failed, but won't be there
            # if it wasn't attempted
            output[mn('nfev')] = ores['nfev']

        if ores['flags']==0:
            for n in ['chi2per','dof','s2n']:
                output[mn(n)] = ores[n]

            ni=[('row',0),('col',1),('g1',2),('g2',3),('T',4)]
            if self.cdict['obj']['model'] in ['bd','bdf']:
                ni += [('fracdev',5)]
                flux_start=6
            else:
                flux_start=5

            pars=ores['pars']
            perr=ores['pars_err']
            for n,i in ni:
                output[mn(n)] = pars[i]
                output[mn(n+'_err')] = perr[i]

            for ifilt, filt in enumerate(config['filters']):

                ind=flux_start+ifilt
                mfn=self.get_model_flux_namer(filt)
                output[mfn('flux')] = pars[ind]
                output[mfn('flux_err')] = perr[ind]

    def get_namer(self):
        """
        get a namer for this output type
        """
        return Namer(front='ngmix')

    def get_model_namer(self):
        """
        get a namer for this output type
        """
        config=self.cdict
        model=config['obj']['model']
        return Namer(front='ngmix_%s' % model)


    def get_psf_namer(self, filt=None):
        """
        get a namer for this output type
        """
        front='ngmix_psf'
        if filt is not None:
            front='%s_%s' % (front,filt)
        return Namer(front=front)


    def get_model_flux_namer(self, filt):
        """
        get a namer for this output type
        """
        config=self.cdict
        model=config['obj']['model']
        front='ngmix_%s' % model
        return Namer(front=front, back=filt)

    def get_psf_flux_namer(self, filt):
        """
        get a namer for this output type
        """
        front='ngmix_psf'
        return Namer(front=front, back=filt)


    @property
    def prior(self):
        """
        set the joint prior used for object fitting
        """
        if not hasattr(self, '_prior'):
            # this is temporary until I can figure out how to get
            # an existing seeded rng

            conf=self.cdict
            nband=len(conf['filters'])
            model=conf['obj']['model']
            self._prior = priors.get_joint_prior(
                conf['obj'],
                nband,
                self.rng,
            )

        return self._prior



class ProcessCoaddsMetacalMaxTask(ProcessCoaddsNGMixBaseTask):
    _DefaultName = "processCoaddsMetacalMax"
    ConfigClass = ProcessCoaddsMetacalMaxConfig

    def defineSchema(self, refSchema):
        """Return the Schema for the output catalog.

        This may add or modify self.

        Parameters
        ----------
        refSchema : `lsst.afw.table.Schema`
            Schema of the input reference catalogs.

        Returns
        -------
        outputSchema : `lsst.afw.table.Schema`
            Schema of the output catalog.  Will be added as ``self.schema``
            by calling code.
        """
        self.mapper = SchemaMapper(refSchema)
        self.mapper.addMinimalSchema(SourceCatalog.Table.makeMinimalSchema(), True)
        schema = self.mapper.getOutputSchema()

        config=self.cdict

        model=config['obj']['model']
        n=self.get_namer()
        pn=self.get_psf_namer()

        # generic ngmix fields
        mtypes=[
            (n('flags'),'overall flags for the processing',np.int32,''),
            (n('stamp_size'),'size of postage stamp',np.int32,''),
        ]

        # psf fitting related fields
        mtypes += [
            (pn('flags'),'overall flags for the PSF processing',np.int32,''),

            # mean over filters
            (pn('g2_mean'),'mean over filters of component 2 of the PSF ellipticity',np.float64,''),
            (pn('g1_mean'),'mean over filters of component 2 of the PSF ellipticity',np.float64,''),
            (pn('T_mean'),'mean over filters <x^2> + <y^2> for the gaussian mixture',np.float64,'arcsec^2'),
        ]

        # PSF measurements by filter
        for filt in config['filters']:
            pfn=self.get_psf_namer(filt=filt)
            mtypes += [
                (pfn('flags'), 'overall flags for PSF processing in %s filter' % filt, np.int32, ''),
                (pfn('row'),'offset from canonical row position',np.float64,'arcsec'),
                (pfn('col'),'offset from canonical col position',np.float64,'arcsec'),
                (pfn('g1'), 'component 1 of the PSF ellipticity in %s filter' % filt, np.float64, ''),
                (pfn('g2'), 'component 2 of the PSF ellipticity in %s filter' % filt, np.float64, ''),
                (pfn('T'), '<x^2> + <y^2> for the PSF in %s filter' % filt, np.float64, 'arcsec^2'),
            ]

        # PSF flux measurements, on the object, by filter
        #for filt in config['filters']:
        #    pfn=self.get_psf_flux_namer(filt)
        #    mtypes += [
        #        (pfn('flux_flags'),'flags for PSF template flux fitting in the %s filter' % filt,np.float64,''),
        #        (pfn('flux'),'PSF template flux in the %s filter' % filt,np.float64,''),
        #        (pfn('flux_err'),'error on PSF template flux in the %s filter' % filt,np.float64,''),
        #    ]

        # object fitting related fields
        for type in config['metacal']['types']:
            mn=self.get_model_namer(type=type)
            mtypes += [
                (mn('flags'),'flags for model fit',np.int32,''),
                (mn('nfev'),'number of function evaluations during fit',np.int32,''),
                (mn('chi2per'),'chi^2 per degree of freedom',np.float64,''),
                (mn('dof'),'number of degrees of freedom',np.int32,''),

                (mn('s2n'),'S/N for the fit',np.float64,''),

                (mn('row'),'offset from canonical row position',np.float64,'arcsec'),
                (mn('row_err'),'error on offset from canonical row position',np.float64,'arcsec'),
                (mn('col'),'offset from canonical col position',np.float64,'arcsec'),
                (mn('col_err'),'error on offset from canonical col position',np.float64,'arcsec'),
                (mn('g1'),'component 1 of the ellipticity',np.float64,''),
                (mn('g1_err'),'error on component 1 of the ellipticity',np.float64,''),
                (mn('g2'),'component 2 of the ellipticity',np.float64,''),
                (mn('g2_err'),'error on component 2 of the ellipticity',np.float64,''),
                (mn('T'),'<x^2> + <y^2> for the gaussian mixture',np.float64,'arcsec^2'),
                (mn('T_err'),'error on <x^2> + <y^2> for the gaussian mixture',np.float64,'arcsec^2'),
            ]
            if model in ['bd','bdf']:
                mtypes += [
                    (mn('fracdev'),'fraction of light in the bulge',np.float64,''),
                    (mn('fracdev_err'),'error on fraction of light in the bulge',np.float64,''),
                ]

            for filt in config['filters']:
                mfn=self.get_model_flux_namer(filt, type=type)
                mtypes += [
                    (mfn('flux'),'flux in the %s filter' % filt,np.float64,''),
                    (mfn('flux_err'),'error on flux in the %s filter' % filt,np.float64,''),
                ]

        for name,doc,dtype,units in mtypes:
            schema.addField(
                name,
                type=dtype,
                doc=doc,
                units=units,
            )

        return schema

    def _process_observations(self, id, mbobs):
        """
        process the input observations

        Parameters
        ----------
        mbobs: ngmix.MultiBandObsList
            ngmix multi-band observation.  we may loosen this to be  alist
            of them, for deblending

        Returns
        -------
        id: int
            ID of this observation
        results : `dict`
            Dictionary of outputs, with keys matching the fields added in
            `defineSchema()`.
        """

        if self.cdict['make_plots']:
            self._make_plots(id, mbobs)

        boot=self._get_bootstrapper(mbobs)
        boot.go()
        return boot.result

    def _get_bootstrapper(self, mbobs):
        """
        get a bootstrapper to automate the processing
        """
        return bootstrap.MetacalMaxBootstrapper(
            mbobs,
            self.cdict,
            self.prior,
            self.rng,
        )

    def _copy_result(self, mbobs, res, output):
        """
        copy the result dict to the output record
        """

        n=self.get_namer()
        stamp_shape = mbobs[0][0].image.shape
        stamp_size=stamp_shape[0]

        output[n('flags')] = res['mcal_flags']
        output[n('stamp_size')] = stamp_size

        self._copy_psf_fit_result(res['noshear']['psf'], output)
        self._copy_psf_fit_results_byband(res['noshear']['psf'], output)

        #self._copy_psf_flux_results_byband(res['psf_flux'], output)

        self._copy_model_result(res, output)

    def _copy_psf_fit_result(self, pres, output):
        """
        copy the PSF result dict to the output record.
        The statistics here are averaged over all bands
        """

        n=self.get_psf_namer()
        output[n('flags')] = pres['flags']
        if pres['flags'] == 0:
            output[n('g1_mean')] = pres['g1_mean']
            output[n('g2_mean')] = pres['g2_mean']
            output[n('T_mean')]  = pres['T_mean']

    def _copy_psf_fit_results_byband(self, pres, output):
        """
        copy the PSF result from each band to the output record.
        """

        config=self.cdict
        for ifilt,filt in enumerate(config['filters']):
            filt_res = pres['byband'][ifilt]

            if filt_res is not None:
                n=self.get_psf_namer(filt=filt)

                output[n('flags')] = filt_res['flags']
                output[n('row')] = filt_res['pars'][0]
                output[n('col')] = filt_res['pars'][1]
                if filt_res['flags']==0:
                    for name in ['g1','g2','T']:
                        output[n(name)] = filt_res[name]

    def _copy_psf_flux_results_byband(self, pres, output):
        """
        copy the PSF flux fitting results from each band to the output record.
        """

        config=self.cdict
        for ifilt,filt in enumerate(config['filters']):
            filt_res = pres['byband'][ifilt]

            if filt_res is not None:
                n=self.get_psf_flux_namer(filt=filt)

                output[n('flux_flags')] = filt_res['flags']
                if filt_res['flags']==0:
                    output[n('flux')] = filt_res['flux']
                    output[n('flux_err')] = filt_res['flux_err']


    def _copy_model_result(self, res, output):
        """
        copy the model fitting result dict to the output record.
        """

        config=self.cdict

        types=config['metacal']['types']

        for type in types:
            ores=res[type]['obj']

            mn=self.get_model_namer(type=type)
            output[mn('flags')] = ores['flags']

            if 'nfev' in ores:
                # can be there even if the fit failed, but won't be there
                # if it wasn't attempted
                output[mn('nfev')] = ores['nfev']

            if ores['flags']==0:
                for n in ['chi2per','dof','s2n']:
                    output[mn(n)] = ores[n]

                ni=[('row',0),('col',1),('g1',2),('g2',3),('T',4)]
                if self.cdict['obj']['model'] in ['bd','bdf']:
                    ni += [('fracdev',5)]
                    flux_start=6
                else:
                    flux_start=5

                pars=ores['pars']
                perr=ores['pars_err']
                for n,i in ni:
                    output[mn(n)] = pars[i]
                    output[mn(n+'_err')] = perr[i]

                for ifilt, filt in enumerate(config['filters']):

                    ind=flux_start+ifilt
                    mfn=self.get_model_flux_namer(filt, type=type)
                    output[mfn('flux')] = pars[ind]
                    output[mfn('flux_err')] = perr[ind]

    def get_namer(self, type=None):
        """
        get a namer for this output type
        """
        front='mcal'

        back=None
        if type is not None:
            if type=='noshear':
                back=None
            else:
                back=type

        return Namer(front='mcal', back=back)

    def get_psf_namer(self, filt=None):
        """
        get a namer for this output type
        """
        front='mcal_psf'
        if filt is not None:
            front='%s_%s' % (front,filt)
        return Namer(front=front)

    def get_model_namer(self, type=None):
        """
        get a namer for this output type
        """
        config=self.cdict
        model=config['obj']['model']

        front='mcal_%s' % model

        if type is not None:
            if type=='noshear':
                back=None
            else:
                back=type

        return Namer(front=front, back=back)

    def get_model_flux_namer(self, filt, type=None):
        """
        get a namer for this output type
        """
        config=self.cdict
        model=config['obj']['model']
        front='mcal_%s' % model
        back=filt

        if type is not None:
            if type!='noshear':
                back='%s_%s' % (back, type)

        return Namer(front=front, back=back)

    def get_psf_flux_namer(self, filt):
        """
        get a namer for this output type
        """
        raise NotImplementedError('make work for metacal')
        front='mcal_psf'
        return Namer(front=front, back=filt)


    @property
    def prior(self):
        """
        set the joint prior used for object fitting
        """
        if not hasattr(self, '_prior'):
            # this is temporary until I can figure out how to get
            # an existing seeded rng

            conf=self.cdict
            nband=len(conf['filters'])
            model=conf['obj']['model']
            self._prior = priors.get_joint_prior(
                conf['obj'],
                nband,
                self.rng,
            )

        return self._prior

#
# versions for deblended coadds
# we need an entirely new class to write a different output file
#

class ProcessDeblendedCoaddsNGMixMaxTask(ProcessCoaddsNGMixMaxTask):
    """
    need a different class to write a file with a different name.  This
    one is for deblended coadds
    """
    _DefaultName = "processDeblendedCoaddsNGMixMax"
    ConfigClass = ProcessDeblendedCoaddsNGMixMaxConfig

    def run(self, images, ref, replacers, imageId):
        """
        make sure we are using the deblended images
        """
        check = (
            replacers is not None
            and
            self.cdict['useDeblends'] is True
        )
        assert check,\
            'You must set useDeblends=True and send noise replacers'

        return super(ProcessDeblendedCoaddsNGMixMaxTask,self).run(
            images, ref, replacers, imageId,
        )



class ProcessDeblendedCoaddsMetacalMaxTask(ProcessCoaddsMetacalMaxTask):
    """
    need a different class to write a file with a different name.  This
    one is for deblended coadds
    """
    _DefaultName = "processDeblendedCoaddsMetacalMax"
    ConfigClass = ProcessDeblendedCoaddsMetacalMaxConfig

    def run(self, images, ref, replacers, imageId):
        """
        make sure we are using the deblended images
        """
        check = (
            replacers is not None
            and
            self.cdict['useDeblends'] is True
        )
        assert check,\
            'You must set useDeblends=True and send noise replacers'

        return super(ProcessDeblendedCoaddsMetacalMaxTask,self).run(
            images, ref, replacers, imageId,
        )

class MBObsExtractor(object):
    """
    class to extract observations from the images

    parameters
    ----------
    images: dict
        A dictionary of image objects

    """
    def __init__(self, config, images):
        self.config=config
        self.images=images
        self.log=lsst.log.Log.getLogger("meas.extensions.ngmix.MBObsExtractor")

        self._verify()

    def get_mbobs(self, rec):
        """
        make an ngmix.MultiBandObsList for input to the fitter

        parameters
        ----------
        images: dict
            A dictionary of image objects
        rec: object record
            TODO I don't actually know what class this is

        returns
        -------
        mbobs: ngmix.MultiBandObsList
            ngmix multi-band observation list
        """

        mbobs=ngmix.MultiBandObsList()

        for filt in self.config['filters']:
            # TODO: run noise replacers here

            imf = self.images[filt]

            bbox = self._get_bbox(rec, imf)
            subim = _get_padded_sub_image(imf, bbox)

            obs = self._extract_obs(subim, rec)

            obslist=ngmix.ObsList()
            obslist.append(obs)
            mbobs.append(obslist)

        self.log.debug('    stamp shape: %s' % str(mbobs[0][0].image.shape))

        return mbobs

    def _get_bbox(self, rec, imobj):
        """
        get the bounding box for this object

        TODO fine tune the bounding box algorithm

        parameters
        ----------
        rec: object record
            TODO I don't actually know what class this is

        returns
        -------
        bbox:
            TODO I don't actually know what class this is
        """

        stamp_radius, stamp_size = self._compute_stamp_size(rec)
        bbox = _project_box(rec, imobj.getWcs(), stamp_radius)
        return bbox

    def _compute_stamp_size(self, rec):
        """
        calculate a stamp radius for the input object, to
        be used for constructing postage stamp sizes
        """
        sconf = self.config['stamps']

        min_radius = sconf['min_stamp_size']/2
        max_radius = sconf['max_stamp_size']/2

        quad = rec.getShape()
        T = quad.getIxx() + quad.getIyy()
        if np.isnan(T):
            T=4.0

        sigma = np.sqrt(T/2.0)
        radius = sconf['sigma_factor']*sigma

        if radius < min_radius:
            radius = min_radius
        elif radius > max_radius:
            radius = max_radius

        radius = int(np.ceil(radius))
        stamp_size = 2*radius+1

        return radius, stamp_size



    def _extract_obs(self, imobj_sub, rec):
        """
        convert an image object into an ngmix.Observation, including
        a psf observation

        parameters
        ----------
        imobj: an image object
            TODO I don't actually know what class this is
        rec: an object record
            TODO I don't actually know what class this is
            
        returns
        --------
        obs: ngmix.Observation
            The Observation, including 
        """

        im = imobj_sub.image.array
        wt = self._extract_weight(imobj_sub)
        bmask = imobj_sub.mask.array

        #cen = rec.getCentroid()
        orig_cen = imobj_sub.getWcs().skyToPixel(rec.getCoord())
        psf_im = self._extract_psf_image(imobj_sub, orig_cen)
        #psf_im = imobj_sub.getPsf().computeKernelImage(orig_cen).array

        # fake the psf pixel noise
        psf_err = psf_im.max()*0.0001
        psf_wt = psf_im*0 + 1.0/psf_err**2
        
        jacob = self._extract_jacobian(imobj_sub, rec)

        # use canonical center for the psf
        psf_cen = (np.array(psf_im.shape)-1.0)/2.0
        psf_jacob = jacob.copy()
        psf_jacob.set_cen(row=psf_cen[0], col=psf_cen[1])

        psf_obs = ngmix.Observation(
            psf_im,
            weight=psf_wt,
            jacobian=psf_jacob,
        )
        obs = ngmix.Observation(
            im,
            weight=wt,
            bmask=bmask,
            jacobian=jacob,
            psf=psf_obs,
        )

        return obs

    def _extract_psf_image(self, stamp, orig_pos):
        """
        get the psf associated with this stamp

        coadded psfs are generally not square, so we will
        trim it to be square and preserve the center to
        be at the new canonical center
        """
        psfobj = stamp.getPsf()
        psfim  = psfobj.computeKernelImage(orig_pos).array
        psfim  = np.array(psfim, dtype='f4', copy=False)

        psfim = util.trim_odd_image(psfim)
        return psfim


    def _extract_jacobian(self, imobj, rec):
        """
        extract an ngmix.Jacobian from the image object
        and object record

        imobj: an image object
            TODO I don't actually know what class this is
        rec: an object record
            TODO I don't actually know what class this is
            
        returns
        --------
        Jacobian: ngmix.Jacobian
            The local jacobian
        """

        xy0 = imobj.getXY0()

        orig_cen = imobj.getWcs().skyToPixel(rec.getCoord())
        cen = orig_cen - Extent2D(xy0)
        row=cen.getY()
        col=cen.getX()

        wcs = imobj.getWcs().linearizePixelToSky(
            orig_cen,
            afwGeom.arcseconds,
        )
        jmatrix = wcs.getLinear().getMatrix()

        jacob = ngmix.Jacobian(
            row=row,
            col=col,
            dudrow = jmatrix[0,0],
            dudcol = jmatrix[0,1],
            dvdrow = jmatrix[1,0],
            dvdcol = jmatrix[1,1],
        )

        self.log.debug("jacob: %s" % repr(jacob))
        return jacob


    def _extract_weight(self, imobj):
        """
        TODO get the estimated sky variance rather than this hack
        TODO should we zero out other bits?

        extract a weight map

        Areas with NO_DATA will get zero weight.

        Because the variance map includes the full poisson variance, which
        depends on the signal, we instead extract the median of the parts of
        the image without NO_DATA set

        parameters
        ----------
        imobj: an image object
            TODO I don't actually know what class this is
        """
        var_image  = imobj.variance.array
        maskobj = imobj.mask
        mask = maskobj.array


        weight = var_image.copy()

        weight[:,:]=0

        bitnames_to_ignore = self.config['stamps']['bits_to_ignore_for_weight']

        bits_to_ignore=_get_ored_bits(maskobj, bitnames_to_ignore)

        wuse=np.where(
            (var_image > 0)
            &
            ( (mask & bits_to_ignore) == 0 )
        )

        if wuse[0].size > 0:
            medvar = np.median(var_image[wuse])
            weight[:,:] = 1.0/medvar
        else:
            self.log.info('    weight is all zero, found none that passed cuts')
            #_print_bits(maskobj, bitnames_to_ignore)

        bitnames_to_null = self.config['stamps']['bits_to_null']
        if len(bitnames_to_null) > 0:
            bits_to_null=_get_ored_bits(maskobj, bitnames_to_null)
            wnull=np.where( (mask & bits_to_null) != 0 )
            if wnull[0].size > 0:
                self.log.debug('    nulling %d in weight' % wnull[0].size)
                weight[wnull] = 0.0

        return weight


    def _verify(self):
        """
        check for consistency between the images. 
        
        TODO An assertion is currently used, we may want to raise an appropriate
        exception
        """
        xy0=None
        for filt in self.config['filters']:
            imf = self.images[filt]
            if xy0 is None:
                xy0 = imf.getXY0()
            else:
                assert xy0 == imf.getXY0(),\
                        "all images must have same reference position"

        if set(self.images.keys()) != set(self.config['filters']):
            raise RuntimeError("One or more filters missing.")



def _project_box(source, wcs, radius):
    """
    create a box for the input source
    """
    pixel = afwGeom.Point2I(wcs.skyToPixel(source.getCoord()))
    box = afwGeom.Box2I()
    box.include(pixel)
    box.grow(radius)
    return box

def _get_padded_sub_image(original, bbox):
    """
    extract a sub-image, padded out when it is not contained
    """
    region = original.getBBox()

    if region.contains(bbox):
        return original.Factory(original, bbox, afwImage.PARENT, True)

    result = original.Factory(bbox)
    bbox2 = afwGeom.Box2I(bbox)
    bbox2.clip(region)
    if isinstance(original, afwImage.Exposure):
        result.setPsf(original.getPsf())
        result.setWcs(original.getWcs())
        result.setCalib(original.getCalib())
        #result.image.array[:, :] = float("nan")
        result.image.array[:, :] = 0.0
        result.variance.array[:, :] = float("inf")
        result.mask.array[:, :] = np.uint16(result.mask.getPlaneBitMask("NO_DATA"))
        subIn = afwImage.MaskedImageF(original.maskedImage, bbox=bbox2,
                                      origin=afwImage.PARENT, deep=False)
        result.maskedImage.assign(subIn, bbox=bbox2, origin=afwImage.PARENT)
    elif isinstance(original, afwImage.ImageI):
        result.array[:, :] = 0
        subIn = afwImage.ImageI(original, bbox=bbox2,
                               origin=afwImage.PARENT, deep=False)
        result.assign(subIn, bbox=bbox2, origin=afwImage.PARENT)
    else:
        raise ValueError("Image type not supported")
    return result

def make_image_plots(id, images, ncols = 1, titles = None, show=False):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    ncols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(ncols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    assert((titles is None)or (len(images) == len(titles)))

    n_images = len(images)
    nrows = np.ceil(n_images/float(ncols))

    if titles is None:
        titles = ['Image (%d)' % i for i in range(1,n_images + 1)]

    dpi=96
    width=20
    fig = plt.figure(figsize=(width,width*nrows/ncols), dpi=dpi)

    fig.suptitle(
        'id: %d' % id
    )
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(nrows, ncols, n + 1)
        if image.ndim == 2:
            plt.gray()
        implt=plt.imshow(image, interpolation='nearest')
        divider = make_axes_locatable(a)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(implt, cax=cax)
        a.set_title(title)

    #fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.tight_layout()
    if show:
        plt.show()
    return plt

def _get_ored_bits(maskobj, bitnames):
    bits=0
    for ibit,bitname in enumerate(bitnames):
        bitval = maskobj.getPlaneBitMask(bitname)
        bits |= bitval

    return bits

def _print_bits(maskobj, bitnames):
    mask=maskobj.array
    bits=0
    for ibit,bitname in enumerate(bitnames):
        bitval = maskobj.getPlaneBitMask(bitname)
        w=np.where( (mask & bitval) != 0 )
        if w[0].size > 0:
            print('%s %d %d/%d' % (bitname,bitval,w[0].size,mask.size))


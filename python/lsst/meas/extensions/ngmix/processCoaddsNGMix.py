"""
Some TODO items (there are many more below in the code)

    - tag a new ngmix, make known to Jim
        - currently using v1.1.0rc1
        - numba needed too, talk to DC2 people (easy)

    - get a real estimate of the background noise. I am faking this
      by taking the median of the weight map, which includes the
      object poisson noise.  metacal is not tested with object poisson
      noise included

        - also for the deblended coadds, there is noise missing because it gets
          sucked into siblings

    - deal properly with the mask plane. I've got some better handling
      but will probably always need re-evaluation

    - add weighted moments

    - maybe aperture corrections?

    - save all parameters for the PSF.  Because the number of parameters can
    vary a lot, this would require either very special code or saving an array
    (preferred for ease of coding)
        - for now do an array
        - key = schema.addField("name", doc="doc", type="ArrayF", size=0)
         set size when schema defined?  yes, should be able to
         based on the ngmix config

    - metacal
        - make sure we are doing the right thing with the weight map.
        - This means checking to see that the underlying metacal
        code is working appropriately.
        - also need to record somewhere an amount of masking that
        is in place so we can cut on that.

    - add Tasks for multi-object fitting (MOF), which will require
      making stamps for multiple objects and fitting simultaneously
          - let's start with the DM groups
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
import ngmix
from lsst.afw.table import SourceCatalog, SchemaMapper
from lsst.pipe.base import Struct

from .processCoaddsTogether import ProcessCoaddsTogetherTask
from . import util
from .util import Namer
from .mbobs_extractor import MBObsExtractor, MBObsMissingDataError
from . import bootstrap
from . import priors
from . import procflags

from .config import (
    BasicProcessConfig,
    ProcessCoaddsNGMixMaxConfig,
    ProcessCoaddsMetacalMaxConfig,
    ProcessDeblendedCoaddsNGMixMaxConfig,
    ProcessDeblendedCoaddsMetacalMaxConfig,
)


import pprint

__all__ = (
    'ProcessCoaddsNGMixMaxTask',
    'ProcessCoaddsMetacalMaxTask',
    'ProcessDeblendedCoaddsNGMixMaxTask',
    'ProcessDeblendedCoaddsMetacalMaxTask',
)


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
        if not hasattr(self, '_cdict'):
            self._cdict = self.config.toDict()
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

        config = self.cdict
        self.log.info(pprint.pformat(config))

        try:
            extractor = self._get_extractor(images)
        except MBObsMissingDataError as err:
            self.log.info(str(err))
            extractor = None

        # Make an empty catalog
        output = SourceCatalog(self.schema)

        # Add mostly-empty rows to it, copying IDs from the ref catalog.
        output.extend(ref, mapper=self.mapper)

        index_range = self.get_index_range(output)

        for n, (outRecord, refRecord) in enumerate(zip(output, ref)):
            if n < index_range[0] or n > index_range[1]:
                continue

            if (n % 100) == 0:
                self.log.info('index: %06d/%06d' % (n, index_range[1]))
            nproc += 1

            outRecord.setFootprint(None)  # copied from ref; don't need to write these again

            # Insert the deblended pixels for just this object into all images.
            if replacers is not None:
                for r in replacers.values():
                    r.insertSource(refRecord.getId())

            if extractor is None:
                # we were missing a band most likely
                res = self._get_default_result()
                mbobs = None
            else:

                try:
                    mbobs = extractor.get_mbobs(refRecord)
                    res = self._process_observations(ref['id'][n], mbobs)
                except MBObsMissingDataError as err:
                    self.log.debug(str(err))
                    mbobs = None
                    res = self._get_default_result()
                except ngmix.GMixFatalError as err:
                    # this occurs when we have an all zero weight map
                    self.log.debug(str(err))
                    mbobs = None
                    res = self._get_default_result()
                    res['flags'] = procflags.HIGH_MASKFRAC

            self._copy_result(mbobs, res, outRecord)

            # Remove the deblended pixels for this object so we can process
            # the next one.
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

    def _check_obs(self, mbobs, maskfrac_byband):
        """
        check for image bits that we skip, mask fraction, and
        other things
        """

        flags = 0
        flags |= self._check_bitmask(mbobs)
        flags |= self._check_masked_frac(maskfrac_byband)

        return flags

    def _check_bitmask(self, mbobs):
        """
        check to see if bits are set that we do not allow
        in the postage stamp
        """
        flags = 0

        bitnames_to_cut = self.cdict['stamps']['bits_to_cut']
        for iband, obslist in enumerate(mbobs):
            obs = obslist[0]

            if len(bitnames_to_cut) > 0:
                maskobj = obs.meta['maskobj']
                bits_to_cut = util.get_ored_bits(maskobj, bitnames_to_cut)
                w = np.where((obs.bmask & bits_to_cut) != 0)
                if w[0].size > 0:

                    band = self.cdict['filters'][iband]

                    self.log.debug(
                        'setting IMAGE_FLAGS '
                        'because in band %s one of these '
                        'are set %s' % (band, str(bitnames_to_cut))
                    )
                    flags |= procflags.IMAGE_FLAGS

        return flags

    def _check_masked_frac(self, maskfrac_byband):
        """
        check to see if the masked fraction is too high
        in any of the stamps
        """

        flags = 0

        mzfrac = self.cdict['stamps']['max_zero_weight_frac']
        if mzfrac >= 1.0:
            return flags

        for iband, frac in enumerate(maskfrac_byband):
            if frac > mzfrac:
                band = self.cdict['filters'][iband]
                self.log.debug(
                    'setting HIGH_MASKFRAC in filter %s '
                    'because zero weight frac '
                    'exceeds max: %g > %g' % (band, frac, mzfrac)
                )
                flags |= procflags.HIGH_MASKFRAC

        return flags

    def _get_masked_fraction(self, mbobs):
        """
        check to see if the masked fraction is too high
        in any of the stamps
        """

        nband = len(mbobs)
        maskfrac_byband = np.zeros(nband)

        for band, obslist in enumerate(mbobs):
            for obs in obslist:

                w = np.where(obs.weight <= 0.0)
                maskfrac_byband[band] = w[0].size/float(obs.weight.size)

        maskfrac = maskfrac_byband.mean()
        return maskfrac, maskfrac_byband

    def _get_default_result(self):
        """
        get the default result dict
        """
        raise NotImplementedError('implement in child class')

    def get_index_range(self, cat):
        """
        Get the range of indices to process.  If these were not set in the
        configuration, [0,cat.size] is returned
        """
        if not hasattr(self, '_index_range'):

            ntot = len(cat)
            start = self.cdict['start_index']
            num = self.cdict['num_to_process']
            if start is None:
                start = 0
            else:
                if start < 0 or start > ntot-1:
                    raise ValueError(
                        'requested start index %d out '
                        'of bounds [%d,%d]' % (start, ntot-1)
                    )

            if num is None:
                num = ntot-start
            if num < 1:
                raise ValueError(
                    'requested number to process %d '
                    'less than 1' % num
                )

            self._index_range = [start, start+num-1]

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
        filters = self.cdict['filters']

        imlist = [o[0].image for o in mbobs]
        titles = [f for f in filters]

        imlist += [o[0].weight for o in mbobs]
        titles += [f+' wt' for f in filters]

        imlist += [o[0].bmask for o in mbobs]
        titles += [f+' bmask' for f in filters]

        plt = make_image_plots(
            id,
            imlist,
            ncols=len(filters),
            titles=titles,
            show=False,
        )

        fname = 'images-%d.png' % id
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

        config = self.cdict

        model = config['obj']['model']
        n = self.get_namer()
        pn = self.get_psf_namer()
        mn = self.get_model_namer()

        # generic ngmix fields
        mtypes = [
            (n('flags'), 'overall flags for the processing', np.int32, ''),
            (n('stamp_size'), 'size of postage stamp', np.int32, ''),
            (n('maskfrac'), 'mean masked fraction', np.float32, ''),
        ]
        for filt in config['filters']:
            mtypes += [
                (n('maskfrac_%s' % filt), 'masked fraction in %s filter' % filt, np.float32, ''),
            ]

        # psf fitting related fields
        mtypes += [
            (pn('flags'), 'overall flags for the PSF processing', np.int32, ''),

            # mean over filters
            (pn('g2_mean'), 'mean over filters of component 2 of the PSF ellipticity', np.float64, ''),
            (pn('g1_mean'), 'mean over filters of component 2 of the PSF ellipticity', np.float64, ''),
            (pn('T_mean'), 'mean over filters <x^2> + <y^2> for the gaussian mixture', np.float64,
             'arcsec^2'),
        ]

        # PSF measurements by filter
        for filt in config['filters']:
            pfn = self.get_psf_namer(filt=filt)
            mtypes += [
                (pfn('flags'), 'overall flags for PSF processing in %s filter' % filt, np.int32, ''),
                (pfn('row'), 'offset from canonical row position', np.float64, 'arcsec'),
                (pfn('col'), 'offset from canonical col position', np.float64, 'arcsec'),
                (pfn('g1'), 'component 1 of the PSF ellipticity in %s filter' % filt, np.float64, ''),
                (pfn('g2'), 'component 2 of the PSF ellipticity in %s filter' % filt, np.float64, ''),
                (pfn('T'), '<x^2> + <y^2> for the PSF in %s filter' % filt, np.float64, 'arcsec^2'),
            ]

        # PSF flux measurements, on the object, by filter
        for filt in config['filters']:
            pfn = self.get_psf_flux_namer(filt)
            mtypes += [
                (pfn('flux_flags'), 'flags for PSF template flux fitting in the %s filter' % filt,
                 np.float64, ''),
                (pfn('flux'), 'PSF template flux in the %s filter' % filt, np.float64, ''),
                (pfn('flux_err'), 'error on PSF template flux in the %s filter' % filt, np.float64, ''),
            ]

        # object fitting related fields
        mtypes += [
            (mn('flags'), 'flags for model fit', np.int32, ''),
            (mn('nfev'), 'number of function evaluations during fit', np.int32, ''),
            (mn('chi2per'), 'chi^2 per degree of freedom', np.float64, ''),
            (mn('dof'), 'number of degrees of freedom', np.int32, ''),

            (mn('s2n'), 'S/N for the fit', np.float64, ''),

            (mn('row'), 'offset from canonical row position', np.float64, 'arcsec'),
            (mn('row_err'), 'error on offset from canonical row position', np.float64, 'arcsec'),
            (mn('col'), 'offset from canonical col position', np.float64, 'arcsec'),
            (mn('col_err'), 'error on offset from canonical col position', np.float64, 'arcsec'),
            (mn('g1'), 'component 1 of the ellipticity', np.float64, ''),
            (mn('g1_err'), 'error on component 1 of the ellipticity', np.float64, ''),
            (mn('g2'), 'component 2 of the ellipticity', np.float64, ''),
            (mn('g2_err'), 'error on component 2 of the ellipticity', np.float64, ''),
            (mn('T'), '<x^2> + <y^2> for the gaussian mixture', np.float64, 'arcsec^2'),
            (mn('T_err'), 'error on <x^2> + <y^2> for the gaussian mixture', np.float64, 'arcsec^2'),
        ]
        if model in ['bd', 'bdf']:
            mtypes += [
                (mn('fracdev'), 'fraction of light in the bulge', np.float64, ''),
                (mn('fracdev_err'), 'error on fraction of light in the bulge', np.float64, ''),
            ]

        for filt in config['filters']:
            mfn = self.get_model_flux_namer(filt)
            mtypes += [
                (mfn('flux'), 'flux in the %s filter' % filt, np.float64, ''),
                (mfn('flux_err'), 'error on flux in the %s filter' % filt, np.float64, ''),
            ]

        for name, doc, dtype, units in mtypes:
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

        maskfrac, maskfrac_byband = self._get_masked_fraction(mbobs)

        flags = self._check_obs(mbobs, maskfrac_byband)
        if flags != 0:
            res = self._get_default_result()
            res['maskfrac'], res['maskfrac_byband'] = maskfrac, maskfrac_byband
            res['flags'] = flags
            return res

        boot = self._get_bootstrapper(mbobs)
        boot.fit_psfs()
        boot.fit_psf_fluxes()
        if boot.result['psf']['flags'] != 0:
            self.log.warn('    skipping model fit due psf fit failure')
        elif boot.result['psf_flux']['flags'] != 0:
            self.log.warn('    skipping model fit due psf flux fit failure')
        else:
            boot.fit_model()

        res = boot.result
        res['maskfrac'], res['maskfrac_byband'] = maskfrac, maskfrac_byband
        return res

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

    def _get_default_result(self):
        """
        get the default result dict
        """
        return bootstrap.get_default_result()

    def _copy_result(self, mbobs, res, output):
        """
        copy the result dict to the output record
        """

        n = self.get_namer()

        if mbobs is None:
            output[n('flags')] = procflags.NO_ATTEMPT
        else:
            output[n('flags')] = res['flags']

            stamp_shape = mbobs[0][0].image.shape
            stamp_size = stamp_shape[0]

            output[n('stamp_size')] = stamp_size
            output[n('maskfrac')] = res['maskfrac']
            for ifilt, filt in enumerate(self.cdict['filters']):
                output[n('maskfrac_%s' % filt)] = res['maskfrac_byband'][ifilt]

            self._copy_psf_fit_result(res['psf'], output)
            self._copy_psf_fit_results_byband(res['psf'], output)
            self._copy_psf_flux_results_byband(res['psf_flux'], output)

            self._copy_model_result(res['obj'], output)

    def _copy_psf_fit_result(self, pres, output):
        """
        copy the PSF result dict to the output record.
        The statistics here are averaged over all bands
        """

        n = self.get_psf_namer()
        output[n('flags')] = pres['flags']
        if pres['flags'] == 0:
            output[n('g1_mean')] = pres['g1_mean']
            output[n('g2_mean')] = pres['g2_mean']
            output[n('T_mean')] = pres['T_mean']

    def _copy_psf_fit_results_byband(self, pres, output):
        """
        copy the PSF result from each band to the output record.
        """

        if len(pres['byband']) == 0:
            return

        config = self.cdict
        for ifilt, filt in enumerate(config['filters']):
            filt_res = pres['byband'][ifilt]

            if filt_res is not None:
                n = self.get_psf_namer(filt=filt)

                output[n('flags')] = filt_res['flags']
                output[n('row')] = filt_res['pars'][0]
                output[n('col')] = filt_res['pars'][1]
                if filt_res['flags'] == 0:
                    for name in ['g1', 'g2', 'T']:
                        output[n(name)] = filt_res[name]

    def _copy_psf_flux_results_byband(self, pres, output):
        """
        copy the PSF flux fitting results from each band to the output record.
        """

        if len(pres['byband']) == 0:
            return

        config = self.cdict
        for ifilt, filt in enumerate(config['filters']):
            filt_res = pres['byband'][ifilt]

            if filt_res is not None:
                n = self.get_psf_flux_namer(filt=filt)

                output[n('flux_flags')] = filt_res['flags']
                if filt_res['flags'] == 0:
                    output[n('flux')] = filt_res['flux']
                    output[n('flux_err')] = filt_res['flux_err']

    def _copy_model_result(self, ores, output):
        """
        copy the model fitting result dict to the output record.
        """

        config = self.cdict

        mn = self.get_model_namer()
        output[mn('flags')] = ores['flags']

        if 'nfev' in ores:
            # can be there even if the fit failed, but won't be there
            # if it wasn't attempted
            output[mn('nfev')] = ores['nfev']

        if ores['flags'] == 0:
            for n in ['chi2per', 'dof', 's2n']:
                output[mn(n)] = ores[n]

            ni = [('row', 0), ('col', 1), ('g1', 2), ('g2', 3), ('T', 4)]
            if self.cdict['obj']['model'] in ['bd', 'bdf']:
                ni += [('fracdev', 5)]
                flux_start = 6
            else:
                flux_start = 5

            pars = ores['pars']
            perr = ores['pars_err']
            for n, i in ni:
                output[mn(n)] = pars[i]
                output[mn(n+'_err')] = perr[i]

            for ifilt, filt in enumerate(config['filters']):

                ind = flux_start+ifilt
                mfn = self.get_model_flux_namer(filt)
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
        config = self.cdict
        model = config['obj']['model']
        return Namer(front='ngmix_%s' % model)

    def get_psf_namer(self, filt=None):
        """
        get a namer for this output type
        """
        front = 'ngmix_psf'
        if filt is not None:
            front = '%s_%s' % (front, filt)
        return Namer(front=front)

    def get_model_flux_namer(self, filt):
        """
        get a namer for this output type
        """
        config = self.cdict
        model = config['obj']['model']
        front = 'ngmix_%s' % model
        return Namer(front=front, back=filt)

    def get_psf_flux_namer(self, filt):
        """
        get a namer for this output type
        """
        front = 'ngmix_psf'
        return Namer(front=front, back=filt)

    @property
    def prior(self):
        """
        set the joint prior used for object fitting
        """
        if not hasattr(self, '_prior'):
            # this is temporary until I can figure out how to get
            # an existing seeded rng

            conf = self.cdict
            nband = len(conf['filters'])
            model = conf['obj']['model']
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

        config = self.cdict

        model = config['obj']['model']
        n = self.get_namer()
        pn = self.get_psf_namer()

        # generic ngmix fields
        mtypes = [
            (n('flags'), 'overall flags for the processing', np.int32, ''),
            (n('stamp_size'), 'size of postage stamp', np.int32, ''),
            (n('maskfrac'), 'mean masked fraction', np.float32, ''),
        ]
        for filt in config['filters']:
            mtypes += [
                (n('maskfrac_%s' % filt), 'masked fraction in %s filter' % filt, np.float32, ''),
            ]

        # psf fitting related fields
        mtypes += [
            (pn('flags'), 'overall flags for the PSF processing', np.int32, ''),

            # mean over filters
            (pn('g2_mean'), 'mean over filters of component 2 of the PSF ellipticity', np.float64, ''),
            (pn('g1_mean'), 'mean over filters of component 2 of the PSF ellipticity', np.float64, ''),
            (pn('T_mean'), 'mean over filters <x^2> + <y^2> for the gaussian mixture', np.float64,
             'arcsec^2'),
        ]

        # PSF measurements by filter
        for filt in config['filters']:
            pfn = self.get_psf_namer(filt=filt)
            mtypes += [
                (pfn('flags'), 'overall flags for PSF processing in %s filter' % filt, np.int32, ''),
                (pfn('row'), 'offset from canonical row position', np.float64, 'arcsec'),
                (pfn('col'), 'offset from canonical col position', np.float64, 'arcsec'),
                (pfn('g1'), 'component 1 of the PSF ellipticity in %s filter' % filt, np.float64, ''),
                (pfn('g2'), 'component 2 of the PSF ellipticity in %s filter' % filt, np.float64, ''),
                (pfn('T'), '<x^2> + <y^2> for the PSF in %s filter' % filt, np.float64, 'arcsec^2'),
            ]

        # PSF flux measurements, on the object, by filter
        # for filt in config['filters']:
        #    pfn=self.get_psf_flux_namer(filt)
        #    mtypes += [
        #        (pfn('flux_flags'),
        #         'flags for PSF template flux fitting in the %s filter' %
        #         filt,np.float64, ''),
        #        (pfn('flux'),'PSF template flux in the %s filter' % filt,
        #         np.float64, ''),
        #        (pfn('flux_err'),
        #         'error on PSF template flux in the %s filter' % filt,
        #         np.float64, ''),
        #    ]

        # object fitting related fields
        for type in config['metacal']['types']:
            mn = self.get_model_namer(type=type)
            mtypes += [
                (mn('flags'), 'flags for model fit', np.int32, ''),
                (mn('nfev'), 'number of function evaluations during fit', np.int32, ''),
                (mn('chi2per'), 'chi^2 per degree of freedom', np.float64, ''),
                (mn('dof'), 'number of degrees of freedom', np.int32, ''),

                (mn('s2n'), 'S/N for the fit', np.float64, ''),

                (mn('row'), 'offset from canonical row position', np.float64, 'arcsec'),
                (mn('row_err'), 'error on offset from canonical row position', np.float64, 'arcsec'),
                (mn('col'), 'offset from canonical col position', np.float64, 'arcsec'),
                (mn('col_err'), 'error on offset from canonical col position', np.float64, 'arcsec'),
                (mn('g1'), 'component 1 of the ellipticity', np.float64, ''),
                (mn('g1_err'), 'error on component 1 of the ellipticity', np.float64, ''),
                (mn('g2'), 'component 2 of the ellipticity', np.float64, ''),
                (mn('g2_err'), 'error on component 2 of the ellipticity', np.float64, ''),
                (mn('T'), '<x^2> + <y^2> for the gaussian mixture', np.float64, 'arcsec^2'),
                (mn('T_err'), 'error on <x^2> + <y^2> for the gaussian mixture', np.float64, 'arcsec^2'),
            ]
            if model in ['bd', 'bdf']:
                mtypes += [
                    (mn('fracdev'), 'fraction of light in the bulge', np.float64, ''),
                    (mn('fracdev_err'), 'error on fraction of light in the bulge', np.float64, ''),
                ]

            for filt in config['filters']:
                mfn = self.get_model_flux_namer(filt, type=type)
                mtypes += [
                    (mfn('flux'), 'flux in the %s filter' % filt, np.float64, ''),
                    (mfn('flux_err'), 'error on flux in the %s filter' % filt, np.float64, ''),
                ]

        for name, doc, dtype, units in mtypes:
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

        # start with a default result.  may not use if we get to
        # measurements
        maskfrac, maskfrac_byband = self._get_masked_fraction(mbobs)

        flags = self._check_obs(mbobs, maskfrac_byband)
        if flags != 0:
            res = self._get_default_result()
            res['maskfrac'], res['maskfrac_byband'] = maskfrac, maskfrac_byband
            res['mcal_flags'] = flags
            return res

        boot = self._get_bootstrapper(mbobs)
        boot.go()
        res = boot.result
        res['maskfrac'], res['maskfrac_byband'] = maskfrac, maskfrac_byband
        return res

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

    def _get_default_result(self):
        """
        get the default result dict
        """
        return bootstrap.get_default_mcal_result()

    def _copy_result(self, mbobs, res, output):
        """
        copy the result dict to the output record
        """

        n = self.get_namer()

        if mbobs is None:
            output[n('flags')] = procflags.NO_ATTEMPT
        else:
            output[n('flags')] = res['mcal_flags']

            stamp_shape = mbobs[0][0].image.shape
            stamp_size = stamp_shape[0]

            output[n('stamp_size')] = stamp_size
            output[n('maskfrac')] = res['maskfrac']
            for ifilt, filt in enumerate(self.cdict['filters']):
                output[n('maskfrac_%s' % filt)] = res['maskfrac_byband'][ifilt]

            self._copy_psf_fit_result(res['noshear']['psf'], output)
            self._copy_psf_fit_results_byband(res['noshear']['psf'], output)

            # self._copy_psf_flux_results_byband(res['psf_flux'], output)

            self._copy_model_result(res, output)

    def _copy_psf_fit_result(self, pres, output):
        """
        copy the PSF result dict to the output record.
        The statistics here are averaged over all bands
        """

        n = self.get_psf_namer()
        output[n('flags')] = pres['flags']
        if pres['flags'] == 0:
            output[n('g1_mean')] = pres['g1_mean']
            output[n('g2_mean')] = pres['g2_mean']
            output[n('T_mean')] = pres['T_mean']

    def _copy_psf_fit_results_byband(self, pres, output):
        """
        copy the PSF result from each band to the output record.
        """

        if len(pres['byband']) == 0:
            return

        config = self.cdict
        for ifilt, filt in enumerate(config['filters']):
            filt_res = pres['byband'][ifilt]

            if filt_res is not None:
                n = self.get_psf_namer(filt=filt)

                output[n('flags')] = filt_res['flags']
                if filt_res['flags'] == 0:
                    output[n('row')] = filt_res['pars'][0]
                    output[n('col')] = filt_res['pars'][1]
                    if filt_res['flags'] == 0:
                        for name in ['g1', 'g2', 'T']:
                            output[n(name)] = filt_res[name]

    def _copy_psf_flux_results_byband(self, pres, output):
        """
        copy the PSF flux fitting results from each band to the output record.
        """
        if len(pres['byband']) == 0:
            return

        config = self.cdict
        for ifilt, filt in enumerate(config['filters']):
            filt_res = pres['byband'][ifilt]

            if filt_res is not None:
                n = self.get_psf_flux_namer(filt=filt)

                output[n('flux_flags')] = filt_res['flags']
                if filt_res['flags'] == 0:
                    output[n('flux')] = filt_res['flux']
                    output[n('flux_err')] = filt_res['flux_err']

    def _copy_model_result(self, res, output):
        """
        copy the model fitting result dict to the output record.
        """

        config = self.cdict

        types = config['metacal']['types']

        for type in types:
            ores = res[type]['obj']

            mn = self.get_model_namer(type=type)
            output[mn('flags')] = ores['flags']

            if 'nfev' in ores:
                # can be there even if the fit failed, but won't be there
                # if it wasn't attempted
                output[mn('nfev')] = ores['nfev']

            if ores['flags'] == 0:
                for n in ['chi2per', 'dof', 's2n']:
                    output[mn(n)] = ores[n]

                ni = [('row', 0), ('col', 1), ('g1', 2), ('g2', 3), ('T', 4)]
                if self.cdict['obj']['model'] in ['bd', 'bdf']:
                    ni += [('fracdev', 5)]
                    flux_start = 6
                else:
                    flux_start = 5

                pars = ores['pars']
                perr = ores['pars_err']
                for n, i in ni:
                    output[mn(n)] = pars[i]
                    output[mn(n+'_err')] = perr[i]

                for ifilt, filt in enumerate(config['filters']):

                    ind = flux_start+ifilt
                    mfn = self.get_model_flux_namer(filt, type=type)
                    output[mfn('flux')] = pars[ind]
                    output[mfn('flux_err')] = perr[ind]

    def get_namer(self, type=None):
        """
        get a namer for this output type
        """
        front = 'mcal'

        back = None
        if type is not None:
            if type == 'noshear':
                back = None
            else:
                back = type

        return Namer(front='mcal', back=back)

    def get_psf_namer(self, filt=None):
        """
        get a namer for this output type
        """
        front = 'mcal_psf'
        if filt is not None:
            front = '%s_%s' % (front, filt)
        return Namer(front=front)

    def get_model_namer(self, type=None):
        """
        get a namer for this output type
        """
        config = self.cdict
        model = config['obj']['model']

        front = 'mcal_%s' % model

        if type is not None:
            if type == 'noshear':
                back = None
            else:
                back = type

        return Namer(front=front, back=back)

    def get_model_flux_namer(self, filt, type=None):
        """
        get a namer for this output type
        """
        config = self.cdict
        model = config['obj']['model']
        front = 'mcal_%s' % model
        back = filt

        if type is not None:
            if type != 'noshear':
                back = '%s_%s' % (back, type)

        return Namer(front=front, back=back)

    def get_psf_flux_namer(self, filt):
        """
        get a namer for this output type
        """
        raise NotImplementedError('make work for metacal')
        front = 'mcal_psf'
        return Namer(front=front, back=filt)

    @property
    def prior(self):
        """
        set the joint prior used for object fitting
        """
        if not hasattr(self, '_prior'):
            # this is temporary until I can figure out how to get
            # an existing seeded rng

            conf = self.cdict
            nband = len(conf['filters'])
            model = conf['obj']['model']
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
            replacers is not None and
            self.cdict['useDeblends'] is True
        )
        assert check,\
            'You must set useDeblends=True and send noise replacers'

        return super(ProcessDeblendedCoaddsNGMixMaxTask, self).run(
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
            replacers is not None and
            self.cdict['useDeblends'] is True
        )
        assert check,\
            'You must set useDeblends=True and send noise replacers'

        return super(ProcessDeblendedCoaddsMetacalMaxTask, self).run(
            images, ref, replacers, imageId,
        )


def make_image_plots(id, images, ncols=1, titles=None, show=False):
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
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]

    dpi = 96
    width = 20
    fig = plt.figure(figsize=(width, width*nrows/ncols), dpi=dpi)

    fig.suptitle(
        'id: %d' % id
    )
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(nrows, ncols, n + 1)
        if image.ndim == 2:
            plt.gray()
        implt = plt.imshow(image, interpolation='nearest')
        divider = make_axes_locatable(a)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(implt, cax=cax)
        a.set_title(title)

    # fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.tight_layout()
    if show:
        plt.show()
    return plt

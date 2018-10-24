"""
big TODO items (more are below in the code)
    - different output file names for different tasks
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

import numpy as np
from lsst.geom import Extent2D
from lsst.afw.table import SourceCatalog, SchemaMapper
import lsst.afw.geom as afwGeom
from lsst.pex.config import Field, ListField, ConfigField, Config, ChoiceField
from lsst.pipe.base import Struct

from .processCoaddsTogether import ProcessCoaddsTogetherConfig, ProcessCoaddsTogetherTask
from .util import Namer

import ngmix

from pprint import pprint

#__all__ = ("ProcessCoaddsNGMixConfig", "ProcessCoaddsNGMixTask")

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
    pars = ListField(dtype=float, doc="parameters for the fracdev prior")

class ObjectPriorsConfig(Config):
    """
    Configuration of priors for the bulge+disk model
    """
    cen = ConfigField(dtype=CenPriorConfig, doc="prior on center")
    g = ConfigField(dtype=GPriorConfig, doc="prior on g")
    T = ConfigField(dtype=TPriorConfig, doc="prior on square size T")
    flux = ConfigField(dtype=FluxPriorConfig, doc="prior on flux")

    # this is optional, only used by the bulge+disk fitter
    fracdev = ConfigField(dtype=FracdevPriorConfig, doc="prior on fracdev")

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
            "bd":"bulge+disk model with fixed size ratio",
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
    ntest = Field(dtype=int, default=None, doc="Do a test with only this many objects")

    def setDefaults(self):
        """
        prefix for the output file
        """
        self.output.name = "deepCoadd_ngmix"

class ProcessCoaddsNGMixMaxConfig(BasicProcessConfig):
    """
    fit the object and PSF using maximum likelihood
    """
    psf = ConfigField(dtype=PSFMaxFitConfig, doc='psf fitting config')
    obj = ConfigField(dtype=ObjectMaxFitConfig,doc="object fitting config")

    #def setDefaults(self):
    #    """
    #    TODO why is this specific to deepCoadd?
    #    """
    #    self.output.name = "deepCoadd_ngmix_max"

class ProcessCoaddsNGMixBaseTask(ProcessCoaddsTogetherTask):
    """
    Base class for ngmix tasks
    """
    _DefaultName = "processCoaddsNGMixBase"
    ConfigClass = BasicProcessConfig

    def get_config(self):
        # we will find it convenient to have a dictionary version of the
        # configuration
        if not hasattr(self,'cdict'):
            self.cdict=self.config.toDict()
        return self.cdict

class ProcessCoaddsNGMixMaxTask(ProcessCoaddsNGMixBaseTask):
    """
    Base class for ngmix tasks
    """
    _DefaultName = "processCoaddsNGMixSimple"
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

        config=self.get_config()

        # TODO: add custom output fields here via calls like
        #
        #    schema.addField("ngmix_r_flux", type=float, doc="flux in the r band", units="Jy")
        #
        # You can iterate over self.config.filters to get the list of filters
        # we expect to see.  If you want to report fluxes in pixel units
        # instead, use an '_instFlux' suffix instead with 'count' as the units.
        #
        # Use type="Flag" to add a boolean flag (for errors and not-errors).
        #
        # For ellipses and positions, you may want to use
        # lsst.afw.table.{QuadrupoleKey, Point2DKey}.addFields to add several
        # related fields at once.
        #
        # Or just add them however you like and DM people can convert them to
        # our conventions later; whatever is easier.

        model=config['obj']['model']
        n=Namer(front='ngmix_%s' % model)

        mtypes=[
            ('flags','overall flags for the processing',np.int32,''),
            ('psf_flags','flags for the PSF processing',np.int32,''),
        ]
        for filt in config['filters']:
            mtypes += [
                ('psf_%s_g1' % filt,
                 'component 1 of the PSF ellipticity in %s filter' % filt,
                 np.float64,
                 ''),

                ('psf_%s_g2' % filt,
                 'component 2 of the PSF ellipticity in %s filter' % filt,
                 np.float64,
                 ''),

                ('psf_%s_T' % filt,
                 '<x^2> + <y^2> for the PSF in %s filter' % filt,
                 np.float64,
                 'arcsec^2'),
            ]

        mtypes += [
            ('psf_g2','mean component 2 of the PSF ellipticity',np.float64,''),
            ('psf_g1','mean component 2 of the PSF ellipticity',np.float64,''),
            ('psf_T','mean <x^2> + <y^2> for the gaussian mixture',np.float64,'arcsec^2'),

            ('stamp_size','size of postage stamp',np.int32,''),
            ('nfev','number of function evaluations during fit',np.int32,''),
            ('chi2per','chi^2 per degree of freedom',np.float64,''),
            ('dof','number of degrees of freedom',np.int32,''),

            ('s2n','S/N for the fit',np.float64,''),

            ('row','offset from canonical row position',np.float64,'arcsec'),
            ('row_err','error on offset from canonical row position',np.float64,'arcsec'),
            ('col','offset from canonical col position',np.float64,'arcsec'),
            ('col_err','error on offset from canonical col position',np.float64,'arcsec'),
            ('g1','component 1 of the ellipticity',np.float64,''),
            ('g1_err','error on component 1 of the ellipticity',np.float64,''),
            ('g2','component 2 of the ellipticity',np.float64,''),
            ('g2_err','error on component 2 of the ellipticity',np.float64,''),
            ('T','<x^2> + <y^2> for the gaussian mixture',np.float64,'arcsec^2'),
            ('T_err','error on <x^2> + <y^2> for the gaussian mixture',np.float64,'arcsec^2'),
        ]
        if model=='bd':
            mtypes += [
                ('fracdev','fraction of light in the bulge',np.float64,''),
                ('fracdev_err','error on fraction of light in the bulge',np.float64,''),
            ]

        for filt in config['filters']:
            mtypes += [
                ('%s_flux' % filt,'flux in the %s filter' % filt,np.float64,''),
                ('%s_flux_err' % filt,'error on flux in the %s filter' % filt,np.float64,''),
            ]

        for tname,doc,dtype,units in mtypes:
            name=n(tname)
            schema.addField(
                name,
                type=dtype,
                doc=doc,
                units=units,
            )

        return schema

    def run(self, images, ref):
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

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            Struct with (at least) an `output` attribute that is a catalog
            to be written as ``self.config.output``.
        """

        config=self.cdict
        pprint(config)

        extractor = self._get_extractor(images)

        # Make an empty catalog
        output = SourceCatalog(self.schema)

        # Add mostly-empty rows to it, copying IDs from the ref catalog.
        output.extend(ref, mapper=self.mapper)

        # TODO: set up noise replacers for using deblender outputs

        for n, (outRecord, refRecord) in enumerate(zip(output, ref)):
            # TODO set up logging
            print(n)

            outRecord.setFootprint(None)  # copied from ref; don't need to write these again

            mbobs = extractor.get_mbobs(refRecord)

            fit = self._fit_object(mbobs)
            for k, v in fit.items():
                outRecord[k] = v

            if config['ntest'] is not None and n == config['ntest']-1:
                break

        return Struct(output=output)

    def _get_extractor(self, images):
        """
        load the appropriate observation extractor
        """
        return MBObsExtractor(self.cdict, images)

    def _fit_object(self, mbobs):
        """Fit a single object.

        Parameters
        ----------
        mbobs: ngmix.MultiBandObsList
            ngmix multi-band observation

        Returns
        -------
        results : `dict`
            Dictionary of outputs, with keys matching the fields added in
            `defineSchema()`.
        """

        # TODO: real work goes here

        return {}


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

        xy0=None
        for filt in self.config['filters']:
            # TODO: run noise replacers here

            imf = self.images[filt]

            bbox = self._get_bbox(rec)

            obs = extract_obs(imf, rec, bbox)

            obslist=ngmix.ObsList()
            obslist.append(obs)
            mbobs.append(obslist)

        return mbobs

    def _get_bbox(self, rec):
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
        return afwGeom.Box2I(rec.getFootprint().getBBox())

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



def extract_obs(imobj, rec, bbox):
    """
    convert an image object into an ngmix.Observation, including
    a psf observation

    TODO cut out a postage stamp

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
    imobj_sub = imobj[bbox]

    im = imobj_sub.image.array
    wt = extract_weight(imobj_sub)

    cen = rec.getCentroid()
    psf_im = imobj.getPsf().computeKernelImage(cen).array

    # fake the psf pixel noise
    psf_err = psf_im.max()*0.0001
    psf_wt = psf_im*0 + 1.0/psf_err**2
    
    jacob = extract_jacobian(imobj, rec)

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
        jacobian=jacob,
        psf=psf_obs,
    )

    return obs

def extract_jacobian(imobj, rec):
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

    orig_cen = rec.getCentroid()
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

    return jacob


def extract_weight(imobj):
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

    zlogic = var_image > 0

    no_data_logic = np.logical_not(
        mask & maskobj.getPlaneBitMask("NO_DATA")
    )
    w=np.where(zlogic & no_data_logic)

    if w[0].size > 0:
        medvar = np.median(var_image[w])
        weight[w] = 1.0/medvar

    return weight




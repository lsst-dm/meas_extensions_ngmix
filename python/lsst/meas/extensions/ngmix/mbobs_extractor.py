import logging
import numpy as np
import ngmix
import lsst.afw.image as afwImage
from lsst.pex.exceptions import InvalidParameterError
import lsst.geom as geom

from . import util


class MBObsMissingDataError(Exception):
    """
    Some number was out of range
    """

    def __init__(self, value):
        super(MBObsMissingDataError, self).__init__(value)
        self.value = value

    def __str__(self):
        return repr(self.value)


class MBObsExtractor(object):
    """
    class to extract observations from the images

    parameters
    ----------
    images: dict
        A dictionary of image objects

    """

    def __init__(self, config, images):
        self.config = config
        self.images = images
        self.log = logging.getLogger("lsst.meas.extensions.ngmix.MBObsExtractor")

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

        mbobs = ngmix.MultiBandObsList()

        for filt in self.config['bands_fit']:
            # TODO: run noise replacers here

            imf = self.images[filt]

            bbox = self._get_bbox(rec, imf)
            subim = _get_padded_sub_image(imf, bbox)

            obs = self._extract_obs(subim, rec)

            obslist = ngmix.ObsList()
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
            T = 4.0

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
        maskobj = imobj_sub.mask
        bmask = maskobj.array

        # cen = rec.getCentroid()
        orig_cen = imobj_sub.getWcs().skyToPixel(rec.getCoord())
        psf_im = self._extract_psf_image(imobj_sub, orig_cen)
        # psf_im = imobj_sub.getPsf().computeKernelImage(orig_cen).array

        # fake the psf pixel noise
        psf_err = psf_im.max()*0.0001
        psf_wt = psf_im*0 + 1.0/psf_err**2

        jacob = self._extract_jacobian(imobj_sub, rec)

        # use canonical center for the psf
        psf_cen = (np.array(psf_im.shape)-1.0)/2.0
        psf_jacob = jacob.copy()
        psf_jacob.set_cen(row=psf_cen[0], col=psf_cen[1])

        # we will have need of the bit names which we can only
        # get from the mask object
        # this is sort of monkey patching, but I'm not sure of
        # a better solution
        meta = {'maskobj': maskobj}

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
            meta=meta,
        )

        return obs

    def _extract_psf_image(self, stamp, orig_pos):
        """
        get the psf associated with this stamp

        coadded psfs are generally not square, so we will
        trim it to be square and preserve the center to
        be at the new canonical center
        """
        try:
            psfobj = stamp.getPsf()
            psfim = psfobj.computeKernelImage(orig_pos).array
        except InvalidParameterError:
            raise MBObsMissingDataError("could not reconstruct PSF")

        psfim = np.array(psfim, dtype='f4', copy=False)

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
        cen = orig_cen - geom.Extent2D(xy0)
        row = cen.getY()
        col = cen.getX()

        wcs = imobj.getWcs().linearizePixelToSky(
            orig_cen,
            geom.arcseconds,
        )
        jmatrix = wcs.getLinear().getMatrix()

        jacob = ngmix.Jacobian(
            row=row,
            col=col,
            dudrow=jmatrix[0, 0],
            dudcol=jmatrix[0, 1],
            dvdrow=jmatrix[1, 0],
            dvdcol=jmatrix[1, 1],
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
        var_image = imobj.variance.array
        maskobj = imobj.mask
        mask = maskobj.array

        weight = var_image.copy()

        weight[:, :] = 0

        bitnames_to_ignore = self.config['stamps']['bits_to_ignore_for_weight']

        bits_to_ignore = util.get_ored_bits(maskobj, bitnames_to_ignore)

        wuse = np.where(
            (var_image > 0)
            & ((mask & bits_to_ignore) == 0)
        )

        if wuse[0].size > 0:
            medvar = np.median(var_image[wuse])
            weight[:, :] = 1.0/medvar
        else:
            self.log.debug('    weight is all zero, found none that passed cuts')
            # _print_bits(maskobj, bitnames_to_ignore)

        bitnames_to_null = self.config['stamps']['bits_to_null']
        if len(bitnames_to_null) > 0:
            bits_to_null = util.get_ored_bits(maskobj, bitnames_to_null)
            wnull = np.where((mask & bits_to_null) != 0)
            if wnull[0].size > 0:
                self.log.debug('    nulling %d in weight' % wnull[0].size)
                weight[wnull] = 0.0

        return weight

    def _verify(self):
        """
        check for consistency between the images.

        .. todo::
           An assertion is currently used, we may want to raise an
           appropriate exception.
        """
        xy0 = None
        for filt in self.config['bands_fit']:
            if filt not in self.images:
                raise MBObsMissingDataError('band missing: %s' % filt)

            imf = self.images[filt]
            if xy0 is None:
                xy0 = imf.getXY0()
            else:
                assert xy0 == imf.getXY0(),\
                    "all images must have same reference position"


def _project_box(source, wcs, radius):
    """
    create a box for the input source
    """
    pixel = geom.Point2I(wcs.skyToPixel(source.getCoord()))
    box = geom.Box2I()
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
    bbox2 = geom.Box2I(bbox)
    bbox2.clip(region)
    if isinstance(original, afwImage.Exposure):
        result.setPsf(original.getPsf())
        result.setWcs(original.getWcs())
        result.setPhotoCalib(original.getPhotoCalib())
        # result.image.array[:, :] = float("nan")
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


def _print_bits(maskobj, bitnames):
    mask = maskobj.array
    for ibit, bitname in enumerate(bitnames):
        bitval = maskobj.getPlaneBitMask(bitname)
        w = np.where((mask & bitval) != 0)
        if w[0].size > 0:
            print('%s %d %d/%d' % (bitname, bitval, w[0].size, mask.size))

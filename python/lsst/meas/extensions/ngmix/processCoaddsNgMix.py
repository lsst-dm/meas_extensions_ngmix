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

from lsst.geom import Extent2D
from lsst.afw.table import SourceCatalog, SchemaMapper
from lsst.pex.config import Field, ListField
from lsst.pipe.base import Struct

from .processCoaddsTogether import ProcessCoaddsTogetherConfig, ProcessCoaddsTogetherTask


__all__ = ("ProcessCoaddsNgMixConfig", "ProcessCoaddsNgMixTask")


class ProcessCoaddsNgMixConfig(ProcessCoaddsTogetherConfig):
    filters = ListField(dtype=str, default=[], doc="List of expected bandpass filters.")
    # TODO: add config fields here, e.g.:
    maxIter = Field(dtype=int, doc="Maximum number of iterations", default=100, optional=False)

    def setDefaults(self):
        self.output.name = "deepCoadd_ngmix"


class ProcessCoaddsNgMixTask(ProcessCoaddsTogetherTask):
    _DefaultName = "processCoaddsNgMix"
    ConfigClass = ProcessCoaddsNgMixConfig

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

        return schema

    def run(self, images, ref):
        """Process coadds from all bands for a single patch.

        This method should not add or modify self.

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

        if set(images.keys()) != set(self.config.filters):
            raise RuntimeError("One or more filters missing.")

        # Make an empty catalog
        output = SourceCatalog(self.schema)

        # Add mostly-empty rows to it, copying IDs from the ref catalog.
        output.extend(ref, mapper=self.mapper)

        # TODO: set up noise replacers for using deblender outputs

        for n, (outRecord, refRecord) in enumerate(zip(output, ref)):
            outRecord.setFootprint(None)  # copied from ref; don't need to write these again

            data = {}
            psfs = {}
            xy0 = None
            for filt in self.config.filters:
                # TODO: run noise replacers here
                data[filt] = images[filt].image.array
                psfs[filt] = images[filt].getPsf().computeKernelImage().array
                if xy0 is None:
                    xy0 = images[filt].getXY0()
                else:
                    assert xy0 == images[filt].getXY0()

            localCentroid = refRecord.getCentroid() - Extent2D(xy0)

            # TODO: other useful things to maybe pass to fitObject:
            #  - refRecord.getShape(): Gaussian-weighted second moments
            #  - images[filt].getCalib(): photometric calibration
            #  - images[filt].getWcs(): WCS (probably want to linearize this first)
            #  - images[filt].getVariance(): translate this into weight image
            #  - images[filt].getMask(): bad pixel mask (probably also goes into weight image)

            fit = self.fitObject(row=localCentroid.getY(), col=localCentroid.getX(),
                                 data=data, psfs=psfs)
            for k, v in fit.items():
                outRecord[k] = v

        return Struct(output=output)

    def fitObject(self, row, col, data, psfs):
        """Fit a single object.

        Parameters
        ----------
        row: `float`
            Position of the object in the first dimension of the `data` arrays.
        col: `float`
            Position of the object in the second dimension of the `data` arrays.
        data: `dict` of `{str: numpy.ndarray}`
            Dictionary of image data arrays, keyed by filter name.
        psfs: `dict` of `{str: numpy.ndarray}`
            Dictionary of PSF images, keyed by filter name.

        Returns
        -------
        results : `dict`
            Dictionary of outputs, with keys matching the fields added in
            `defineSchema()`.
        """

        # TODO: real work goes here

        return {}

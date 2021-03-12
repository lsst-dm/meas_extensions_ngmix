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

# TODO: this module should be moved elsewhere (probably pipe_tasks)
# once it's stable.

from lsst.afw.table import Schema, SourceCatalog
from lsst.pipe.base import (
    CmdLineTask, ArgumentParser,
)
from lsst.meas.base import NoiseReplacerConfig
from lsst.pex.config import ConfigField, Field

from lsst.coadd.utils.coaddDataIdContainer import ExistingCoaddDataIdContainer
from lsst.pipe.tasks.multiBand import MergeSourcesRunner
from lsst.pipe.tasks.fit_multiband import CatalogExposure, MultibandFitSubConfig, MultibandFitSubTask

from typing import List

__all__ = ("ProcessCoaddsTogetherConfig", "ProcessCoaddsTogetherTask")


class ProcessCoaddsTogetherConfig(MultibandFitSubConfig):
    images = Field(
        doc="Coadd image DatasetType used as input (one for each band)",
        default="deepCoadd_calexp",
        dtype=str,
    )
    ref = Field(
        doc="Coadd catalog DatasetType reference input (one instance across all bands).",
        default="deepCoadd_ref",
        dtype=str,
    )
    output = Field(
        doc="Output catalog DatasetType (one instance across all bands)",
        default=None,   # Must be overridden by derived classes to a DatasetType known to obs_base
        dtype=str,
    )
    numSourcesLog = Field(doc='Number of sources to fit before logging status', default=100, dtype=int)
    deblendReplacer = ConfigField(
        dtype=NoiseReplacerConfig,
        doc=("Details for how to replace neighbors with noise when applying deblender outputs. "
             "Ignored if `useDeblends` == False."),
    )
    deblendReplacerUseMetadata = Field(
        doc="Whether to use catalog metadata to initialize NoiseReplacer config."
            " Overrides `deblendReplacer` if True.",
        dtype=bool,
        default=True,
    )
    deblendCatalog = Field(
        doc=("Catalog DatasetType from which to extract deblended [Heavy]Footprints (one for each band). "
             "Ignored if 'useDeblends` == False."),
        default="deepCoadd_meas",
        dtype=str,
    )
    useDeblends = Field(
        dtype=bool,
        doc="Whether to apply deblender outputs by replacing neighboring sources with noise.",
        optional=True,
        default=True,
    )


class ProcessCoaddsTogetherTask(CmdLineTask, MultibandFitSubTask):
    _DefaultName = "processCoaddsTogether"
    ConfigClass = ProcessCoaddsTogetherConfig

    # This feeds the runDataRef() method all bands at once, rather than each
    # one separately.
    # The name reflects how it's used elsewhere, not what it does
    RunnerClass = MergeSourcesRunner

    # TODO: override DatasetType introspection for PipelineTask.  Probably
    # blocked on DM-16275.

    @property
    def schema(self) -> Schema:
        return self._schema

    @classmethod
    def _makeArgumentParser(cls):
        # Customize argument parsing for CmdLineTask.
        parser = ArgumentParser(name=cls._DefaultName)
        # This should be config.images, but there's no way to pass that
        # information in here in Gen2.
        datasetType = "deepCoadd_calexp"
        parser.add_id_argument("--id", datasetType,
                               ContainerClass=ExistingCoaddDataIdContainer,
                               help="data ID, e.g. --id tract=12345 patch=1,2 filter=g^r^i")
        return parser

    def __init__(self, *, schema=None, config=None, butler=None, initInputs=None, **kwds):
        CmdLineTask.__init__(self, config=config, **kwds)
        MultibandFitSubTask.__init__(self, config=config, schema=schema, **kwds)
        if schema is None:
            if butler is None:
                if initInputs is not None:
                    schema = initInputs.get("refSchema", None)
                if schema is None:
                    schema = SourceCatalog.Table.makeMinimalSchema()
            else:
                schema = butler.get(self.config.ref + "_schema").schema
        self._schema = self.defineSchema(schema)

    def getInitOutputDatasets(self):
        # Customize init output dataset retrieval for PipelineTask.
        return {"outputSchema": SourceCatalog(self.schema)}

    def getSchemaCatalogs(self):
        # Customize schema dataset retrieval for CmdLineTask
        return {self.config.output: SourceCatalog(self.schema)}

    def _getConfigName(self):
        # Config writing with CmdLineTask is disabled for this class.
        return None

    def _getMetadataName(self):
        # Metadata writing with CmdLineTask is disabled for this class.
        return None

    def runDataRef(self, patchRefList):
        """Run this task via CmdLineTask and Gen2 Butler.

        Parameters
        ----------
        patchRefList : `list` of `lsst.daf.persistence.ButlerDataRef`
            A list of DataRefs for all filters in a single patch.
        """
        catexps = []
        mergedDataId = {"tract": patchRefList[0].dataId["tract"],
                        "patch": patchRefList[0].dataId["patch"]}
        butler = patchRefList[0].butlerSubset.butler
        ref = butler.get("deepCoadd_ref", dataId=mergedDataId)
        expId = butler.get("deepMergedCoaddId", dataId=mergedDataId)

        for patchRef in patchRefList:
            dataId = mergedDataId.copy()
            dataId['band'] = patchRef.dataId["filter"]
            image = patchRef.get(self.config.images)
            if self.config.useDeblends:
                fpCat = patchRef.get(self.config.deblendCatalog)
            else:
                fpCat = None
            catexps.append(
                CatalogExposure(dataId=dataId, exposure=image, catalog=fpCat, id_tract_patch=expId)
            )
        results = self.run(catexps, ref)
        butler.put(results.output, self.config.output, dataId=mergedDataId)

    def defineSchema(self, refSchema):
        """Return the Schema for the output catalog.

        This may add or modify self.

        Parameters
        ----------
        refSchema : `lsst.afw.table.Schema`
            Schema of the input reference catalog.

        Returns
        -------
        outputSchema : `lsst.afw.table.Schema`
            Schema of the output catalog.
        """
        raise NotImplementedError("Must be implemented by derived classes.")

    def run(self, catexps: List[CatalogExposure], cat_ref: SourceCatalog):
        """Process coadds from all bands for a single patch.

        This method should not add or modify self.

        Parameters
        ----------
        catexps : `List [CatalogExposure]`
            Coadd images and associated metadata, keyed by filter name.
        cat_ref : `lsst.afw.table.SourceCatalog`
            A catalog with one record for each object, containing "best"
            measurements across all bands.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            Struct with (at least) an `output` attribute that is a catalog
            to be written as ``self.config.output``.
        """
        raise NotImplementedError("Must be implemented by derived classes.")

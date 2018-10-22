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

from lsst.afw.table import SourceCatalog
from lsst.pipe.base import (
    CmdLineTask, ArgumentParser,
    PipelineTask, InputDatasetField, OutputDatasetField, QuantumConfig,
)
from lsst.pex.config import Config

from lsst.coadd.utils.coaddDataIdContainer import ExistingCoaddDataIdContainer
from lsst.pipe.tasks.multiBand import DeblendCoaddSourcesRunner, getShortFilterName


__all__ = ("ProcessCoaddsTogetherConfig", "ProcessCoaddsTogetherTask")


class ProcessCoaddsTogetherConfig(Config):
    images = InputDatasetField(
        doc="Coadd image DatasetType used as input (one for each band)",
        name="deepCoadd_calexp",
        scalar=False,
        storageClass="Exposure",
        units=("Tract", "Patch", "AbstractFilter", "SkyMap")
    )
    ref = InputDatasetField(
        doc="Coadd catalog DatasetType reference input (one instance across all bands)",
        name="deepCoadd_ref",
        scalar=True,
        storageClass="SourceCatalog",
        units=("Tract", "Patch", "SkyMap")
    )
    output = OutputDatasetField(
        doc="Output catalog DatasetType (one instance across all bands)",
        name=None,   # Must be overridden by derived classes to a DatasetType known to obs_base
        scalar=True,
        storageClass="SourceCatalog",
        units=("Tract", "Patch", "SkyMap")
    )
    quantum = QuantumConfig(
        units=("Tract", "Patch", "SkyMap")
    )


class ProcessCoaddsTogetherTask(CmdLineTask, PipelineTask):
    _DefaultName = "processCoaddsTogether"
    ConfigClass = ProcessCoaddsTogetherConfig

    # This feeds the runDataRef() method all bands at once, rather than each one separately.
    # The name reflects how it's used elsewhere, not what it does
    RunnerClass = DeblendCoaddSourcesRunner

    # TODO: override DatasetType introspection for PipelineTask.  Probably
    # blocked on DM-16275.

    @classmethod
    def _makeArgumentParser(cls):
        # Customize argument parsing for CmdLineTask.
        parser = ArgumentParser(name=cls._DefaultName)
        # This should be config.images.name, but there's no way to pass that
        # information in here in Gen2.
        datasetType = "deepCoadd_calexp"
        parser.add_id_argument("--id", datasetType,
                               ContainerClass=ExistingCoaddDataIdContainer,
                               help="data ID, e.g. --id tract=12345 patch=1,2 filter=g^r^i")
        return parser

    def __init__(self, *, config=None, refSchema=None, butler=None, initInputs=None, **kwds):
        super().__init__(config=config, **kwds)
        if refSchema is None:
            if butler is None:
                if initInputs is not None:
                    refSchema = initInputs.get("refSchema", None)
                if refSchema is None:
                    refSchema = SourceCatalog.Table.makeMinimalSchema()
            else:
                refSchema = butler.get(self.config.ref.name + "_schema")
        self.schema = self.defineSchema(refSchema)

    def getInitOutputDatasets(self):
        # Customize init output dataset retrieval for PipelineTask.
        return {"outputSchema": SourceCatalog(self.schema)}

    def getSchemaCatalogs(self):
        # Customize schema dataset retrieval for CmdLineTask
        return {self.config.output.name + "_schema": SourceCatalog(self.schema)}

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
        images = {}
        butler = None
        mergedDataId = {}
        for patchRef in patchRefList:
            filt = getShortFilterName(patchRef.dataId["filter"])
            images[filt] = patchRef.get(self.config.images.name)
            if butler is None:
                butler = patchRef.butlerSubset.butler
                mergedDataId = {"tract": patchRef.dataId["tract"], "patch": patchRef.dataId["patch"]}
        assert butler is not None
        ref = butler.get("deepCoadd_ref", dataId=mergedDataId)
        results = self.run(images, ref)
        butler.put(results.output, self.config.output.name, dataId=mergedDataId)

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
        raise NotImplementedError("Must be implemented by derived classes.")

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
import numpy

import lsst.pex.exceptions
import lsst.afw.detection
import lsst.afw.geom
import lsst.afw.math as afwMath

from lsst.meas.base.pluginRegistry import register
from lsst.meas.base.sfm import SingleFramePluginConfig, SingleFramePlugin

__all__ = (
    "SingleFrameNgmixConfig", "SingleFrameNgmixPlugin"
)

# --- Wrapped C++ Plugins ---


# --- Single-Frame Measurement Plugins ---
class SingleFrameNgmixConfig(SingleFramePluginConfig):
    numGaussians = lsst.pex.config.Field(dtype=int, default=1, optional=False,
                                  doc="Number of gaussians")

@register("meas_extensions_ngmix")
class SingleFrameNgmixPlugin(SingleFramePlugin):
    '''
    Algorithm to calculate the position of a centroid on the focal plane
    '''

    ConfigClass = SingleFrameNgmixConfig

    @classmethod
    def getExecutionOrder(cls):
        return cls.SHAPE_ORDER

    def __init__(self, config, name, schema, metadata):
        SingleFramePlugin.__init__(self, config, name, schema, metadata)
        self.failFlag = schema.addField(name + '_flag', type="Flag", doc="Set to 1 for any fatal failure")
        n = config.numGaussians
        self.keys = []
        for i in range(n):
            self.keys.append(schema.addField(name + '_Gaussian_%d_value'%i , type="D", doc="Gaussian value"))

    def measure(self, measRecord, exposure):
        for i in range(len(self.keys)):
            measRecord.set(self.keys[i], 100*i)

    def fail(self, measRecord, error=None):
        measRecord.set(failFlag, True)

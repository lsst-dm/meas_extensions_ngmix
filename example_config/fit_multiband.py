from lsst.meas.extensions.ngmix.processCoaddsNGMix import ProcessCoaddsNGMixMaxTask

config.connections.name_output_cat = "ngmix_deblended"
config.fit_multiband.retarget(ProcessCoaddsNGMixMaxTask)

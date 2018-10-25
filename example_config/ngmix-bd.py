# optionally limit the number of objects we process
# the range is inclusive, not like a slice
config.obj_range = [1000, 1009]

#
# postage stamp config
#

config.stamps.min_stamp_size = 32
config.stamps.max_stamp_size = 256
# how many "sigma" to make the radius of the box
config.stamps.sigma_factor = 5.0

###############
# object config

config.obj.model="bd"

#
# prior on center
#

config.obj.priors.cen.type="gauss2d"

# this is the width in both directions
config.obj.priors.cen.pars=[0.2]

#
# prior on the ellipticity g
#

config.obj.priors.g.type="ba"

# this is the width
config.obj.priors.g.pars=[0.3]


#
# prior on the size squared T
#

config.obj.priors.T.type="two-sided-erf"

# this is the width
config.obj.priors.T.pars=[-10.0, 0.03, 1.0e+06, 1.0e+05]

#
# prior on fracdev, only used if model is "bd"
#

config.obj.priors.fracdev.type="gauss"

# this is the center and sigma of the gaussian
config.obj.priors.fracdev.pars=[0.5, 0.1]


#
# prior on the flux
#

config.obj.priors.flux.type="two-sided-erf"

# this is the width
config.obj.priors.flux.pars=[-1.0e+04, 1.0, 1.0e+09, 0.25e+08]

# fitting parameters
config.obj.max_pars.ntry=2
config.obj.max_pars.lm_pars.maxfev=2000
config.obj.max_pars.lm_pars.xtol=5.0e-5
config.obj.max_pars.lm_pars.ftol=5.0e-5


#############
# psf config

# 3 cocentric, coelliptical gaussians
config.psf.model="coellip3"
config.psf.fwhm_guess=0.8

# fitting parameters
config.psf.max_pars.ntry=4
config.psf.max_pars.lm_pars.maxfev=2000
config.psf.max_pars.lm_pars.xtol=5.0e-5
config.psf.max_pars.lm_pars.ftol=5.0e-5


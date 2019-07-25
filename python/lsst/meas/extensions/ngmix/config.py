from lsst.pex.config import Field, ListField, ConfigField, Config, ChoiceField
from .processCoaddsTogether import ProcessCoaddsTogetherConfig


class MetacalConfig(Config):
    """
    configuration of metacalibration

    we can add more options later
    """
    types = ListField(
        dtype=str,
        default=['noshear', '1p', '1m', '2p', '2m'],
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

    bits_to_cut = ListField(
        dtype=str,
        default=[],
        doc='do not process objects that have these bits set',
    )

    max_zero_weight_frac = Field(
        dtype=float,
        default=0.45,
        doc='max allowed fraction of stamp with zero weight',
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
            "gauss2d": "2d gaussian",
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
            "ba": "See Bernstein & Armstrong",
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
            "two-sided-erf": "two-sided error function, smoother than a flat prior",
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
            "two-sided-erf": "two-sided error function, smoother than a flat prior",
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
            "gauss": "gaussian prior on fracdev",
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
        # optional=True,
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
            "gauss": "gaussian model",
            "coellip2": "coelliptical 2 gauss model",
            "coellip3": "coelliptical 3 gauss model",
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
            "gauss": "gaussian model",
            "exp": "exponential model",
            "dev": "dev model",
            # bd and bdf are the same
            "bd": "bulge+disk model with fixed size ratio",
            "bdf": "bulge+disk model with fixed size ratio",
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
    plot_prefix = Field(
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
    obj = ConfigField(dtype=ObjectMaxFitConfig, doc="object fitting config")

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
    obj = ConfigField(dtype=ObjectMaxFitConfig, doc='object fitting config')
    metacal = ConfigField(dtype=MetacalConfig, doc='metacal config')

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

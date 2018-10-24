# flags used by NGMixer
BAD_OBJ              = 2**25
IMAGE_FLAGS          = 2**26
NO_CUTOUTS           = 2**27
BOX_SIZE_TOO_BIG     = 2**28
UTTER_FAILURE        = 2**29
NO_ATTEMPT           = 2**30

# flags for fitting codes
PSF_FIT_FAILURE      = 2**0
GAL_FIT_FAILURE      = 2**1
PSF_FLUX_FIT_FAILURE = 2**2
LOW_PSF_FLUX         = 2**3

# when doing forced photometry, there were flags in the models file for
# this object

FORCEPHOT_BAD_MODEL = 2**4
FORCEPHOT_FAILURE = 2**5

object_blacklist=[3126629751,3126910598]
OBJECT_IN_BLACKLIST = 2**24

# an overall flag for metacal fitting
# this will be set if any flags in
# the mcal_flags field are set
METACAL_FAILURE = 2**4



# flags for fitting codes
PSF_FIT_FAILURE      = 2**0
PSF_FLUX_FIT_FAILURE = 2**1
GAL_FIT_FAILURE      = 2**2

# an overall flag for metacal fitting
# this will be set if any flags in
# the mcal_flags field are set
METACAL_FAILURE = 2**4

# flags used by NGMixer
#BAD_OBJ              = 2**25
#IMAGE_FLAGS          = 2**26
#NO_CUTOUTS           = 2**27
#BOX_SIZE_TOO_BIG     = 2**28
#UTTER_FAILURE        = 2**29
NO_ATTEMPT           = 2**30




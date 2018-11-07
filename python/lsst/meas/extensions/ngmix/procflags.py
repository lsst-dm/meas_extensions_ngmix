# no attempt was made for some measurment.
NO_ATTEMPT           = 2**0

# the weight map had too many zeros in it
HIGH_MASKFRAC        = 2**1

# some bits were set in the image that are not allowed
IMAGE_FLAGS          = 2**2

# PSF fitting failed
PSF_FIT_FAILURE      = 2**3

# the PSF flux fitting failed
PSF_FLUX_FIT_FAILURE = 2**4

# the object fitting failed
#OBJ_FIT_FAILURE      = 2**5


# failure of PSF fitting at some point in metacal
METACAL_PSF_FAILURE = 2**0

# failure of PSF flux fitting at some point in metacal
METACAL_PSF_FLUX_FAILURE = 2**1

# failure of object fitting at some point in metacal
METACAL_OBJ_FAILURE = 2**2

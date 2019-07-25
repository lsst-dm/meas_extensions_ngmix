import ngmix


def get_joint_prior(objconf, nband, rng):
    """
    get a joint prior on parameters used in the object maximum
    likelihood fitting
    """
    conf = objconf['priors']

    cen_prior = _get_cen_prior(conf['cen'], rng)
    g_prior = _get_g_prior(conf['g'], rng)
    T_prior = _get_generic_prior(conf['T'], rng)
    flux_prior = _get_generic_prior(conf['flux'], rng)

    if nband > 1:
        flux_prior = [flux_prior]*nband

    if objconf['model'] == 'bd':
        fracdev_prior = _get_generic_prior(
            conf['fracdev'],
            rng,
            bounds=[0.0, 1.0],
        )
        jprior = ngmix.joint_prior.PriorBDFSep(
            cen_prior,
            g_prior,
            T_prior,
            fracdev_prior,
            flux_prior,
        )
    else:
        jprior = ngmix.joint_prior.PriorSimpleSep(
            cen_prior,
            g_prior,
            T_prior,
            flux_prior,
        )

    return jprior


def _get_cen_prior(conf, rng):
    if conf['type'] == 'gauss2d':
        assert len(conf['pars']) == 1, "gauss2d cen prior requires 1 parameters"
        width = conf['pars'][0]
        prior = ngmix.priors.CenPrior(0.0, 0.0, width, width, rng=rng)
    else:
        raise ValueError("bad cen prior: '%s'" % conf['type'])

    return prior


def _get_g_prior(conf, rng):
    if conf['type'] == 'ba':
        assert len(conf['pars']) == 1, "BA g prior requires 1 parameters"
        g_prior = ngmix.priors.GPriorBA(*conf['pars'], rng=rng)
    else:
        raise ValueError("bad g prior '%s'" % conf['type'])

    return g_prior


def _get_generic_prior(conf, rng, bounds=None):

    if conf['type'] == 'gauss':
        assert len(conf['pars']) == 2, "gauss requires 2 parameters"
        prior = ngmix.priors.Normal(*conf['pars'], bounds=bounds, rng=rng)

    elif conf['type'] == 'two-sided-erf':
        assert len(conf['pars']) == 4, "two-sided-erf requires 4 parameters"
        prior = ngmix.priors.TwoSidedErf(*conf['pars'])

    else:
        raise ValueError("bad generic prior: '%s'" % conf['type'])

    return prior

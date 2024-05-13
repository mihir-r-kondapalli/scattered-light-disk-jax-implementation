import jax
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt
from functools import partial
from scipy.optimize import newton
from scipy.interpolate import interp1d

class Jax_class:

    param_names = {}

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def unpack_pars(cls, p_arr):
        """
        This function takes a parameter array (params) and unpacks it into a
        dictionary with the parameter names as keys.
        """
        p_dict = {}
        keys = list(cls.param_names.keys())
        i = 0
        for i in range(0, len(p_arr)):
            p_dict[keys[i]] = p_arr[i]

        return p_dict

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def pack_pars(cls, p_dict):
        """
        This function takes a parameter dictionary and packs it into a JAX array
        where the order is set by the parameter name list defined on the class.
        """    
        p_arrs = []
        for name in cls.param_names.keys():
            p_arrs.append(p_dict[name])
        return jnp.asarray(p_arrs)


class DustEllipticalDistribution2PowerLaws(Jax_class):
    """
    """

    param_names = {'ain': 5., 'aout': -5., 'a': 60., 'e': 0., 'ksi0': 1.,'gamma': 2., 'beta': 1.,
                        'amin': 0., 'dens_at_r0': 1., 'accuracy': 5.e-3, 'zmax': 0., "p": 0., "rmax": 0.,
                        'pmin': 0., "apeak": 0., "apeak_surface_density": 0., "itiltthreshold": 0.}

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def init(cls, accuracy=5.e-3, ain=5., aout=-5., a=60., e=0., ksi0=1., gamma=2., beta=1., amin=0., dens_at_r0=1.):
        """
        Constructor for the Dust_distribution class.

        We assume the dust density is 0 radially after it drops below 0.5%
        (the accuracy variable) of the peak density in
        the midplane, and vertically whenever it drops below 0.5% of the
        peak density in the midplane
        """

        p_dict = {}
        p_dict["accuracy"] = accuracy

        p_dict["ksi0"] = ksi0
        p_dict["gamma"] = gamma
        p_dict["beta"] = beta
        p_dict["zmax"] = ksi0*(-jnp.log(p_dict["accuracy"]))**(1./gamma)

        # Set Vertical Density Analogue
        gamma = jnp.where(gamma < 0., 0.1, gamma)
        ksi0 = jnp.where(ksi0 < 0., 0.1, ksi0)
        beta = jnp.where(beta < 0., 0., beta)

        # Set Radial Density Analogue
        ain = jnp.where(ain < 0.1, 0.1, ain)
        aout = jnp.where(aout > -0.1, -0.1, aout)
        e = jnp.where(e < 0, 0., e)
        e = jnp.where(e >= 1, 0.99, e)
        amin = jnp.where(amin < 0., 0., amin)
        dens_at_r0 = jnp.where(dens_at_r0 < 0, 1, dens_at_r0)

        p_dict["ain"] = ain
        p_dict["aout"] = aout
        p_dict["a"] = a
        p_dict["e"] = e
        p_dict["p"] = p_dict["a"]*(1-p_dict["e"]**2)
        p_dict["amin"] = amin
        # we assume the inner hole is also elliptic (convention)
        p_dict["pmin"] = p_dict["amin"]*(1-p_dict["e"]**2)
        p_dict["dens_at_r0"] = dens_at_r0

        # maximum distance of integration, AU
        p_dict["rmax"] = p_dict["a"]*p_dict["accuracy"]**(1/p_dict["aout"])
        p_dict["apeak"] = p_dict["a"] * jnp.power(-p_dict["ain"]/p_dict["aout"],
                                        1./(2.*(p_dict["ain"]-p_dict["aout"])))
        Gamma_in = p_dict["ain"]+p_dict["beta"]
        Gamma_out = p_dict["aout"]+p_dict["beta"]
        p_dict["apeak_surface_density"] = p_dict["a"] * jnp.power(-Gamma_in/Gamma_out,
                                                        1./(2.*(Gamma_in-Gamma_out)))
        # the above formula comes from Augereau et al. 1999.
        p_dict["itiltthreshold"] = jnp.rad2deg(jnp.arctan(p_dict["rmax"]/p_dict["zmax"]))

        return cls.pack_pars(p_dict)
    
    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def density_cylindrical(cls, distr_params, r, costheta, z):
        """ Returns the particule volume density at r, theta, z
        """
        distr = cls.unpack_pars(distr_params)
        radial_ratio = r/(distr["p"]/(1-distr["e"]*costheta))
        den = (jnp.power(radial_ratio, -2*distr["ain"]) +
               jnp.power(radial_ratio, -2*distr["aout"]))
        radial_density_term = jnp.sqrt(2./den)*distr["dens_at_r0"]
        
        #if distr["pmin"] > 0:
        #    radial_density_term[r/(distr["pmin"]/(1-distr["e"]*costheta)) <= 1] = 0
        radial_density_term = jnp.where(distr["pmin"] > 0, 
                                        jnp.where(r/(distr["pmin"]/(1-distr["e"]*costheta)) <= 1, 0, radial_density_term),
                                        radial_density_term)

        den2 = (distr["ksi0"]*jnp.power(radial_ratio, distr["beta"]))
        vertical_density_term = jnp.exp(-jnp.power(jnp.abs(z)/den2, distr["gamma"]))
        return radial_density_term*vertical_density_term

class HenyeyGreenstein_SPF(Jax_class):
    """
    Implementation of a scattering phase function with a single Henyey
    Greenstein function.
    """

    param_names = {'g': 0.}

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def init(cls, func_params):
        """
        Constructor of a Heyney Greenstein phase function.

        Parameters
        ----------
        spf_dico :  dictionnary containing the key "g" (float)
            g is the Heyney Greenstein coefficient and should be between -1
            (backward scattering) and 1 (forward scattering).
        """

        p_dict = {}
        g = func_params[0]
        g = jnp.where(g>=1, 0.99, g)
        g = jnp.where(g<=-1, -0.99, g)
        p_dict["g"] = g

        return cls.pack_pars(p_dict)
    
    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def compute_phase_function_from_cosphi(cls, phase_func_params, cos_phi):
        """
        Compute the phase function at (a) specific scattering scattering
        angle(s) phi. The argument is not phi but cos(phi) for optimization
        reasons.

        Parameters
        ----------
        cos_phi : float or array
            cosine of the scattering angle(s) at which the scattering function
            must be calculated.
        """
        p_dict = cls.unpack_pars(phase_func_params)
        
        return 1./(4*jnp.pi)*(1-p_dict["g"]**2) / \
            (1+p_dict["g"]**2-2*p_dict["g"]*cos_phi)**(3./2.)


class DoubleHenyeyGreenstein_SPF(Jax_class):
    """
    Implementation of a scattering phase function with a double Henyey
    Greenstein function.
    """

    param_names = {'g1': 0.5, 'g2': -0.3, 'weight': 0.7}

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def init(cls, func_params):
        """
        """

        p_dict = {}
        p_dict['g1'] = func_params[0]
        p_dict['g2'] = func_params[1]
        p_dict['weight'] = func_params[2]

        return cls.pack_pars(p_dict)
    
    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def compute_phase_function_from_cosphi(cls, phase_func_params, cos_phi):
        """
        Compute the phase function at (a) specific scattering scattering
        angle(s) phi. The argument is not phi but cos(phi) for optimization
        reasons.

        Parameters
        ----------
        cos_phi : float or array
            cosine of the scattering angle(s) at which the scattering function
            must be calculated.
        """

        p_dict = cls.unpack_pars(phase_func_params)

        hg1 = p_dict['weight'] * 1./(4*jnp.pi)*(1-p_dict["g1"]**2) / \
            (1+p_dict["g1"]**2-2*p_dict["g1"]*cos_phi)**(3./2.)
        hg2 = (1-p_dict['weight']) * 1./(4*jnp.pi)*(1-p_dict["g2"]**2) / \
            (1+p_dict["g2"]**2-2*p_dict["g2"]*cos_phi)**(3./2.)
        
        return hg1+hg2
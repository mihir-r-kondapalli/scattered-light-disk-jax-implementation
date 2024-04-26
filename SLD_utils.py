import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
from functools import partial
from scipy.optimize import newton
from scipy.interpolate import interp1d


def frame_center(array, verbose=False):
    """

    Taken from vip_hci.var.coords.py

    Return the coordinates y,x of the frame(s) center.
    If odd: dim/2-0.5
    If even: dim/2

    Parameters
    ----------
    array : 2d/3d/4d numpy ndarray
        Frame or cube.
    verbose : bool optional
        If True the center coordinates are printed out.

    Returns
    -------
    cy, cx : int
        Coordinates of the center.

    """
    if array.ndim == 2:
        shape = array.shape
    elif array.ndim == 3:
        shape = array[0].shape
    elif array.ndim == 4:
        shape = array[0, 0].shape
    else:
        raise ValueError('`array` is not a 2d, 3d or 4d array')

    cy = shape[0] / 2
    cx = shape[1] / 2

    if shape[0] % 2:
        cy -= 0.5
    if shape[1] % 2:
        cx -= 0.5

    if verbose:
        print('Center px coordinates at x,y = ({}, {})'.format(cx, cy))

    return int(cy), int(cx)

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

class Dust_distribution(Jax_class):
    """This class represents the dust distribution
    """

    param_names = {'ain': 5., 'aout': -5.,
                        'a': 60., 'e': 0., 'ksi0': 1., 'gamma': 2.,
                        'beta': 1., 'amin': 0., 'dens_at_r0': 1.,
                        'accuracy': 5.e-3}

    def __init__(self, density_dico={'name': '2PowerLaws', 'ain': 5, 'aout': -5,
                                     'a': 60, 'e': 0, 'ksi0': 1., 'gamma': 2.,
                                     'beta': 1., 'amin': 0., 'dens_at_r0': 1.}):
        """
        Constructor for the Dust_distribution class.

        We assume the dust density is 0 radially after it drops below 0.5%
        (the accuracy variable) of the peak density in
        the midplane, and vertically whenever it drops below 0.5% of the
        peak density in the midplane
        """
        self.p_dict = {}
        self.accuracy = 5.e-3
        if not isinstance(density_dico, dict):
            errmsg = 'The parameters describing the dust density distribution' \
                     ' must be a Python dictionnary'
            raise TypeError(errmsg)
        if 'name' not in density_dico.keys():
            errmsg = 'The dictionnary describing the dust density ' \
                     'distribution must contain the key "name"'
            raise TypeError(errmsg)
        self.type = density_dico['name']
        if self.type == '2PowerLaws':
            self.dust_distribution_calc = DustEllipticalDistribution2PowerLaws(
                                                    self.accuracy, density_dico)
        else:
            errmsg = 'The only dust distribution implemented so far is the' \
                     ' "2PowerLaws"'
            raise TypeError(errmsg)

    def set_density_distribution(self, density_dico):
        """
        Update the parameters of the density distribution.
        """
        self.dust_distribution_calc.set_density_distribution(density_dico)

    def density_cylindrical(self, r, costheta, z):
        """
        Return the particule volume density at r, theta, z.
        """
        return self.dust_distribution_calc.density_cylindrical(r, costheta, z)

    def density_cartesian(self, x, y, z):
        """
        Return the particule volume density at x,y,z, taking into account the
        offset of the disk.
        """
        return self.dust_distribution_calc.density_cartesian(x, y, z)

    def print_info(self, pxInAu=None):
        """
        Utility function that displays the parameters of the radial distribution
        of the dust

        Input:
            - pxInAu (optional): the pixel size in au
        """
        print('----------------------------')
        print('Dust distribution parameters')
        print('----------------------------')
        self.dust_distribution_calc.print_info(pxInAu)


class DustEllipticalDistribution2PowerLaws(Jax_class):
    """
    """

    param_names = {'ain': 5., 'aout': -5., 'a': 60., 'e': 0., 'ksi0': 1.,'gamma': 2., 'beta': 1.,
                        'amin': 0., 'dens_at_r0': 1., 'accuracy': 5.e-3, 'zmax': 0., "p": 0., "rmax": 0.,
                        'pmin': 0., "apeak": 0., "apeak_surface_density": 0., "itiltthreshold": 0.}

    def __init__(self, accuracy=5.e-3, density_dico={'ain': 5, 'aout': -5,
                                                     'a': 60, 'e': 0, 'ksi0': 1.,
                                                     'gamma': 2., 'beta': 1.,
                                                     'amin': 0., 'dens_at_r0': 1.}):
        """
        Constructor for the Dust_distribution class.

        We assume the dust density is 0 radially after it drops below 0.5%
        (the accuracy variable) of the peak density in
        the midplane, and vertically whenever it drops below 0.5% of the
        peak density in the midplane
        """
        self.p_dict = {}
        self.p_dict["accuracy"] = accuracy
        self.set_density_distribution(density_dico)

    def set_density_distribution(self, density_dico):
        """
        """
        if 'ksi0' not in density_dico.keys():
            ksi0 = 1.
        else:
            ksi0 = density_dico['ksi0']
        if 'beta' not in density_dico.keys():
            beta = 1.
        else:
            beta = density_dico['beta']
        if 'gamma' not in density_dico.keys():
            gamma = 1.
        else:
            gamma = density_dico['gamma']
        if 'aout' not in density_dico.keys():
            aout = -5.
        else:
            aout = density_dico['aout']
        if 'ain' not in density_dico.keys():
            ain = 5.
        else:
            ain = density_dico['ain']
        if 'e' not in density_dico.keys():
            e = 0.
        else:
            e = density_dico['e']
        if 'a' not in density_dico.keys():
            a = 60.
        else:
            a = density_dico['a']
        if 'amin' not in density_dico.keys():
            amin = 0.
        else:
            amin = density_dico['amin']
        if 'dens_at_r0' not in density_dico.keys():
            dens_at_r0 = 1.
        else:
            dens_at_r0 = density_dico['dens_at_r0']
        self.set_vertical_density(ksi0=ksi0, gamma=gamma, beta=beta)
        self.set_radial_density(
            ain=ain,
            aout=aout,
            a=a,
            e=e,
            amin=amin,
            dens_at_r0=dens_at_r0)

    def set_vertical_density(self, ksi0=1., gamma=2., beta=1.):
        """
        Sets the parameters of the vertical density function

        Parameters
        ----------
        ksi0 : float
            scale height in au at the reference radius (default 1 a.u.)
        gamma : float
            exponent (2=gaussian,1=exponential profile, default 2)
        beta : float
            flaring index (0=no flaring, 1=linear flaring, default 1)
        """
        if gamma < 0.:
            print('Warning the vertical exponent gamma is negative')
            print('Gamma was changed from {0:6.2f} to 0.1'.format(gamma))
            gamma = 0.1
        if ksi0 < 0.:
            print('Warning the scale height ksi0 is negative')
            print('ksi0 was changed from {0:6.2f} to 0.1'.format(ksi0))
            ksi0 = 0.1
        if beta < 0.:
            print('Warning the flaring coefficient beta is negative')
            print(
                'beta was changed from {0:6.2f} to 0 (flat disk)'.format(beta))
            beta = 0.
        self.p_dict["ksi0"] = float(ksi0)
        self.p_dict["gamma"] = float(gamma)
        self.p_dict["beta"] = float(beta)
        self.p_dict["zmax"] = ksi0*(-np.log(self.p_dict["accuracy"]))**(1./gamma)

    def set_radial_density(self, ain=5., aout=-5., a=60.,
                           e=0., amin=0., dens_at_r0=1.):
        """
        Sets the parameters of the radial density function

        Parameters
        ----------
        ain : float
            slope of the power-low distribution in the inner disk. It
            must be positive (default 5)
        aout : float
            slope of the power-low distribution in the outer disk. It
            must be negative (default -5)
        a : float
            reference radius in au (default 60)
        e : float
            eccentricity (default 0)
        amin: float
            minimim semi-major axis: the dust density is 0 below this
            value (default 0)
        """
        if ain < 0.1:
            print('Warning the inner slope is greater than 0.1')
            print('ain was changed from {0:6.2f} to 0.1'.format(ain))
            ain = 0.1
        if aout > -0.1:
            print('Warning the outer slope is greater than -0.1')
            print('aout was changed from {0:6.2f} to -0.1'.format(aout))
            aout = -0.1
        if e < 0:
            print('Warning the eccentricity is negative')
            print('e was changed from {0:6.2f} to 0'.format(e))
            e = 0.
        if e >= 1:
            print('Warning the eccentricity is greater or equal to 1')
            print('e was changed from {0:6.2f} to 0.99'.format(e))
            e = 0.99
        if a < 0:
            raise ValueError('Warning the semi-major axis a is negative')
        if amin < 0:
            raise ValueError('Warning the minimum radius a is negative')
            print('amin was changed from {0:6.2f} to 0.'.format(amin))
            amin = 0.
        if dens_at_r0 < 0:
            raise ValueError(
                'Warning the reference dust density at r0 is negative')
            print('It was changed from {0:6.2f} to 1.'.format(dens_at_r0))
            dens_at_r0 = 1.
        self.p_dict["ain"] = float(ain)
        self.p_dict["aout"] = float(aout)
        self.p_dict["a"] = float(a)
        self.p_dict["e"] = float(e)
        self.p_dict["p"] = self.p_dict["a"]*(1-self.p_dict["e"]**2)
        self.p_dict["amin"] = float(amin)
        # we assume the inner hole is also elliptic (convention)
        self.p_dict["pmin"] = self.p_dict["amin"]*(1-self.p_dict["e"]**2)
        self.p_dict["dens_at_r0"] = float(dens_at_r0)
        try:
            # maximum distance of integration, AU
            self.p_dict["rmax"] = self.p_dict["a"]*self.p_dict["accuracy"]**(1/self.p_dict["aout"])
            if self.p_dict["ain"] != self.p_dict["aout"]:
                self.p_dict["apeak"] = self.p_dict["a"] * jnp.power(-self.p_dict["ain"]/self.p_dict["aout"],
                                               1./(2.*(self.p_dict["ain"]-self.p_dict["aout"])))
                Gamma_in = self.p_dict["ain"]+self.p_dict["beta"]
                Gamma_out = self.p_dict["aout"]+self.p_dict["beta"]
                self.p_dict["apeak_surface_density"] = self.p_dict["a"] * jnp.power(-Gamma_in/Gamma_out,
                                                               1./(2.*(Gamma_in-Gamma_out)))
                # the above formula comes from Augereau et al. 1999.
            else:
                self.p_dict["apeak"] = self.p_dict["a"]
                self.p_dict["apeak_surface_density"] = self.p_dict["a"]
        except OverflowError:
            print('The error occured during the calculation of rmax or apeak')
            print('Inner slope: {0:.6e}'.format(self.ain))
            print('Outer slope: {0:.6e}'.format(self.aout))
            print('Accuracy: {0:.6e}'.format(self.accuracy))
            raise OverflowError
        except ZeroDivisionError:
            print('The error occured during the calculation of rmax or apeak')
            print('Inner slope: {0:.6e}'.format(self.ain))
            print('Outer slope: {0:.6e}'.format(self.aout))
            print('Accuracy: {0:.6e}'.format(self.accuracy))
            raise ZeroDivisionError
        self.p_dict["itiltthreshold"] = jnp.rad2deg(np.arctan(self.p_dict["rmax"]/self.p_dict["zmax"]))

    def print_info(self, pxInAu=None):
        """
        Utility function that displays the parameters of the radial distribution
        of the dust

        Input:
            - pxInAu (optional): the pixel size in au
        """
        def rad_density(r):
            return jnp.sqrt(2/(jnp.power(r/self.p_dict["a"], -2*self.p_dict["ain"]) +
                              jnp.power(r/self.p_dict["a"], -2*self.p_dict["aout"])))

        def half_max_density(r): return rad_density(r) / \
            rad_density(self.p_dict["apeak"])-1./2.
        try:
            if self.p_dict["aout"] < -3:
                a_plus_hwhm = newton(half_max_density, self.p_dict["apeak"]*1.04)
            else:
                a_plus_hwhm = newton(half_max_density, self.p_dict["apeak"]*1.1)
        except RuntimeError:
            a_plus_hwhm = np.nan
        try:
            if self.ain < 2:
                a_minus_hwhm = newton(half_max_density, self.p_dict["apeak"]*0.5)
            else:
                a_minus_hwhm = newton(half_max_density, self.p_dict["apeak"]*0.95)
        except RuntimeError:
            a_minus_hwhm = np.nan
        if pxInAu is not None:
            msg = 'Reference semi-major axis: {0:.1f}au or {1:.1f}px'
            print(msg.format(self.p_dict["a"], self.p_dict["a"]/pxInAu))
            msg2 = 'Semi-major axis at maximum dust density in plane z=0: {0:.1f}au or ' \
                   '{1:.1f}px (same as ref sma if ain=-aout)'
            print(msg2.format(self.p_dict["apeak"], self.p_dict["apeak"]/pxInAu))
            msg3 = 'Semi-major axis at half max dust density in plane z=0: {0:.1f}au or ' \
                '{1:.1f}px for the inner edge ' \
                '/ {2:.1f}au or {3:.1f}px for the outer edge, with a FWHM of ' \
                '{4:.1f}au or {5:.1f}px'
            print(msg3.format(a_minus_hwhm, a_minus_hwhm/pxInAu, a_plus_hwhm,
                              a_plus_hwhm/pxInAu, a_plus_hwhm-a_minus_hwhm,
                              (a_plus_hwhm-a_minus_hwhm)/pxInAu))
            msg4 = 'Semi-major axis at maximum dust surface density: {0:.1f}au or ' \
                   '{1:.1f}px (same as ref sma if ain=-aout)'
            print(
                msg4.format(
                    self.p_dict["apeak_surface_density"],
                    self.p_dict["apeak_surface_density"] /
                    pxInAu))
            msg5 = 'Ellipse p parameter: {0:.1f}au or {1:.1f}px'
            print(msg5.format(self.p_dict["p"], self.p_dict["p"]/pxInAu))
        else:
            print('Reference semi-major axis: {0:.1f}au'.format(self.p_dict["a"]))
            msg = 'Semi-major axis at maximum dust density in plane z=0: {0:.1f}au (same ' \
                  'as ref sma if ain=-aout)'
            print(msg.format(self.p_dict["apeak"]))
            msg3 = 'Semi-major axis at half max dust density: {0:.1f}au ' \
                '/ {1:.1f}au for the inner/outer edge, or a FWHM of ' \
                '{2:.1f}au'
            print(
                msg3.format(
                    a_minus_hwhm,
                    a_plus_hwhm,
                    a_plus_hwhm -
                    a_minus_hwhm))
            msg4 = 'Semi-major axis at maximum dust surface density: {0:.1f}au ' \
                   '(same as ref sma if ain=-aout)'
            print(
                msg4.format(
                    self.p_dict["apeak_surface_density"]))
            print('Ellipse p parameter: {0:.1f}au'.format(self.p_dict["p"]))
        print('Ellipticity: {0:.3f}'.format(self.p_dict["e"]))
        print('Inner slope: {0:.2f}'.format(self.p_dict["ain"]))
        print('Outer slope: {0:.2f}'.format(self.p_dict["aout"]))
        print(
            'Density at the reference semi-major axis: {0:4.3e} (arbitrary unit'.format(self.p_dict["dens_at_r0"]))
        if self.p_dict["amin"] > 0:
            print('Minimum radius (sma): {0:.2f}au'.format(self.p_dict["amin"]))
        if pxInAu is not None:
            msg = 'Scale height: {0:.1f}au or {1:.1f}px at {2:.1f}'
            print(msg.format(self.p_dict["ksi0"], self.p_dict["ksi0"]/pxInAu, self.p_dict["a"]))
        else:
            print('Scale height: {0:.2f} au at {1:.2f}'.format(self.p_dict["ksi0"],
                                                               self.p_dict["a"]))
        print('Vertical profile index: {0:.2f}'.format(self.p_dict["gamma"]))
        msg = 'Disc vertical FWHM: {0:.2f} at {1:.2f}'
        print(msg.format(2.*self.p_dict["ksi0"]*np.power(np.log10(2.), 1./self.p_dict["gamma"]),
                         self.p_dict["a"]))
        print('Flaring coefficient: {0:.2f}'.format(self.p_dict["beta"]))
        print('------------------------------------')
        print('Properties for numerical integration')
        print('------------------------------------')
        print('Requested accuracy {0:.2e}'.format(self.p_dict["accuracy"]))
#        print('Minimum radius for integration: {0:.2f} au'.format(self.rmin))
        print('Maximum radius for integration: {0:.2f} au'.format(self.p_dict["rmax"]))
        print('Maximum height for integration: {0:.2f} au'.format(self.p_dict["zmax"]))
        msg = 'Inclination threshold: {0:.2f} degrees'
        print(msg.format(self.p_dict["itiltthreshold"]))
        return

    def density_cylindrical(self, r, costheta, z):
        """ Returns the particule volume density at r, theta, z
        """
        radial_ratio = r/(self.p_dict["p"]/(1-self.p_dict["e"]*costheta))
        den = (np.power(radial_ratio, -2*self.p_dict["ain"]) +
               np.power(radial_ratio, -2*self.p_dict["aout"]))
        radial_density_term = np.sqrt(2./den)*self.p_dict["dens_at_r0"]
        if self.p_dict["pmin"] > 0:
            radial_density_term[r/(self.p_dict["pmin"]/(1-self.p_dict["e"]*costheta)) <= 1] = 0
        den2 = (self.p_dict["ksi0"]*np.power(radial_ratio, self.p_dict["beta"]))
        vertical_density_term = np.exp(-np.power(np.abs(z)/den2, self.p_dict["gamma"]))
        return radial_density_term*vertical_density_term
    
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

    def density_cartesian(self, x, y, z):
        """ Returns the particule volume density at x,y,z, taking into account
        the offset of the disk
        """
        r = np.sqrt(x**2+y**2)
        if r == 0:
            costheta = 0
        else:
            costheta = x/r
        return self.density_cylindrical(r, costheta, z)


class Phase_function(object):
    """ This class represents the scattering phase function (spf).
    It contains the attribute phase_function_calc that implements either a
    single Henyey Greenstein phase function, a double Heyney Greenstein,
    or any custom function (data interpolated from
    an input list of phi, spf(phi)).
    """

    def __init__(self, spf_dico={'name': 'HG', 'g': 0., 'polar': False}):
        """
        Constructor of the Phase_function class. It checks whether the spf_dico
        contains a correct name and sets the attribute phase_function_calc

        Parameters
        ----------
        spf_dico :  dictionnary
            Parameters describing the scattering phase function to be
            implemented. By default, an isotropic phase function is implemented.
            It should at least contain the key "name" chosen between 'HG'
            (single Henyey Greenstein), 'DoubleHG' (double Heyney Greenstein) or
            'interpolated' (custom function).
            The parameter "polar" enables to switch on the polarisation (if set
            to True, the default is False). In this case it assumes either
                - a Rayleigh polarised fraction (1-(cos phi)^2) / (1+(cos phi)^2).
                  if nothing else is specified
                - a polynomial if the keyword 'polar_polynom_coeff' is defined
                  and corresponds to an array of polynomial coefficient, e.g.
                  [p3,p2,p1,p0] evaluated as np.polyval([p3,p2,p1,p0],np.arange(0, 180, 1))
        """
        if not isinstance(spf_dico, dict):
            msg = 'The parameters describing the phase function must be a ' \
                  'Python dictionnary'
            raise TypeError(msg)
        if 'name' not in spf_dico.keys():
            msg = 'The dictionnary describing the phase function must contain' \
                  ' the key "name"'
            raise TypeError(msg)
        self.type = spf_dico['name']
        if 'polar' not in spf_dico.keys():
            self.polar = False
        else:
            if not isinstance(spf_dico['polar'], bool):
                msg = 'The dictionnary describing the polarisation must be a ' \
                      'boolean'
                raise TypeError(msg)
            self.polar = spf_dico['polar']
            if 'polar_polynom_coeff' in spf_dico.keys():
                self.polar_polynom = True
                if isinstance(spf_dico['polar_polynom_coeff'],
                              (tuple, list, np.ndarray)):
                    self.polar_polynom_coeff = spf_dico['polar_polynom_coeff']
                else:
                    msg = 'The dictionnary describing the polarisation polynomial function must be an ' \
                          'array'
                    raise TypeError(msg)
            else:
                self.polar_polynom = False
        if self.type == 'HG':
            self.phase_function_calc = HenyeyGreenstein_SPF(spf_dico)
        elif self.type == 'DoubleHG':
            self.phase_function_calc = DoubleHenyeyGreenstein_SPF(spf_dico)
        #elif self.type == 'interpolated':
        #    self.phase_function_calc = Interpolated_SPF(spf_dico)
        else:
            msg = 'Type of phase function not understood: {0:s}'
            raise TypeError(msg.format(self.type))

    def compute_phase_function_from_cosphi(self, cos_phi):
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
        phf = self.phase_function_calc.compute_phase_function_from_cosphi(
            cos_phi)
        if self.polar:
            if self.polar_polynom:
                phi = np.rad2deg(np.arccos(cos_phi))
                return np.polyval(self.polar_polynom_coeff, phi) * phf
            else:
                return (1-cos_phi**2)/(1+cos_phi**2) * phf
        else:
            return phf

    def print_info(self):
        """
        Prints information on the type and parameters of the scattering phase
        function.
        """
        print('----------------------------')
        print('Phase function parameters')
        print('----------------------------')
        print('Type of phase function: {0:s}'.format(self.type))
        print('Linear polarisation: {0!r}'.format(self.polar))
        self.phase_function_calc.print_info()

    def plot_phase_function(self):
        """
        Plots the scattering phase function
        """
        phi = np.arange(0, 180, 1)
        phase_func = self.compute_phase_function_from_cosphi(
            np.cos(np.deg2rad(phi)))
        if self.polar:
            if self.polar_polynom:
                phase_func = np.polyval(
                    self.polar_polynom_coeff, phi) * phase_func
            else:
                phase_func = (1-np.cos(np.deg2rad(phi))**2) / \
                             (1+np.cos(np.deg2rad(phi))**2) * phase_func

        plt.close(0)
        plt.figure(0)
        plt.plot(phi, phase_func)
        plt.xlabel('Scattering phase angle in degrees')
        plt.ylabel('Scattering phase function')
        plt.grid()
        plt.xlim(0, 180)
        plt.show()


class HenyeyGreenstein_SPF(Jax_class):
    """
    Implementation of a scattering phase function with a single Henyey
    Greenstein function.
    """

    param_names = {'g': 0.}

    def __init__(self, spf_dico={'g': 0.}):
        """
        Constructor of a Heyney Greenstein phase function.

        Parameters
        ----------
        spf_dico :  dictionnary containing the key "g" (float)
            g is the Heyney Greenstein coefficient and should be between -1
            (backward scattering) and 1 (forward scattering).
        """

        self.p_dict = {}

        # it must contain the key "g"
        if 'g' not in spf_dico.keys():
            raise TypeError('The dictionnary describing a Heyney Greenstein '
                            'phase function must contain the key "g"')
        # the value of "g" must be a float or a list of floats
        elif not isinstance(spf_dico['g'], (float, int)):
            raise TypeError('The key "g" of a Heyney Greenstein phase function'
                            ' dictionnary must be a float or an integer')
        self.set_phase_function(spf_dico['g'])

    def set_phase_function(self, g):
        """
        Set the value of g
        """
        if g >= 1:
            print('Warning the Henyey Greenstein parameter is greater than or '
                  'equal to 1')
            print('The value was changed from {0:6.2f} to 0.99'.format(g))
            g = 0.99
        elif g <= -1:
            print('Warning the Henyey Greenstein parameter is smaller than or '
                  'equal to -1')
            print('The value was changed from {0:6.2f} to -0.99'.format(g))
            g = -0.99
        self.p_dict["g"] = float(g)

    def compute_phase_function_from_cosphi(self, cos_phi):
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
        return 1./(4*np.pi)*(1-self.p_dict["g"]**2) / \
            (1+self.p_dict["g"]**2-2*self.p_dict["g"]*cos_phi)**(3./2.)
    
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

    def print_info(self):
        """
        Prints the value of the HG coefficient
        """
        print('Heynyey Greenstein coefficient: {0:.2f}'.format(self.g))


class DoubleHenyeyGreenstein_SPF(Jax_class):
    """
    Implementation of a scattering phase function with a double Henyey
    Greenstein function.
    """

    def __init__(self, spf_dico={'g': [0.5, -0.3], 'weight': 0.7}):
        """
        """
        # it must contain the key "g"
        if 'g' not in spf_dico.keys():
            raise TypeError('The dictionnary describing a Heyney Greenstein'
                            ' phase function must contain the key "g"')
        # the value of "g" must be a list of floats
        elif not isinstance(spf_dico['g'], (list, tuple, np.ndarray)):
            raise TypeError('The key "g" of a Heyney Greenstein phase '
                            'function dictionnary must be  a list of floats')
        # it must contain the key "weight"
        if 'weight' not in spf_dico.keys():
            raise TypeError('The dictionnary describing a multiple Henyey '
                            'Greenstein phase function must contain the '
                            'key "weight"')
        # the value of "weight" must be a list of floats
        elif not isinstance(spf_dico['weight'], (float, int)):
            raise TypeError('The key "weight" of a Heyney Greenstein phase '
                            'function dictionnary must be a float (weight of '
                            'the first HG coefficient between 0 and 1)')
        elif spf_dico['weight'] < 0 or spf_dico['weight'] > 1:
            raise ValueError('The key "weight" of a Heyney Greenstein phase'
                             ' function dictionnary must be between 0 and 1. It'
                             ' corresponds to the weight of the first HG '
                             'coefficient')
        if len(spf_dico['g']) != 2:
            raise TypeError('The keys "weight" and "g" must contain the same'
                            ' number of elements')
        self.g = spf_dico['g']
        self.weight = spf_dico['weight']

    def print_info(self):
        """
        Prints the value of the HG coefficients and weights
        """
        print('Heynyey Greenstein first component : coeff {0:.2f} , '
              'weight {1:.1f}%'.format(self.g[0], self.weight*100))
        print('Heynyey Greenstein second component: coeff {0:.2f} , '
              'weight {1:.1f}%'.format(self.g[1], (1-self.weight)*100.))

    def compute_singleHG_from_cosphi(self, g, cos_phi):
        """
        Compute a single Heyney Greenstein phase function at (a) specific
        scattering scattering angle(s) phi. The argument is not phi but cos(phi)
        for optimization reasons.

        Parameters
        ----------
        g : float
            Heyney Greenstein coefficient
        cos_phi : float or array
            cosine of the scattering angle(s) at which the scattering function
            must be calculated.
        """
        return 1./(4*np.pi)*(1-g**2)/(1+g**2-2*g*cos_phi)**(3./2.)

    def compute_phase_function_from_cosphi(self, cos_phi):
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
        return self.weight * self.compute_singleHG_from_cosphi(self.g[0],
                                                               cos_phi) + \
            (1-self.weight) * self.compute_singleHG_from_cosphi(self.g[1],
                                                                cos_phi)
    
    def compute_phase_function_from_cosphi(self, cos_phi):
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
        return self.weight * self.compute_singleHG_from_cosphi(self.g[0],
                                                               cos_phi) + \
            (1-self.weight) * self.compute_singleHG_from_cosphi(self.g[1],
                                                                cos_phi)
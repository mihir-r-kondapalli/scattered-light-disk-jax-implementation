#! /usr/bin/env python
"""
Class definition for ScatteredLightDisk, Dust_distribution and Phase_function

.. [AUG99]
   | Augereau et al. 1999
   | **On the HR 4796 A circumstellar disk**
   | *Astronomy & Astrophysics, Volume 348, pp. 557-569*
   | `https://arxiv.org/abs/astro-ph/9906429
     <https://arxiv.org/abs/astro-ph/9906429>`_

"""

__author__ = 'Julien Milli'
__all__ = ['ScatteredLightDisk',
           'Dust_distribution',
           'Phase_function']

import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from scipy.optimize import newton
from scipy.interpolate import interp1d
from functools import partial
from SLD_utils import *


class ScatteredLightDisk(Jax_class):
    """
    Class used to generate a synthetic disc, inspired from a light version of
    the GRATER tool (GRenoble RAdiative TransfER) written originally in IDL
    [AUG99]_, and converted to Python by J. Milli.
    """

    # Jax Parameters
    param_names = {
        "distance": 0.,
        "itilt": 0.,
        "omega": 0.,
        "pxInArcsec": 0.,
        "pxInAU": 0.,
        "pa": 0.,
        "flux_max": None,
        "xdo": 0., "ydo": 0.,
        "rmin": 0.,
        "xc": 0, "yc": 0,
        "cospa": 0., "sinpa": 0.,
        "cosi": 0., "sini": 0.
    }

    def __init__(self, nx=200, ny=200, distance=50., itilt=60., omega=0.,
                 pxInArcsec=0.01225, pa=0., flux_max=None,
                 density_dico={'name': '2PowerLaws', 'ain': 5, 'aout': -5,
                               'a': 40, 'e': 0, 'ksi0': 1., 'gamma': 2.,
                               'beta': 1., 'dens_at_r0': 1.},
                 spf_dico={'name': 'HG', 'g': 0., 'polar': False}, xdo=0.,
                 ydo=0.,cent=[160,160]):
        """
        Constructor of the Scattered_light_disk object, taking in input the
        geometric parameters of the disk, the radial density distribution
        and the scattering phase function.
        So far, only one radial distribution is implemented: a smoothed
        2-power-law distribution, but more complex radial profiles can be
        implemented on demand.
        The star is assumed to be centered at the frame center as defined in
        the vip_hci.var.frame_center function (geometric center of the image,
        e.g. either in the middle of the central pixel for odd-size images or
        in between 4 pixel for even-size images).

        Parameters
        ----------
        nx : int
            number of pixels along the x axis of the image (default 200)
        ny : int
            number of pixels along the y axis of the image (default 200)
        distance : float
            distance to the star in pc (default 70.)
        itilt : float
            inclination wrt the line of sight in degrees (0 means pole-on,
            90 means edge-on, default 60 degrees)
        omega : float
            argument of the pericenter in degrees (0 by default)
        pxInArcsec : float
            pixel field of view in arcsec/px (default the SPHERE pixel
            scale 0.01225 arcsec/px)
        pa : float
            position angle of the disc in degrees (default 0 degrees, e.g. North)
        flux_max : float
            the max flux of the disk in ADU. By default None, meaning that
            the disk flux is not normalized to any value.
        density_dico : dict
            Parameters describing the dust density distribution function
            to be implemented. By default, it uses a two-power law dust
            distribution with a vertical gaussian distribution with
            linear flaring. This dictionary should at least contain the key
            "name".
            For a to-power law distribution, you can set it with
            'name:'2PowerLaws' and with the following parameters:
                a : float
                    reference radius in au (default 40)
                ksi0 : float
                    scale height in au at the reference radius (default 1 a.u.)
                gamma : float
                    exponent (2=gaussian,1=exponential profile, default 2)
                beta : float
                    flaring index (0=no flaring, 1=linear flaring, default 1)
                ain : float
                    slope of the power-low distribution in the inner disk. It
                    must be positive (default 5)
                aout : float
                    slope of the power-low distribution in the outer disk. It
                    must be negative (default -5)
                e : float
                    eccentricity (default 0)
                amin: float
                    minimim semi-major axis: the dust density is 0 below this
                    value (default 0)
        spf_dico :  dictionnary
            Parameters describing the scattering phase function to be implemented.
            By default, an isotropic phase function is implemented. It should
            at least contain the key "name".
        xdo : float
            disk offset along the x-axis in the disk frame (=semi-major axis),
            in a.u. (default 0)
        ydo : float
            disk offset along the y-axis in the disk frame (=semi-minor axis),
            in a.u. (default 0)
        """

        self.p_dict = {}

        self.p_dict["nx"] = nx    # number of pixels along the x axis of the image
        self.p_dict["ny"] = ny    # number of pixels along the y axis of the image
        self.p_dict["distance"] = distance  # distance to the star in pc
        self.set_inclination(itilt)
        self.set_omega(omega)
        self.set_flux_max(flux_max)
        self.p_dict["pxInArcsec"] = pxInArcsec  # pixel field of view in arcsec/px
        self.p_dict["pxInAU"] = self.p_dict["pxInArcsec"]*self.p_dict["distance"]     # 1 pixel in AU
        # disk offset along the x-axis in the disk frame (semi-major axis), AU
        self.p_dict["xdo"] = xdo
        # disk offset along the y-axis in the disk frame (semi-minor axis), AU
        self.p_dict["ydo"] = ydo
        self.p_dict["rmin"] = np.sqrt(self.p_dict["xdo"]**2+self.p_dict["ydo"]**2)+self.p_dict["pxInAU"]
        self.dust_density = Dust_distribution(density_dico)
        # star center along the y- and x-axis, in pixels
        self.p_dict["yc"], self.p_dict["xc"] = frame_center(np.ndarray((self.p_dict["ny"], self.p_dict["nx"])))
        
        self.set_pa(pa)
        self.phase_function = Phase_function(spf_dico=spf_dico)
        self.scattered_light_map = jnp.zeros((self.p_dict["ny"], self.p_dict["nx"]))
        self.image = jnp.zeros((self.p_dict["ny"], self.p_dict["nx"]))

    def set_inclination(self, itilt):
        """
        Sets the inclination of the disk.

        Parameters
        ----------
        itilt : float
            inclination of the disk wrt the line of sight in degrees (0 means
            pole-on, 90 means edge-on, default 60 degrees)
        """
        self.p_dict["itilt"] = float(itilt)  # inclination wrt the line of sight in deg
        self.p_dict["cosi"] = jnp.cos(np.deg2rad(self.p_dict["itilt"]))
        self.p_dict["sini"] = jnp.sin(np.deg2rad(self.p_dict["itilt"]))

    def set_pa(self, pa):
        """
        Sets the disk position angle

        Parameters
        ----------
        pa : float
            position angle in degrees
        """
        self.p_dict["pa"] = pa    # position angle of the disc in degrees
        self.p_dict["cospa"] = jnp.cos(np.deg2rad(pa))
        self.p_dict["sinpa"] = jnp.sin(np.deg2rad(pa))

        self.x_vector = (jnp.arange(0, self.p_dict["nx"]) - self.p_dict["xc"])*self.p_dict["pxInAU"]
        self.y_vector = (jnp.arange(0, self.p_dict["ny"]) - self.p_dict["yc"])*self.p_dict["pxInAU"]

    def set_omega(self, omega):
        """
        Sets the argument of pericenter

        Parameters
        ----------
        omega : float
            angle in degrees
        """
        self.p_dict["omega"] = float(omega)

    def set_flux_max(self, flux_max):
        """
        Sets the mas flux of the disk

        Parameters
        ----------
        flux_max : float
            the max flux of the disk in ADU
        """
        self.p_dict["flux_max"] = flux_max

    def set_density_distribution(self, density_dico):
        """
        Sets or updates the parameters of the density distribution

        Parameters
        ----------
        density_dico : dict
            Parameters describing the dust density distribution function
            to be implemented. By default, it uses a two-power law dust
            distribution with a vertical gaussian distribution with
            linear flaring. This dictionary should at least contain the key
            "name". For a to-power law distribution, you can set it with
            name:'2PowerLaws' and with the following parameters:

                - a : float
                    Reference radius in au (default 60)
                - ksi0 : float
                    Scale height in au at the reference radius (default 1 a.u.)
                - gamma : float
                    Exponent (2=gaussian,1=exponential profile, default 2)
                - beta : float
                    Flaring index (0=no flaring, 1=linear flaring, default 1)
                - ain : float
                    Slope of the power-low distribution in the inner disk. It
                    must be positive (default 5)
                - aout : float
                    Slope of the power-low distribution in the outer disk. It
                    must be negative (default -5)
                - e : float
                    Eccentricity (default 0)
        """
        self.dust_density.set_density_distribution(density_dico)

    def set_phase_function(self, spf_dico):
        """
        Sets the phase function of the dust

        Parameters
        ----------
        spf_dico :  dict
            Parameters describing the scattering phase function to be
            implemented. Three phase functions are implemented so far: single
            Heyney Greenstein, double Heyney Greenstein and custum phase
            functions through interpolation. Read the constructor of each of
            those classes to know which parameters must be set in the dictionary
            in each case.
        """
        self.phase_function = Phase_function(spf_dico=spf_dico)

    def print_info(self):
        """
        Prints the information of the disk and image parameters
        """
        '''print('-----------------------------------')
        print('Geometrical properties of the image')
        print('-----------------------------------')
        print('Image size: {0:d} px by {1:d} px'.format(self.nx, self.ny))
        msg1 = 'Pixel size: {0:.4f} arcsec/px or {1:.2f} au/px'
        print(msg1.format(self.pxInArcsec, self.pxInAU))
        msg2 = 'Distance of the star {0:.1f} pc'
        print(msg2.format(self.distance))
        msg3 = 'From {0:.1f} au to {1:.1f} au in X'
        print(msg3.format(self.x_vector[0], self.x_vector[self.nx-1]))
        msg4 = 'From {0:.1f} au to {1:.1f} au in Y'
        print(msg4.format(self.y_vector[0], self.y_vector[self.nx-1]))
        print('Position angle of the disc: {0:.2f} degrees'.format(self.pa))
        print('Inclination {0:.2f} degrees'.format(self.itilt))
        print('Argument of pericenter {0:.2f} degrees'.format(self.omega))
        if self.flux_max is not None:
            print('Maximum flux of the disk {0:.2f}'.format(self.flux_max))
        self.dust_density.print_info()
        self.phase_function.print_info()'''

    def check_inclination(self):
        """
        Checks whether the inclination set is close to edge-on and risks to
        induce artefacts from the limited numerical accuracy. In such a case
        the inclination is changed to be less edge-on.
        """
        if np.abs(np.mod(self.p_dict["itilt"], 180)-90) < np.abs(
                np.mod(self.dust_density.dust_distribution_calc.p_dict["itiltthreshold"], 180)-90):
            print('Warning the disk is too close to edge-on')
            msg = 'The inclination was changed from {0:.2f} to {1:.2f}'
            print(msg.format(self.p_dict["itilt"],
                             self.dust_density.dust_distribution_calc.p_dict["itiltthreshold"]))
            self.set_inclination(
                self.dust_density.dust_distribution_calc.p_dict["itiltthreshold"])

    @classmethod
    @partial(jax.jit, static_argnums=(0, 3, 5))
    def compute_scattered_light_jax(cls, disk_params, distr_params, distr_cls, phase_func_params,
                                    phase_func_cls, x_vector, y_vector, scattered_light_map,
                                    limage, image, tmp, nx, ny, halfNbSlices=25):
        """
        Computes the scattered light image of the disk.

        Parameters
        ----------
        halfNbSlices : integer
            half number of distances along the line of sight l
        """

        disk = cls.unpack_pars(disk_params)
        distr = distr_cls.unpack_pars(distr_params)

        #x_vector = (jnp.arange(0, disk["nx"]) - disk["xc"])*disk["pxInAU"]  # x axis in au
        #y_vector = (jnp.arange(0, disk["ny"]) - disk["yc"])*disk["pxInAU"]  # y axis in au
        
        x_map_0PA, y_map_0PA = jnp.meshgrid(x_vector, y_vector)
        # rotation to get the disk major axis properly oriented, x in AU
        y_map = (disk["cospa"]*x_map_0PA + disk["sinpa"]*y_map_0PA)
        # rotation to get the disk major axis properly oriented, y in AU
        x_map = (-disk["sinpa"]*x_map_0PA + disk["cospa"]*y_map_0PA)

        # dist along the line of sight to reach the disk midplane (z_D=0), AU:
        lz0_map = y_map * jnp.tan(jnp.deg2rad(disk["itilt"]))
        # dist to reach +zmax, AU:
        lzp_map = distr["zmax"]/disk["cosi"] + \
            lz0_map
        # dist to reach -zmax, AU:
        lzm_map = -distr["zmax"]/disk["cosi"] + \
            lz0_map
        dl_map = jnp.absolute(lzp_map-lzm_map)  # l range, in AU
        # squared maximum l value to reach the outer disk radius, in AU^2:
        lmax2 = distr["rmax"]**2 - \
            (x_map**2+y_map**2)
        # squared minimum l value to reach the inner disk radius, in AU^2:
        lmin2 = (x_map**2+y_map**2)-disk["rmin"]**2
        validPixel_map = (lmax2 > 0.) * (lmin2 > 0.)
        lwidth = 100.  # control the distribution of distances along l
        nbSlices = 2*halfNbSlices-1  # total number of distances
        # along the line of sight

        tmp = (jnp.exp(jnp.arange(halfNbSlices)*jnp.log(lwidth+1.) /
                      (halfNbSlices-1.))-1.)/lwidth  # between 0 and 1
        
        ll = jnp.concatenate((-tmp[:0:-1], tmp))

        # 1d array pre-calculated values, AU
        ycs_vector = jnp.where(validPixel_map, disk["cosi"]*y_map, 0)
        # 1d array pre-calculated values, AU
        zsn_vector = jnp.where(validPixel_map, -disk["sini"]*y_map, 0)
        xd_vector = jnp.where(validPixel_map, x_map, 0)  # x_disk, in AU

        #limage = jnp.ndarray((nbSlices, disk["ny"], disk["nx"]))
        #limage = jnp.zeros([nbSlices, ny, nx])
        #image = jnp.zeros((ny, nx))

        for il in range(nbSlices):
            # distance along the line of sight to reach the plane z

            l_vector = jnp.where(validPixel_map, lz0_map + ll[il]*dl_map, 0)
            #l_vector = lz0_map + ll[il]*dl_map

            # rotation about x axis
            yd_vector = ycs_vector + disk["sini"] * l_vector  # y_Disk in AU
            zd_vector = zsn_vector + disk["cosi"] * l_vector  # z_Disk, in AU
            # Dist and polar angles in the frame centered on the star position:
            # squared distance to the star, in AU^2
            d2star_vector = xd_vector**2+yd_vector**2+zd_vector**2
            dstar_vector = jnp.sqrt(d2star_vector)  # distance to the star, in AU
            # midplane distance to the star (r coordinate), in AU
            rstar_vector = jnp.sqrt(xd_vector**2+yd_vector**2)
            thetastar_vector = jnp.arctan2(yd_vector, xd_vector)
            # Phase angles:
            cosphi_vector = (rstar_vector*disk["sini"]*jnp.sin(thetastar_vector) +
                             zd_vector*disk["cosi"])/dstar_vector  # in radians
            # Polar coordinates in the disk frame, and semi-major axis:
            # midplane distance to the disk center (r coordinate), in AU
            r_vector = jnp.sqrt((xd_vector-disk["xdo"])**2+(yd_vector-disk["ydo"])**2)
            # polar angle in radians between 0 and pi
            theta_vector = jnp.arctan2(yd_vector-disk["ydo"], xd_vector-disk["xdo"])

            costheta_vector = jnp.cos(theta_vector-jnp.deg2rad(disk["omega"]))
            # Scattered light:
            # volume density
            rho_vector = distr_cls.density_cylindrical(distr_params, r_vector,
                                                               costheta_vector,
                                                               zd_vector)
            
            phase_function = phase_func_cls.compute_phase_function_from_cosphi(phase_func_params, cosphi_vector)
            #image = np.ndarray((disk["ny"], disk["nx"]))
            image = jnp.where(validPixel_map, rho_vector*phase_function/d2star_vector, 0)
            #limage[il, :, :] = image
            limage = limage.at[il,:,:].set(image)

        for il in range(1, nbSlices):
            scattered_light_map += (ll[il]-ll[il-1]) * (limage[il-1, :, :] +
                                                             limage[il, :, :])
        scattered_light_map = jnp.where(validPixel_map, scattered_light_map * dl_map / 2. * disk["pxInAU"]**2, 0)
        #ideally should check for valid pixel map
        if disk["flux_max"] is not None:
            scattered_light_map = scattered_light_map * (disk["flux_max"] /
                                         jnp.nanmax(scattered_light_map))
        return scattered_light_map
    
    def compute_scattered_light(self, halfNbSlices=25):
        self.check_inclination()
        self.limage = jnp.zeros([2*halfNbSlices-1, self.p_dict["ny"], self.p_dict["nx"]])
        self.tmp = jnp.arange(0, halfNbSlices)
        return ScatteredLightDisk.compute_scattered_light_jax(
            self.pack_pars(self.p_dict),
            self.dust_density.dust_cls.pack_pars(self.dust_density.dust_distribution_calc.p_dict),
            self.dust_density.dust_cls, self.phase_function.func_cls.pack_pars(self.phase_function.phase_function_calc.p_dict),
            self.phase_function.func_cls, self.x_vector, self.y_vector, self.scattered_light_map, self.limage, self.image,
            self.tmp, self.p_dict["nx"], self.p_dict["ny"])
from scattered_light_disk import ScatteredLightDisk as SLD
import matplotlib.pyplot as plt
import numpy as np

global disk_sld

# Not written by me

def single_hg2_disk_model_original(inc,pa,alpha_in,alpha_out,a, cent,gs_ws,flux_scaling,
                    ksi0=3.,gamma=2.,beta=1.,dstar=111.61,
                    nx=140,ny=140,pixel_scale=0.063,n_nodes=6):
    '''
    Make a single disk model! 

    Note the scattering phase function input to ScatteredLightDisk is overwritten
    '''

    #The ScatteredLightDisk object
    disk_sld = SLD(nx=nx, ny=ny, distance=dstar,
                                itilt=inc, omega=0, pxInArcsec=pixel_scale, pa=pa,
                                density_dico={'name':'2PowerLaws','ain':alpha_in,'aout':alpha_out,
                                              'a':a,'e':0.0,'ksi0':ksi0,'gamma':gamma,'beta':beta},
                                spf_dico={'name':'HG', 'g':0., 'polar':False},
                                flux_max=1, cent=cent)
    
    #The scattering phase function object
    density = disk_sld.dust_density.dust_distribution_calc

    #Generate the disk image

    disk_sld.set_phase_function({'name': 'DoubleHG',
                                              'g': [0.6, -0.6], 'weight': 0.7,
                                              'polar': True})
    disk_image = disk_sld.compute_scattered_light()
    
    return disk_sld, flux_scaling*disk_image


def single_hg_disk_model_original(inc,pa,alpha_in,alpha_out,a, cent,gs_ws,flux_scaling,
                    ksi0=3.,gamma=2.,beta=1.,dstar=111.61,
                    nx=140,ny=140,pixel_scale=0.063,n_nodes=6):
    '''
    Make a single disk model! 

    Note the scattering phase function input to ScatteredLightDisk is overwritten
    '''

    #The ScatteredLightDisk object
    disk_sld = SLD(nx=nx, ny=ny, distance=dstar,
                                itilt=inc, omega=0, pxInArcsec=pixel_scale, pa=pa,
                                density_dico={'name':'2PowerLaws','ain':alpha_in,'aout':alpha_out,
                                              'a':a,'e':0.0,'ksi0':ksi0,'gamma':gamma,'beta':beta},
                                spf_dico={'name':'HG', 'g':0., 'polar':False},
                                flux_max=1, cent=cent)

    disk_image = disk_sld.compute_scattered_light()
    
    return disk_sld, flux_scaling*disk_image
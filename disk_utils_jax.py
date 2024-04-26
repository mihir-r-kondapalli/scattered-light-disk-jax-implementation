from jax import jit
from SLD_ojax import ScatteredLightDisk
from SLD_utils import Dust_distribution

def jax_disk_model(inc,pa,alpha_in,alpha_out,a, cent,gs_ws,flux_scaling,
                    ksi0=3.,gamma=2.,beta=1.,dstar=111.61,
                    nx=140,ny=140,pixel_scale=0.063,n_nodes=6):
    '''
    Make a single disk model! 

    Note the scattering phase function input to ScatteredLightDisk is overwritten
    '''

    #The ScatteredLightDisk object
    disk = ScatteredLightDisk(nx=nx, ny=ny, distance=dstar,
                                itilt=inc, omega=0, pxInArcsec=pixel_scale, pa=pa,
                                density_dico={'name':'2PowerLaws','ain':alpha_in,'aout':alpha_out,
                                              'a':a,'e':0.0,'ksi0':ksi0,'gamma':gamma,'beta':beta},
                                spf_dico={'name':'HG', 'g':0., 'polar':False},
                                flux_max=1)

    #The scattering phase function object

    # Index 13 is dust_density
    #density = disk_params[13].dust_distribution_calc

    #Generate the disk image
    disk_image = disk.compute_scattered_light()
    
    return flux_scaling*disk_image
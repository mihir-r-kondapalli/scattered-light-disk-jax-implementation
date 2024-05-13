import jax
import jax.numpy as jnp
from SLD_ojax import ScatteredLightDisk
from SLD_utils import DustEllipticalDistribution2PowerLaws, DoubleHenyeyGreenstein_SPF
from functools import partial

@partial(jax.jit, static_argnums=(0,1))
def jax_model(DistrModel, FuncModel, disk_params={'inclination': 0., 'position_angle': 0., "alpha_in": 0., "alpha_out": 0.,
                                                  'sma': 0., 'flux_scaling': 0.},
                  gs_ws={'g1': 0.}, halfNbSlices=25, ksi0=3., gamma=2., beta=1., dstar=111.61,
                  nx=140, ny=140, pixel_scale=0.063, n_nodes=6, pxInArcsec=0.01225, distance=50.):

    distr_params = DistrModel.init(accuracy=5.e-3, ain=disk_params['alpha_in'], aout=disk_params['alpha_out'], a=disk_params['sma'],
                                   e=0., ksi0=ksi0, gamma=gamma, beta=beta, amin=0., dens_at_r0=1.)
    disk_params_jax = ScatteredLightDisk.init(distr_params, disk_params['inclination'], disk_params['position_angle'],
                                              disk_params['alpha_in'], disk_params['alpha_out'], disk_params['sma'],
                                              nx=nx, ny=ny, distance = distance, itilt = 60., omega =0., pxInArcsec=pxInArcsec,
                                              flux_max=None, xdo=0., ydo=0.)

    yc, xc = ny, nx
    xc = jnp.where(nx%2==1, nx/2-0.5, nx/2).astype(int)
    yc = jnp.where(ny%2==1, ny/2-0.5, ny/2).astype(int)

    x_vector = (jnp.arange(0, nx) - xc)*pxInArcsec*distance
    y_vector = (jnp.arange(0, ny) - yc)*pxInArcsec*distance

    scattered_light_map = jnp.zeros((ny, nx))
    image = jnp.zeros((ny, nx))

    limage = jnp.zeros([2*halfNbSlices-1, ny, nx])
    tmp = jnp.arange(0, halfNbSlices)
    
    func_params = FuncModel.pack_pars(gs_ws)
    
    scattered_light_image = ScatteredLightDisk.compute_scattered_light_jax(disk_params_jax, distr_params, DistrModel, func_params, FuncModel,
                                                                  x_vector, y_vector, scattered_light_map, image, limage, tmp,
                                                                  halfNbSlices)
    
    return disk_params['flux_scaling']*scattered_light_image
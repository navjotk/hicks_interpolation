import click
import numpy as np
from math import floor
from scipy.signal.windows import kaiser
import matplotlib.pyplot as plt
from devito import PrecomputedSparseTimeFunction, TimeFunction, solve, Eq, Operator
from examples.seismic import Receiver, TimeAxis, RickerSource
from overthrust import overthrust_model_iso


@click.command()
@click.option("--r", type=int, default=16, help="The radius of the source scatter")
@click.option("--srcpd", type=int, default=785, help="Number of sources per dimension (total number will be square of this number)")
def run(r=16, srcpd=785):
    initial_model_filename = "overthrust_3D_initial_model.h5"
    tn = 5
    so = 6
    dtype = np.float32
    datakey = "m0"
    nbl = 40
    dt = 1.75
    
    time_axis = TimeAxis(start=0, stop=tn, step=dt)
    nt = time_axis.num
    model = overthrust_model_iso(initial_model_filename, datakey, so, nbl,
                                 dtype)
    shape = model.shape
    grid = model.grid
    origin = (0, 0, 0)
    spacing = model.spacing

    coeffs = kaiser(M=r, beta=6.31)

    # What we accepted as a parameter was sources per dimension. We are going to lay them out on a grid so square the number
    nsrc = srcpd * srcpd 
    src_coordinates = np.empty((nsrc, len(spacing)))
    offset = grid.spacing[0] * r/2
    onedpoints = np.linspace(offset, model.domain_size[0]-offset, num=srcpd)
    twodpoints = np.meshgrid(onedpoints, onedpoints)
    xcoords = np.ravel(twodpoints[0])
    ycoords = np.ravel(twodpoints[1])
    src_coordinates[:, 0] = xcoords
    src_coordinates[:, 1] = ycoords
    src_coordinates[:, -1] = model.origin[-1] + (2 + r/2)* spacing[-1]
    plt.scatter(xcoords, ycoords, s=0.01)
    plt.show()

    gridpoints = [tuple((int(floor((point[i]-origin[i])/grid.spacing[i])) - r/2)
                        for i in range(len(point))) for point in src_coordinates]

    src = PrecomputedSparseTimeFunction(name="src", grid=grid, npoint=nsrc, r=r, gridpoints=gridpoints, interpolation_coeffs=coeffs, nt=nt)
    
    ricker = RickerSource(time_range=time_axis, grid=grid, name="ricker", f0=0.008)
    for p in range(nsrc):
        src.data[:, p] = ricker.wavelet

    m = model.m

    # Create symbols for forward wavefield, source and receivers
    u = TimeFunction(name='u', grid=grid,
                     save=None,
                     time_order=2, space_order=so)

    rec = Receiver(name='rec', grid=grid, time_range=time_axis,
                   npoint=nsrc)

    s = model.grid.stepping_dim.spacing

    # Define PDE and update rule
    eq_time = solve(model.m * u.dt2 - u.laplace + model.damp * u.dt, u.forward)

    # Time-stepping stencil.
    eqns = [Eq(u.forward, eq_time, subdomain=model.grid.subdomains['physdomain'])]

    # Construct expression to inject source values
    src_term = src.inject(field=u.forward, expr=src * s**2 / m)

    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=u)
    # Substitute spacing terms to reduce flops
    op = Operator(eqns + src_term + rec_term, subs=model.spacing_map,
                    name='Forward')
    
    op.apply(dt=dt)

    print("u_norm", np.linalg.norm(u.data))
    


if __name__ == "__main__":
    run()

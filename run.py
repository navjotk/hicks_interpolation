import numpy as np

from scipy.signal.windows import kaiser

from devito import PrecomputedSparseFunction

from fwi.overthrust import overthrust_solver_iso, overthrust_model_iso


def run():
    initial_model_filename = "overthrust_3D_initial_model_2D.h5"
    tn = 4000
    so = 6
    dtype = np.float32
    datakey = "m0"
    nbl = 40
    model = overthrust_model_iso(initial_model_filename, datakey, so, nbl,
                                 dtype)
    shape = model.shape
    spacing = model.spacing
    coeffs = kaiser(M=8, beta=6.31)
    print(coeffs)

    sf = PrecomputedSparseFunction(name="src")
    nsrc = shape[0]
    src_coordinates = np.empty((nsrc, len(spacing)))
    src_coordinates[:, 0] = np.linspace(0., model.domain_size[0], num=nsrc)
    if len(shape) > 1:
        src_coordinates[:, 1] = np.array(model.domain_size)[1] * .5
        src_coordinates[:, -1] = model.origin[-1] + 2 * spacing[-1]

    solver_params = {'h5_file': initial_model_filename, 'tn': tn,
                     'space_order': so, 'dtype': dtype, 'datakey': datakey,
                     'nbl': nbl, 'src_coordinates': src_coordinates}

    solver = overthrust_solver_iso(**solver_params)
    solver.forward()


if __name__ == "__main__":
    run()

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
world = comm.size
rank = comm.Get_rank()
name = MPI.Get_processor_name()


def dot(a, b):
    if world == 1:
        result = np.dot(a, b)
    else:
        if rank == 0:
            a_row = a.shape[0]
            if a_row >= world:
                split = np.array_split(a, world, axis=0)
        else:
            split = None
        split = comm.scatter(split, root=0)
        split = np.dot(split, b)
        data = comm.gather(split, root=0)
        if rank == 0:
            result = np.vstack(data)
    return result

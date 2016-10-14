from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
threads = comm.Get_size()
data = np.zeros(1)
if rank == 0:
    for i in range(threads):
        data[0] = i*100
        MPI.COMM_WORLD.Send(data, dest = i)
else:
    print(data[0])
    MPI.COMM_WORLD.Recv(data, source = 0)
    print(data[0])
    data[0] = data[0] + 4
    print("hello world from process ", rank, threads)
    
    
    MPI.COMM_WORLD.Send(data, dest = 0)
    
if rank == 0:
    for i in range(threads):
        MPI.COMM_WORLD.Recv(data, source = ANY_SOURCE)
        print(data[0])
        
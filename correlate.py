from mpi4py import MPI
def correlate(a):
    res = []
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    for d in a:
        a = d['a']
        b = d['b']
        windowsize = d['windowsize']
        block = d['block']
        corr = np.zeros(len(a)-windowsize)
        y = b[block*windowsize:(1+block)*windowsize]
        #for j in range(len(a)-windowsize):
        #    x = a[0+j:windowsize+j]
        #    corr[j] = np.corrcoef(x, y)[0,1]
        res.append(block)
    return res
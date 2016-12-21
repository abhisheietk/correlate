import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.fftpack import rfft, irfft, fftfreq
import pickle
import os
from random import randrange

def threshhold(data, thld):
    length = len(data)
    tdata = np.zeros((length))
    for i_index, i in enumerate(data):
        if i < thld:
            tdata[i_index] = 0
        else:
            tdata[i_index] = i
    return tdata

def filterDC(data, freq = 10):
    f_data = rfft(data)
    for j in range(10):
        f_data[j] = 0
    return irfft(f_data)

def visualiseCorr(corr):
    fig, ax = plt.subplots(figsize=(40,15))
    image = corr
    ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    ax.set_title('dropped spines')
    
    # Move left and bottom spines outward by 10 points
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.show()

from ipyparallel import Client
import numpy as np
c = Client(profile='default')
dview = c[:]
print(len(dview))

@dview.parallel(block=True)
def correlateblock(data):
    import numpy as np
    res = []
    for d in data:
        a = d['a']
        b = d['b']
        block = d['block']
        sweep = d['sweep']
        #if not block % 5:
        windowsize = d['windowsize']
        #print('block',block)
        #print('sweep', sweep)
        #print('windowsize', windowsize)
        corr = np.zeros(len(sweep))
        y = a[block*windowsize:(1+block)*windowsize]
        #print('a[',block*windowsize,':',(1+block)*windowsize,']')
        
        for index, value in enumerate(sweep):#range(len(a)-windowsize):
            x = b[0+value:windowsize+value]
            #print('\tb[',0+value,':',windowsize+value,']', end=' ')
            corr[index] = np.corrcoef(x, y)[0,1]
            #print(corr[index])
        res.append({'block':block, 'sweep': sweep, 'corr':corr})
    return res

def correlate(a, b, windowsize, block, sweep):
    corr = np.zeros(len(sweep))
    jobs = []
    for j in block:
        ublock = j
        usweep = (sweep + (ublock * windowsize)) % (len(a)-windowsize)
        #print(usweep, len(usweep))
        sweepB = [usweep[i:i + 10000] for i in range(0, len(usweep), 10000)]
        jobs += [{'a':a, 'b':b, 'block':ublock, 'windowsize':windowsize, 
                 'sweep':i} for i in sweepB]
    #print("jobs:", jobs)
    #print("total jobs:", len(jobs))
    data = correlateblock(jobs)
    for i in data:
        newsweep = (np.array(i['sweep']) - (i['block'] * windowsize)) % (len(a)-windowsize)
        for ind, val in enumerate(newsweep):                
            #corr[np.where(sweep==val)[0]] += i['corr'][ind] 
            corr[np.searchsorted(sweep,val)] += i['corr'][ind]
                
    return np.array(corr)/len(block)

def filter_interest(corr, sweep, thld):
    newsweep = []
    smoothed_sweep = []
    newcorr = []
    #print(corr, len(corr))
    #print(sweep, len(sweep))
    #print(max(sweep))
    for index, val in enumerate(threshhold(corr, thld)):
        if val:
            newcorr.append(val)
            interest_point = sweep[index]
            smoothed_sweep.append(interest_point)
            #print([range(interest_point-50, interest_point+50)])
            newsweep += [i for i in range(interest_point-1, interest_point+1)]
    #print (len(smoothed_sweep))
    #print (len(newsweep))
    #plt.plot(sweep, corr)
    #plt.scatter(sweep, corr, s=80, marker=(5, 2)) #s=1, c=1, alpha=0.5) #plot(sweep, corr)
    #plt.plot(smoothed_sweep, newcorr)
    #plt.ylabel('Correlation')
    plt.scatter(sweep, corr, s=1, marker='.')
    plt.show()
    #for i in newsweep:
    return np.array(sorted(list(set(newsweep))))
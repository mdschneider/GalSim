"""Totally dumb thing Melanie Simet put together to test line-counting algorithms in Python.
"""

import numpy
import time

ntrials=10

def bufcount(filename):
    f = open(filename)                  
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.read # loop optimization

    buf = read_f(buf_size)
    while buf:
        if buf[0]=='#':
            lines -= 1
        lines += buf.count('\n')
        lines -= buf.count("\n#")
        buf = read_f(buf_size)
    f.close()
    return lines

def numpycount(filename):
    f = numpy.loadtxt(filename)
    return f.shape[0]
    
def dumbcount(filename):
    f = open(filename)
    nr_of_lines = sum(1 for line in f if not line.startswith('#'))
    f.close()
    return nr_of_lines

def main():
    filename = '../tests/lensing_reference_data/tmp.txt'
    filename_with_hashes = '../tests/lensing_reference_data/nfw_lens.dat'
    
    numpytime=0
    numpyhashtime=0
    dumbtime=0
    dumbhashtime=0
    buftime=0 
    bufhashtime=0
    
    for i in range(ntrials):
        t1=time.time()
        n = dumbcount(filename)
        t2=time.time()
        print "Dumb test: trial without comments", i, "took", t2-t1, "with result", n, "(should be 10000)"
        dumbtime+=t2-t1
    for i in range(ntrials):
        t1=time.time()
        n = dumbcount(filename_with_hashes)
        t2=time.time()
        print "Dumb test: trial with comments", i, "took", t2-t1, "with result", n, "(should be 599)"
        dumbhashtime+=t2-t1
    
    for i in range(ntrials):
        t1=time.time()
        n = bufcount(filename)
        t2=time.time()
        print "Buffer test: trial without comments", i, "took", t2-t1, "with result", n, "(should be 10000)"
        buftime+=t2-t1
    for i in range(ntrials):
        t1=time.time()
        n = bufcount(filename_with_hashes)
        t2=time.time()
        print "Buffer test: trial with comments", i, "took", t2-t1, "with result", n, "(should be 599)"
        bufhashtime+=t2-t1
    
    for i in range(ntrials):
        t1=time.time()
        n = numpycount(filename)
        t2=time.time()
        print "Numpy test: trial without comments", i, "took", t2-t1, "with result", n, "(should be 10000)"
        numpytime+=t2-t1
    for i in range(ntrials):
        t1=time.time()
        n = numpycount(filename_with_hashes)
        t2=time.time()
        print "Numpy test: trial with comments", i, "took", t2-t1, "with result", n, "(should be 599)"
        numpyhashtime+=t2-t1
    
    print "***FINAL RESULTS***"
    print "Dumb test:", dumbtime/ntrials, "without comments"
    print "Dumb test:", dumbhashtime/ntrials, "with comments"
    print "Buffer test:", buftime/ntrials, "without comments"
    print "Buffer test:", bufhashtime/ntrials, "with comments"
    print "Numpy test:", numpytime/ntrials, "without comments"
    print "Numpy test:", numpyhashtime/ntrials, "with comments"
    
if __name__=='__main__':
    main()

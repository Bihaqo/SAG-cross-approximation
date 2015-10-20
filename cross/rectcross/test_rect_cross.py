import numpy as np
import tt
import rect_cross

#Tested function
def myfun(x):
    return np.sin((x.sum(axis=1))) #** 2
    #return 1.0 / ((x.sum(axis=1)) + 1e-3)

    #return (x + 1).prod(axis=1)
    #return np.ones(x.shape[0])
d = 80
n = 5
r = 2

#sz = [n] * d
#ind_all = np.unravel_index(np.arange(n ** d), sz)
#ind_all = np.array(ind_all).T
#ft = reshape(myfun(ind_all), sz)
#xall = tt.tensor(ft, 1e-8)
#x0 = tt.tensor(ft, 1e-8)


x0 = tt.rand(n, d, r)

x1 = rect_cross.cross(myfun, x0, nswp=5, kickrank=1, rf=2)

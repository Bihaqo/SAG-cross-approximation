import numpy as np
import numpy.linalg as la
import sys
from rect_maxvol import rect_maxvol, maxvol 
import tt #For norm computation
from numpy.linalg import qr
# setting functions and variables for 'import *'
__all__ = ['cross']

def reshape(a, sz):
    return np.reshape(a, sz, order = 'F')

def mkron(a, b):
    return np.kron(a, b)


class node:
    def __init__(self):
        self.edges = [None for i in xrange(2)]
        
    
    """ Here we put in typical recipies for what we can do with a node """
    
class edge:
    def __init__(self):
        self.nodes = [None for i in xrange(2)]

    
    """ Here we put in typical recipies for what we can do with an edge """
    @staticmethod
    def init_qr_boundary(e):
        e.R = np.ones((1, 1))
            
    
class tt_as_tree:
    def __init__(self, d, node_type, edge_type, init_boundary = None):
        self.d = d
        self.nodes = [node_type() for i in xrange(d)]
        self.edges = [edge_type() for i in xrange(d + 1)]
        #  None - N - N - N - None
        for i in xrange(d):
            self.nodes[i].edges[0] = self.edges[i]
            self.nodes[i].edges[1] = self.edges[i + 1]
        
        if init_boundary is not None:
            init_boundary(self.edges[0])
            init_boundary(self.edges[d])
        
        for i in xrange(d - 1):
            self.edges[i].nodes[1] = self.nodes[i]
            self.edges[i + 1].nodes[0] = self.nodes[i]
        
        self.edges[0].nodes[0] = None
        self.edges[d].nodes[1] = None
        
    @staticmethod
    def from_tt(b, *args):
        c = tt_as_tree(b.d, *args)
        for i in xrange(b.d):
            c.nodes[i].core = b.core[b.ps[i] - 1:b.ps[i + 1] - 1].reshape(b.r[i], b.n[i], b.r[i + 1]).copy()
        return c

    def lr_sweep(self, node_update, edge_update=None):
        for i in xrange(self.d):
            node_update(self.nodes[i])
            if edge_update is not None:
                edge_update(self.edges[i + 1])

    def to_tt(self):
        core = [self.nodes[i].core for i in xrange(self.d)]
        return tt.tensor.from_list(core)


def left_qr_update(nd):
    cr = nd.core.copy()
    r1, n1, r2 = cr.shape
    cr = reshape(cr, (r1, n1 * r2))
    cr = np.dot(nd.edges[0].R, cr)
    r1 = cr.shape[0]
    cr = reshape(cr, (r1 * n1, r2))
    q, R = qr(cr)
    nd.core = reshape(q, (r1, n1, r2)).copy()
    nd.edges[1].R = R.copy()

def left_qr(R, Y):
    r1, n, m, r2 = Y.shape
    r1 = R.shape[1]
    Y = np.dot(R, reshape(Y, (r1, -1)))
    r1 = Y.shape[0]
    Y = reshape(Y, (r1 * n * m, r2))
    Y, R_new = np.linalg.qr(Y)
    Y = reshape(Y, (r1, n, m, -1))
    return Y, R_new


def right_qr(nd):
    Y = nd.Ycore
    R = nd.edges[1].R
    r1, n, m, r2 = Y.shape
    r2 = R.shape[0]
    Y = np.dot(reshape(Y, (-1, r2)), R)
    r2 = Y.shape[1]
    Y = reshape(Y, (r1, n * m * r2))
    Y = Y.T #Y is n m r2 x r1
    Y, R_new = np.linalg.qr(Y) # 
    Y = reshape(Y, (n, m, r2, -1))
    Y = np.transpose(Y, (3, 0, 1, 2))
    R_new = R_new.T
    nd.Ycore = Y.copy()
    nd.edges[0].R = R_new.copy()


def right_R(Y, R):
    r1, n, m, r2 = Y.shape
    r2 = R.shape[0]
    Y = np.dot(reshape(Y, (-1, r2)), R)
    Y = reshape(Y, (r1, n, m, -1))
    return Y


def left_R(nd):
    Y = nd.Ycore
    R = nd.edges[0].R
    r1, n, m, r2 = Y.shape
    r1 = R.shape[1]
    Y = np.dot(R, reshape(Y, (r1, -1)))
    Y = reshape(Y, (-1, n, m, r2))
    nd.Ycore = Y


def left_qr(nd):
    Y = nd.Ycore
    r1, n, m, r2 = Y.shape
    Y = reshape(Y, (r1 * n * m, r2))
    Y, R_new = np.linalg.qr(Y)
    Y = reshape(Y, (r1, n, m, -1))
    nd.Ycore = Y
    nd.edges[1].R = R_new.copy()




def init_alg(fun, x0):
    d = x0.d
    c = tt_as_tree(d, node, edge)
    x1 = tt.tensor.to_list(x0)
    c.fun = fun #Save the function
    c.fun_eval = 0
    for i in xrange(d):
        c.nodes[i].core = x1[i].copy()
    for i in xrange(d+1):
        c.edges[i].Ru = []
        c.edges[i].Rv = []
    
    c.edges[0].Ru = np.ones((1, 1))
    c.edges[d].Rv = np.ones((1, 1))
    c.edges[0].ind_left = np.empty((1, 0), dtype=np.int32)
    c.edges[d].ind_right = np.empty((1, 0), dtype=np.int32)
    return c


def my_chop2(sv, eps):
    if eps <= 0.0:
        r = len(sv)
        return r
    sv0 = np.cumsum(abs(sv[::-1]) ** 2)[::-1]
    ff = [i for i in range(len(sv0)) if sv0[i] < eps ** 2]
    if len(ff) == 0:
        return len(sv)
    else:
        return np.amin(ff)


def index_merge(i1, i2, i3):
    #Either i1 or i3 can be empty
    if i1 is not []:
        r1 = i1.shape[0]
    else:
        r1 = 1
    r2 = i2.shape[0]
    if i1 is not []:
        r3 = i3.shape[0]
    else:
        r3 = 1
    w1 = mkron(np.ones((r3 * r2, 1), dtype=np.int32), i1) #For some weird reasons, mkron(a, []) gives [],
                                                          #while mkron([], a) gives expection
    try: 
        w2 = mkron(i3, np.ones((r1 * r2, 1), dtype=np.int32))
    except:
        w2 = np.empty((w1.shape[0], 0))
    w3 = mkron(mkron(np.ones((r3, 1), dtype=np.int32), i2), np.ones((r1, 1), dtype=np.int32))
    J = np.hstack((w1, w3, w2))
    return J


def left_qr_maxvol(nd):
    cr = nd.core.copy()
    r1, n1, r2 = cr.shape
    cr = np.tensordot(nd.edges[0].Ru, cr, 1)
    #nd.edges[0].Ru = np.ones((1, 1))
    r1 = cr.shape[0]
    cr = reshape(cr, (r1 * n1, r2))
    q, Ru = qr(cr)
    ind, c = maxvol(q)
    Ru = np.dot(q[ind, :], Ru)
    q = c.copy()
    nd.core = reshape(q, (r1, n1, r2)).copy()
    nd.edges[1].Ru = Ru.copy()
    nd.maxvol_left = np.unravel_index(ind, (r1, n1), order='F')
    #The philosophical question if this index should be stored on the edge or in the node
    #The typical recomputation:
    #Take left index somewhere and update. For the first node it comes from the left edge
    #So, we can store ind_left on an edge, whereas ind_left_add in the node
    """ This is a logically separate function """
    i_left = nd.edges[0].ind_left
    #Update index
    w1 = mkron(np.ones((n1, 1), dtype=np.int32), i_left)
    w2 = mkron(reshape(np.arange(n1, dtype=np.int32),(-1, 1)), np.ones((r1, 1), dtype=np.int32))
    i_next = np.hstack((w1, w2))
    i_next = reshape(i_next, (r1 * n1, -1))
    i_next = i_next[ind, :]
    
    nd.edges[1].ind_left = i_next.copy()
    nd.edges[1].ind_left_add = i_next.copy()
    
    
def right_qr_maxvol(nd):
    cr = nd.core.copy()
    r1, n, r2 = cr.shape
    Rv = nd.edges[1].Rv.copy()
    cr = np.tensordot(cr, Rv, 1)
    r2 = cr.shape[2]
    cr = reshape(cr, (r1, -1))
    cr = cr.T
    q, Rv = np.linalg.qr(cr)
    ind, c = maxvol(q)
    Rv = np.dot(q[ind, :], Rv)
    #q = np.linalg.solve(q[ind, :].T, q.T).T
    q = c.copy()
    nd.edges[0].Rv = Rv.T.copy()
    q = reshape(q.T, (-1, n, r2))
    nd.core = q.copy()
    nd.maxvol_right = np.unravel_index(ind, (n, r2), order='F')
    """ Recomputation of right indices goes here """ 
    """ Finally, update indices """
    i_right = nd.edges[1].ind_right
    w1 = mkron(np.ones((r2, 1), dtype=np.int32), reshape(np.arange(n, dtype=np.int32),(-1, 1)))
    try:
        w2 = mkron(i_right, np.ones((n, 1), dtype=np.int32))
        i_next = np.hstack((w1, w2))
    except:
        i_next = w1

    
    i_next = reshape(i_next, (n * r2, -1))
    i_next = i_next[ind, :]
    nd.edges[0].ind_right = i_next.copy()
    nd.edges[0].ind_right_add = i_next.copy()
    
def apply_Ru(nd):
    cr = nd.core.copy()
    try:
        Ru = nd.edges[0].Ru.copy()
        cr = np.tensordot(Ru, cr, 1)
    except: 
        print 'Failed'
        pass
    nd.core = cr
    nd.edges[0].Ru = []
def apply_Rv(nd):
    cr = nd.core.copy()
    try:
        Rv = nd.edges[1].Rv.copy()
        cr = np.tensordot(cr, Rv, 1)
    except:
        pass
    nd.core = cr
    nd.edges[1].Rv = []
    
#Setup indices
def setup_indices(c):
    d = c.d
    #Setup left indices
    for i in xrange(d-1):
        nd = c.nodes[i]
        left_qr_maxvol(nd)  
    
    for i in xrange(d-1, 0, -1):
        nd = c.nodes[i]
        #compute_add(nd)
        right_qr_maxvol(nd)

def update_core_left(nd, fun, kickrank=3, tau=1.1, rf=2):
    fun_ev = 0
    cr = nd.core.copy()
    r1, n, r2 = cr.shape
    i_left = nd.edges[0].ind_left
    i_right = nd.edges[1].ind_right
    p = np.arange(n, dtype=np.int32); p = reshape(p, (-1, 1))
    J = index_merge(i_left, p, i_right)
    cr = fun(J)
    fun_ev += cr.size
    cr = reshape(cr, (-1, r2))
    #q, s1, v1 = la.svd(cr, full_matrices=False)
    #rr = my_chop2(s1, c.eps * la.norm(s1))
    #q = q[:, :rr]
    #R = np.diag(s1[:rr]).dot(v1[:rr, :])

    q, s1, v1 = la.svd(cr, full_matrices=False)
    R = np.diag(s1).dot(v1)

    #ind_new = maxvol(q)
    #nd.core = reshape(np.linalg.solve(q[ind_new, :].T, q.T).T, (r1, n, -1))
    ind_new, C = rect_maxvol(q, tau, maxK = q.shape[1] + kickrank + rf, min_add_K=kickrank)
    nd.core = reshape(C, (r1, n, -1))
    #QR = QQ^{-1}Q R
    #Ru = q[ind_new, :].dot(r)
    Ru = q[ind_new, :].dot(R)
    #And also recompute the next index
    i_left = nd.edges[0].ind_left
    #Update index full
    w1 = mkron(np.ones((n, 1), dtype=np.int32), i_left)
    w2 = mkron(reshape(np.arange(n, dtype=np.int32), (-1, 1)), np.ones((r1, 1), dtype=np.int32))
    
    i_next = np.hstack((w1, w2))
    
    i_next = reshape(i_next, (r1 * n, -1))
    
    nd.edges[1].ind_left = i_next[ind_new, :].copy()
    try:
        nd.edges[1].Ru = np.dot(Ru, nd.edges[1].Ru)
    except:
        nd.edges[1].Ru = Ru.copy()

    return fun_ev

def update_core_right(nd, fun, kickrank=1, tau=1.1, rf=2):
    fun_ev = 0
    cr = nd.core
    r1, n, r2 = cr.shape

    i_left = nd.edges[0].ind_left
    i_right = nd.edges[1].ind_right
    
    p = np.arange(n, dtype=np.int32); p = reshape(p, (-1, 1))
    
    J = index_merge(i_left, p, i_right)
    
    
    #J = np.hstack((w1, w3, w2))
    
    cr = fun(J)
    fun_ev += cr.size
    cr = reshape(cr, (r1, -1))
    cr = cr.T
    q, s1, v1 = la.svd(cr, full_matrices=False)
    #rr = my_chop2(s1, c.eps * la.norm(s1))
    #q = q[:, :rr]
    R = np.diag(s1).dot(v1)
    ind_new, C = rect_maxvol(q, 1.1, q.shape[1] + kickrank + rf, min_add_K = kickrank)
    nd.core = C

    #ind_new = maxvol(q)
    #nd.core = np.linalg.solve(q[ind_new, :].T, q.T).T
    nd.core = reshape(nd.core.T, (-1, n, r2))
    Rv = q[ind_new, :].dot(R)

    try:
        nd.edges[0].Rv = np.dot(nd.edges[0].Rv, Rv.T)
    except:
        nd.edges[0].Rv = Rv.T.copy()

    
    """ Finally, update indices """
    i_right = nd.edges[1].ind_right
    w1 = mkron(np.ones((r2, 1), dtype=np.int32), reshape(np.arange(n, dtype=np.int32),(-1, 1)))
    try:
        w2 = mkron(i_right, np.ones((n, 1), dtype=np.int32))
        i_next = np.hstack((w1, w2))
    except:
        i_next = w1
    
    
    i_next = reshape(i_next, (n * r2, -1))
    nd.edges[0].ind_right = i_next[ind_new, :].copy()
    return fun_ev

def iterate(c, kickrank=1, rf=2):
    """ Main iteration for the cross """
    d = c.d
    for i in xrange(c.d):
        c.nodes[i].edges[0].Rv = np.ones((1, 1))
        c.nodes[i].edges[1].Rv = np.ones((1, 1))
        c.nodes[i].edges[0].Ru = np.ones((1, 1))
        c.nodes[i].edges[1].Ru = np.ones((1, 1))

    for i in xrange(c.d):
        apply_Ru(c.nodes[i])
        c.fun_eval += update_core_left(c.nodes[i], c.fun, kickrank=kickrank, rf=rf)
   
    c.nodes[d-1].core = np.tensordot(c.nodes[d-1].core, c.nodes[d-1].edges[1].Ru, 1)
    
    for i in xrange(c.d-1, -1, -1):
        apply_Rv(c.nodes[i])
        c.fun_eval += update_core_right(c.nodes[i], c.fun, kickrank=kickrank, rf=rf)

    c.nodes[0].core = np.tensordot(c.nodes[0].edges[0].Rv, c.nodes[0].core, 1)

def get_tt(c):
    x1 = [c.nodes[i].core for i in xrange(c.d)]
    x1 = tt.tensor.from_list(x1)
    return x1


def cross(myfun, x0, nswp=10, eps=1e-10, eps_abs=0.0, kickrank=2, rf=2.0, verbose=True, stop_fun=None, approx_fun=None):
    """ Approximate a given black-box tensor in the tensor-train (TT) format 
    The algorithm uses original TT-cross method with maximum-volume submatrices replaced by rectangular maximum volume submatrices, 
    rectmaxvol algorithm proposed by A. Yu. Mikhalev

    :Reference: 


        I. V. Oseledets and E. E. Tyrtyshnikov. TT-cross approximation 
        for multidimensional arrays. Linear Algebra Appl., 432(1):70-88, 2010. 
        http://dx.doi.org/10.1016/j.laa.2009.07.024


    :param myfun: Vectorized function of d variables (it accepts I x D integer array as an input, and produces I numbers as output)
    :type myfun: function handle
    
    :param x0: Initial approximation, can be just random
    :type x0: tensor
    
    :param nswp: Auxiliary parameter, maximum number of sweeps of the cross method
    :type nswp: int
    
    :param eps: Required accuracy of the approximation
    :type eps: float
     
    :param eps_abs: Required absolute error of the approximation
    :type eps_abs: float


    :param kickrank: Tuning parameter, what is the minimal rank increase at each step, the larger the more robust 
                     is the method, but it is also slower and gives larger ranks
    :type kickrank: int
    
    :param rf: Tuning parameter, maximal rank grow factor at each microstep (i.e. the new rank can not be larger than rf * rold)
    :type rf: float

    :param verbose: Print convergence info or not
    :type verbose: bool

    
    :param approx_fun: 
    :type param: function
    
    :param stop_fun: Custom stopping criteria function, takes previous approximation and the new one
    :type param: function 

    :rtype: tensor
    
    
    :Examples:

        >>> #Simple function
        >>> import tt
        >>> import numpy as
        >>> from tt.cross import rect_cross as cross
        >>> x0 = tt.ones(2, 10)
        >>> fun = lambda x: x.sum(axis=1)
        >>> a = cross(myfun, x0)
        swp: 0/9 er = 8.3e-01 erank = 3.8
        swp: 1/9 er = 1.8e-01 erank = 6.7
        swp: 2/9 er = 9.3e-16 erank = 8.8
        >>> print a
        This is a 10-dimensional tensor 
        r(0)=1, n(0)=2 
        r(1)=4, n(1)=2 
        r(2)=6, n(2)=2 
        r(3)=10, n(3)=2 
        r(4)=12, n(4)=2 
        r(5)=12, n(5)=2 
        r(6)=12, n(6)=2 
        r(7)=8, n(7)=2 
        r(8)=4, n(8)=2 
        r(9)=2, n(9)=2 
        r(10)=1 
        >>> print a.round(1e-8)
        This is a 10-dimensional tensor 
        r(0)=1, n(0)=2 
        r(1)=2, n(1)=2 
        r(2)=2, n(2)=2 
        r(3)=2, n(3)=2 
        r(4)=2, n(4)=2 
        r(5)=2, n(5)=2 
        r(6)=2, n(6)=2 
        r(7)=2, n(7)=2 
        r(8)=2, n(8)=2 
        r(9)=2, n(9)=2 
        r(10)=1 
    """        
    c = init_alg(myfun, x0)

    c.eps = eps
    setup_indices(c)
    xold = x0.copy()

    for s in xrange(nswp):
        iterate(c, kickrank=kickrank, rf=rf)
        x1 = get_tt(c)
        #er = (x1 - xold).norm() / x1.norm()
        if approx_fun is not None:
            print 'Approximated value: %10.10f' % approx_fun(x1)
            er = abs(approx_fun(x1) - approx_fun(xold))
            er_rel = er / abs(approx_fun(x1))
        else:
            er = (x1 - xold).norm()
            er_rel = er / x1.norm()
        nrm = x1.norm()
        
        if verbose is True:
           print 'swp: %d/%d er_rel = %3.1e er_abs = %3.1e erank = %3.1f fun_eval: %d' % (s, nswp-1, er_rel, er, x1.erank, c.fun_eval)
        
        if stop_fun is not None:
            if stop_fun(xold, x1):
                break
        elif er < max(eps * nrm, eps_abs):
            break
        xold = x1.copy()
    return x1, c


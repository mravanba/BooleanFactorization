import numpy as np
import sys
import pdb
from utilities import log_ratio


class MatrixCompletion(object):
    
    def __init__(self,
                 O, #observed matrix (only the parts indicated by mask will be used)
                 K, #hidden dim
                 mask = None,#boolean matrix the same size as O
                 min_sum = True,                 
                 tol = 1e-4,#tolerance for message updates
                 learning_rate = .2, #damping parameter
                 max_iter = 500, #maximum number of message passing updates
                 verbose = False,
                 p_x_1 = .5, #the prior probability of x=1. For regularization use small or large values in [0,1]
                 p_y_1 = .5, #the prior probability of y=1. For regularization use small or large values in [0,1]
                 #note that when p_x and p_y are uniform the MAP assignment is not sensitive
                 #to the following values, assuming they are the same and above .5
                 p_1_given_1 = .99, #the model of the noisy channel: probability of observing 1 for the input of 1
                 p_0_given_0 = .99, #similar to the above
                ):
        
        assert(p_x_1 < 1 and p_x_1 > 0)
        assert(p_y_1 < 1 and p_y_1 > 0)
        assert(p_1_given_1 > .5 and p_1_given_1 < 1)
        assert(p_0_given_0 > .5 and p_0_given_0 < 1)                
        
        self.O = O.astype(int)
        self.M,self.N = O.shape
        self.K = K
        self.verbose = verbose

        assert(self.K < min(self.M,self.N))
        if mask is not None:
            assert(mask.shape[0] == self.M and mask.shape[1] == self.N)
            self.mask = mask.astype(bool)
        else:
            self.mask = np.ones(mat.shape, dtype=bool)
            
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.min_sum = min_sum
        self.num_edges = np.sum(self.mask)        

        self.update_adj_list()
        
        # will be used frequently
        self.pos_edges = np.nonzero(O[mask])[0]
        self.neg_edges = np.nonzero(1 - O[mask])[0]
        self.range_edges = np.arange(self.num_edges)
        self.cx = np.log(p_x_1) - np.log(1 - p_x_1)
        self.cy = np.log(p_y_1) - np.log(1 - p_y_1)
        self.co1 = np.log(p_1_given_1) - np.log(1. - p_0_given_0) #log(p(1|1)/p(1|0))
        self.co0 = np.log(1. - p_1_given_1) - np.log(p_0_given_0) ##log(p(0|1)/p(0|0))

    
    def init_msgs_n_marginals(self):
        self.marg_x = np.zeros((self.M, self.K))
        self.marg_y = np.zeros((self.N, self.K))
        self.in_x = np.zeros((self.num_edges, self.K)) #message going towards variable X: phi in the papger
        self.new_in_x = np.zeros((self.num_edges, self.K)) #the new one
        
        self.out_x = np.log((np.random.rand(self.num_edges, self.K)))#/self.M #message leaving variable x: phi_hat in the paper 
        self.in_y = np.zeros((self.num_edges, self.K)) #message leaving variable y: psi in the paper
        self.new_in_y = np.zeros((self.num_edges, self.K))
        self.out_y = np.log(np.random.rand(self.num_edges, self.K))#/self.N #psi_hat in the paper
        self.in_z = np.zeros((self.num_edges, self.K)) #gamma in the paper
        self.out_z = np.zeros((self.num_edges, self.K)) #gamma_hat in the paper
        
        
    def update_adj_list(self):
        ''' nbM: list of indices of nonzeros organized in rows
        nbM: list of indices of nonzeros organized in columns
        '''
        
        Mnz,Nnz = np.nonzero(self.mask)
        M = self.M
        N = self.N
        nbM = [[] for i in range(M)] 
        nbN = [[] for i in range(N)]

        for z in range(len(Mnz)):
            nbN[Nnz[z]].append(z)
            nbM[Mnz[z]].append(z)

        for i in range(M):
            nbM[i] = np.array(nbM[i], dtype=int)
        for i in range(N):
            nbN[i] = np.array(nbN[i], dtype=int)
            
        self.rows = nbM
        self.cols = nbN

        
    
    def run(self):
        self.init_msgs_n_marginals()
        iters = 1
        diff_msg = np.inf
        # self.msg_rec = []
        # self.mean_msg_rec = []
        while (diff_msg > self.tol and iters <= self.max_iter) or iters < 5:
            self.update_min_sum()#(outX, outY, inZ, outZ, newInX, newInY, posEdges, negEdges,  opt)
            diff_msg = np.max(np.abs(self.new_in_x - self.in_x))
            self.in_x *= (1. - self.learning_rate)
            self.in_x += self.learning_rate * (self.new_in_x)
            self.in_y *= (1. - self.learning_rate)
            self.in_y += self.learning_rate * (self.new_in_y)
            self.update_margs()
            if self.verbose:
                print "iter %d, diff:%f" %(iters, diff_msg)
            else:
                print ".",
                sys.stdout.flush()
            self.mean_msg_rec.append(np.count_nonzero(self.in_x.ravel < 1e-3) / float(np.prod(self.in_x.shape)))
            # if iters in [1,10,100] or diff_msg < self.tol:
            #     self.msg_rec.append(self.in_x.ravel().copy())
                
            iters += 1

        #recover X and Y from marginals and reconstruct Z
        self.X = (self.marg_x > 0).astype(int)
        self.Y = (self.marg_y > 0).astype(int)
        self.Z = (self.X.dot(self.Y.T) > 0).astype(int)

        
        
    def update_min_sum(self):
        self.in_z = np.minimum(np.minimum(self.out_x + self.out_y, self.out_x), self.out_y) #gamma update in the paper
        
        inz_pos = np.maximum(0.,self.in_z) # calculate it now, because we're chaning inz
        #find the second larges element along the 1st axis (there's also a 0nd! axis)
        inz_max_ind = np.argmax(self.in_z, axis=1)
        inz_max = np.maximum(-self.in_z[self.range_edges, inz_max_ind],0)
        self.in_z[self.range_edges, inz_max_ind] = -np.inf
        inz_max_sec = np.maximum(-np.max(self.in_z, axis=1),0) # update for gamma_hat in the paper
        sum_val = np.sum(inz_pos, axis=1)
        #penalties/rewards for confoming with observations
        sum_val[self.pos_edges] += self.co1
        sum_val[self.neg_edges] += self.co0
        
        tmp_inz_max = inz_max.copy()
        inz_pos =  sum_val[:, np.newaxis] - inz_pos
        
        for k in range(self.K):
            self_max_ind = np.nonzero(inz_max_ind == k)[0]#find the indices where the max incoming message is from k
            tmp_inz_max[self_max_ind] = inz_max_sec.take(self_max_ind)#replace the value of the max with the second largest value
            self.out_z[:, k] = np.minimum( tmp_inz_max, inz_pos[:,k])#see the update for gamma_hat
            tmp_inz_max[self_max_ind] = inz_max.take(self_max_ind)#fix tmp_iz_max for the next iter

        # update in_x and in_y: phi_hat and psi_hat in the paper
        self.new_in_x = np.maximum(self.out_z + self.out_y, 0) - np.maximum(self.out_y,0)
        self.new_in_y = np.maximum(self.out_z + self.out_x, 0) - np.maximum(self.out_x,0)

    

    def update_margs(self):
        #updates for phi and psi
        for m in range(self.M):
            self.marg_x[m,:] = np.sum(self.in_x.take(self.rows[m],axis=0), axis=0) + self.cx
            self.out_x[self.rows[m], :] = -self.in_x.take(self.rows[m],axis=0) + self.marg_x[m,:]

        for n in range(self.N):
            self.marg_y[n, :] = np.sum(self.in_y.take(self.cols[n], axis=0), axis=0) + self.cy
            self.out_y[self.cols[n], :] = -self.in_y.take(self.cols[n], axis=0) + self.marg_y[n,:]

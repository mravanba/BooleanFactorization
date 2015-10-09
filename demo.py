from matrix_completion import MatrixCompletion
from utilities import *


def random_instance():
    #dimensionality of the random input matrix
    M = 1000
    N = 1000
    #rank
    K = 2

    #information for generating a random matrix 
    #percentage of observed elements (for matrix completion)
    p_observe = .1
    #the probability of flipping each bit 
    p_flip = .2


    #generate a random instance of matrix Z = X \times Y, observed matrix O, and observation mask
    mat_dic = get_random_matrices(M, N, K, p_flip = p_flip, p_observe = p_observe, p_x_1 = .6, p_y_1 = .6)

    #inference method
    print "running inference"
    #the input to inference is a 1) matrix O, 2) rank K, 3) a Boolean matrix mask indicating which values in O should be used/observed
    #for more options, e.g. the priors over X and Y, number of iterations, see the code for MatrixCompletion
    comp = MatrixCompletion(mat_dic['O'], K, mask = mat_dic['mask'], min_sum = True, verbose = True)
    comp.run()

    #printing some info now
    print "Z: %d by %d input matrix.\n %.1f percent of bits were flipped.\n observing %.2f percent of elements.\n reconstruction error: %.3f" \
        %(M,N,p_flip*100, p_observe*100, rec_error(comp.Z, mat_dic['Z']))
    print "non-zero percentage X:%.3f Y:%.3f Z:%.3f O:%.3f and Zh:%.3f mask:%.3f" %(
        density(mat_dic['X']), density(mat_dic['Y']),
        density(mat_dic['Z']), density(mat_dic['O']),
        density(comp.Z), density(comp.mask))

    
if __name__ == "__main__":
    random_instance()
    

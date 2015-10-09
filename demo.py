from matrix_completion import MatrixCompletion
from utilities import *


def random_instance():
    M = 1000
    N = 1000
    K = 2
    p_observe = 1
    p_flip = 0
    min_sum = True
    mat_dic = get_random_matrices(M, N, K, p_flip = p_flip, p_observe = p_observe, p_x_1 = .6, p_y_1 = .6)
    print "running inference"
    comp = MatrixCompletion(mat_dic['O'], K, mask = mat_dic['mask'], min_sum = True, verbose = True)
    comp.run()
    
    print "Z: %d by %d input matrix.\n %.1f percent of bits were flipped.\n observing %.2f percent of elements.\n reconstruction error: %.3f" \
        %(M,N,p_flip*100, p_observe*100, rec_error(comp.Z, mat_dic['Z']))
    print "non-zero percentage X:%.3f Y:%.3f Z:%.3f O:%.3f and Zh:%.3f mask:%.3f" %(
        density(mat_dic['X']), density(mat_dic['Y']),
        density(mat_dic['Z']), density(mat_dic['O']),
        density(comp.Z), density(comp.mask))
    

    
if __name__ == "__main__":
    random_instance()
    

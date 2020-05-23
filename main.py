import numpy as np

'''Main function of method'''
def localVariance(func, dimension, x0, p0, iterNum):
    
    '''create additional matrix of vectors h^i (i=[1, 2n]) where h^(2j-1) = e(j), h^(2j) = -e(j)'''
    def __calcH(dimension):
        I = np.eye(dimension)
        res = np.zeros((dimension, 2 * dimension))
        for col in range(dimension):
            res[:, 2*col] = I[:, col]
            res[:, 2*col+1] = -I[:, col]
        return res
    
    # init additional data
    mH = __calcH(dimension)
    x = x0
    p = p0
    f0 = func(x)
    vx = []
    # main loop
    for _ in range(iterNum):
        for i in range(2 * dimension): 
            f1 = func(x + p*mH[:,i])
            while f1 < f0:
                x += p*mH[:,i]
                f0 = func(x)
                f1 = func(x + p*mH[:,i])
            i+=1
        vx.append(x)
        p /= 2
    return vx


    

    pass
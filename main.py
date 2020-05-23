import numpy as np

'''test function, x is a vector'''
def testfunction(x):
    return 2*x[0]**4 + 12*(x[0]*x[0])*(x[2]*x[2]) + 18*(x[2]**4) + x[1]**4 \
            - 8*(x[0]**3)*x[2] - 8*(x[0])*(x[2]**3) + 24*(x[1]*x[1])*(x[2]*x[2]) \
            + 8*(x[1]**3)*x[2] + 32*(x[2]**3)*x[1] + 2*x[2]*x[2] + 3*x[1]*x[1] \
            - 4*x[0]*x[2] + 12*x[1] - 8           


'''Finds argmin of function'''
def localVariance(func, x0, p0, iterNum):
    
    '''create additional matrix of vectors h^i (i=[1, 2n]) where h^(2j-1) = e(j), h^(2j) = -e(j)'''
    def __calcH(dimension):
        I = np.eye(dimension)
        res = np.zeros((dimension, 2 * dimension))
        for col in range(dimension):
            res[:, 2*col] = I[:, col]
            res[:, 2*col+1] = -I[:, col]
        return res
    
    # init additional data
    mH = __calcH(len(x0))
    x = x0
    p = p0
    f0 = func(x)
    vx = []
    # main loop
    for _ in range(iterNum):
        for i in range(2 * len(x0)): 
            f1 = func(x + p*mH[:,i])
            while f1 < f0:
                x += p*mH[:,i]
                f0 = func(x)
                f1 = func(x + p*mH[:,i])
            i+=1
        vx.append(x.copy())
        p /= 2
    return vx

x0 = [0, -1, -1]
fv = testfunction(x0)
print(fv)
optx = localVariance(testfunction, x0, 1, 20) 
print(optx)
for x in optx :
    print(testfunction(x))

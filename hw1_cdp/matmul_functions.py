import numpy as np
from numba import njit, cuda, prange
import timeit
import math

def matmul_transpose_trivial(X):
    rows = len(X)
    cols = len(X[0])
    
    result = np.zeros((rows, rows))
    
    for i in range(rows):
        for j in range(rows):
            for k in range(cols):
                result[i][j] += X[i][k] * X[j][k]

    return result


@njit(parallel=True)
def matmul_transpose_numba(X):
    rows = len(X)
    cols = len(X[0])
    
    result = np.zeros((rows, rows))
    
    for i in prange(rows):
        for j in prange(rows):
            for k in prange(cols):
                result[i][j] += X[i][k] * X[j][k]
    
    return result

def matmul_transpose_gpu(X):
    rows, cols = X.shape
    result = np.zeros((rows, rows), dtype=np.float32)

    d_X = cuda.to_device(X)
    d_result = cuda.to_device(result)

    threads_per_block = 1024
    blocks_per_grid = 1  

    matmul_kernel[blocks_per_grid, threads_per_block](d_X, d_result, cols, rows)

    return d_result.copy_to_host()


@cuda.jit
def matmul_kernel(A, C, cols, rows):
    thread_id = cuda.threadIdx.x  
    total_elements = rows * rows  

    for idx in range(thread_id, total_elements, 1024):  
        row = idx // rows
        col = idx % rows

        if row < rows and col < rows:
            temp_sum = 0.0
            for k in range(cols):
                temp_sum += A[row, k] * A[col, k]
            C[row, col] = temp_sum

def verify_solution():
    X = np.random.randn(784, 128)
    Xt = X.copy().transpose()

    if not np.allclose(matmul_transpose_trivial(X), np.matmul(X, Xt)):
        print('[-] matmul_transpose_trivial failed')
        exit(0)
    else:
        print('[+] matmul_transpose_trivial passed')

    if not np.allclose(matmul_transpose_numba(X), np.matmul(X, Xt)):
        print('[-] matmul_transpose_numba failed')
        exit(0)
    else:
        print('[+] matmul_transpose_numba passed')

    if not np.allclose(matmul_transpose_gpu(X), np.matmul(X, Xt)):
        print('[-] matmul_transpose_gpu failed')
        exit(0)
    else:
        print('[+] matmul_transpose_gpu passed')

    print('[+] All tests passed\n')


# this is the comparison function - keep it as it is, don't change X or Y.
def matmul_comparison():
    X = np.random.randn(784, 128)
    Xt = X.copy().transpose()

    def timer(f, functionParameters):
        return min(timeit.Timer(lambda: f(X) if functionParameters == 1 else f(X, Xt)).repeat(3, 100))

    # print('Python:', timer(matmul_transpose_trivial, 1)) we will not consider this since it takes infinite time :)
    print('Numpy:', timer(np.matmul, 2))
    print('Numba:', timer(matmul_transpose_numba, 1))
    print('CUDA:', timer(matmul_transpose_gpu, 1))


if __name__ == '__main__':
    verify_solution()
    matmul_comparison()

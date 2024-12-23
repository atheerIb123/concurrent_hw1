import numpy as np
from numba import cuda, njit, prange, float32
import timeit


def max_cpu(A, B):
    """
    Returns
    -------
    list of lists
        element-wise maximum between A and B
    """
    for i, row in enumerate(A):
        for j, e in enumerate(row):
            A[i][j] = max(e, B[i][j])
    return A


@njit(parallel=True)
def max_numba(A, B):
    """
    Returns
    -------
    np.array
        element-wise maximum between A and B
    """
    rows_num = len(A)
    cols_num = len(A[0])
    result_matrix = np.zeros((rows_num, cols_num), dtype=A.dtype)

    for i in prange(rows_num):
        for j in prange(cols_num):
            result_matrix[i][j] = max(A[i][j], B[i][j])
    return result_matrix


def max_gpu(A, B):
    """
    Returns
    -------
    np.array
        element-wise maximum between A and B
    """
    rows, cols = A.shape

    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.device_array((rows, cols), dtype=A.dtype)

    block_size = (20, 50)
    grid_size = ((rows + block_size[0] - 1) // block_size[0],
                 (cols + block_size[1] - 1) // block_size[1])

    max_kernel[grid_size, block_size](d_A, d_B, d_C)

    return d_C.copy_to_host()


@cuda.jit
def max_kernel(A, B, C):
    row, col = cuda.grid(2)

    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = max(A[row, col], B[row, col])


def verify_solution():
    A = np.random.randint(0, 256, (1000, 1000)).astype(np.int32)
    B = np.random.randint(0, 256, (1000, 1000)).astype(np.int32)

    if not np.all(max_cpu(A.copy(), B.copy()) == np.maximum(A, B)):
        print('[-] max_cpu failed')
        exit(0)
    else:
        print('[+] max_cpu passed')

    if not np.all(max_numba(A.copy(), B.copy()) == np.maximum(A, B)):
        print('[-] max_numba failed')
        exit(0)
    else:
        print('[+] max_numba passed')

    if not np.all(max_gpu(A.copy(), B.copy()) == np.maximum(A, B)):
        print('[-] max_gpu failed')
        exit(0)
    else:
        print('[+] max_gpu passed')

    print('[+] All tests passed\n')

# this is the comparison function - keep it as it is.
def max_comparison():
    A = np.random.randint(0, 256, (1000, 1000)).astype(np.int32)
    B = np.random.randint(0, 256, (1000, 1000)).astype(np.int32)

    def timer(f):
        return min(timeit.Timer(lambda: f(A, B)).repeat(3, 10))

    print('[*] CPU:', timer(max_cpu))
    print('[*] Numba:', timer(max_numba))
    print('[*] CUDA:', timer(max_gpu))


if __name__ == '__main__':
    verify_solution()
    max_comparison()



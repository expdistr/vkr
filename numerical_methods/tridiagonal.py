import numpy as np

def solve_block_tridiag(R, L, M, F):
    """
    Решает блочно-трёхдиагональную систему R[i]*x[i-1] + L[i]*x[i] + M[i]*x[i+1] = F[i].
    R[0] = None, M[last] = None.
    Возвращает список векторов x[i].
    """
    n = len(L)
    # Прямой ход (факторизация блоков)
    alpha = [None]*n
    gamma = [None]*n
    # Первый блок
    alpha[0] = L[0].copy()
    gamma[0] = F[0].copy()
    
    for i in range(1, n):
        # Находим X = inv(alpha[i-1]) * M[i-1], Y = inv(alpha[i-1]) * gamma[i-1]
        inv_prev = np.linalg.inv(alpha[i-1])
        if M[i-1] is not None:
            Z = inv_prev @ M[i-1]
        else:
            Z = None
        Y = inv_prev @ gamma[i-1]
        # Вычисляем обновлённые L и F
        if Z is not None:
            alpha[i] = L[i] - R[i] @ Z
        else:
            alpha[i] = L[i]
        gamma[i] = F[i] - R[i] @ Y
    
    # Обратный ход
    x = [None]*n
    # Решаем последний блок
    x[-1] = np.linalg.solve(alpha[-1], gamma[-1])
    for i in range(n-2, -1, -1):
        if M[i] is not None:
            rhs = gamma[i] - M[i] @ x[i+1]
        else:
            rhs = gamma[i]
        x[i] = np.linalg.solve(alpha[i], rhs)
    return x

def matrix_tridiagonal_solve(A_blocks, B_blocks, C_blocks, F_blocks):
    """
    Решает систему с блочно-трёхдиагональной матрицей методом матричной прогонки.
    A_blocks[i], B_blocks[i], C_blocks[i] — матрицы размером m x m
    F_blocks[i] — вектор размерности m
    """
    n = len(B_blocks)
    m = B_blocks[0].shape[0]

    # Прогоночные коэффициенты
    P = [np.zeros((m, m)) for _ in range(n)]
    Q = [np.zeros(m) for _ in range(n)]

    # Прямой ход
    P[0] = np.linalg.inv(B_blocks[0]) @ C_blocks[0]
    Q[0] = np.linalg.inv(B_blocks[0]) @ F_blocks[0]

    for i in range(1, n - 1):
        inv = np.linalg.inv(B_blocks[i] - A_blocks[i] @ P[i - 1])
        P[i] = inv @ C_blocks[i]
        Q[i] = inv @ (F_blocks[i] + A_blocks[i] @ Q[i - 1])

    # Последний шаг
    u = np.zeros((n, m))
    u[-1] = np.linalg.inv(B_blocks[-1] - A_blocks[-1] @ P[-2]) @ (F_blocks[-1] + A_blocks[-1] @ Q[-2])

    # Обратный ход
    for i in reversed(range(n - 1)):
        u[i] = P[i] @ u[i + 1] + Q[i]

    return u.flatten()

def solve_tridiagonal(A, b):
    """
    Решает трёхдиагональную систему Ax = b.
    Вызывает LinAlgError, если матрица вырождена.
    """
    if np.linalg.matrix_rank(A) < A.shape[0]:
        raise np.linalg.LinAlgError("Матрица вырождена (неполный ранг). Проверьте граничные условия.")
    return np.linalg.solve(A, b) 
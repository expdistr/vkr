import numpy as np
from .tridiagonal import solve_tridiagonal, matrix_tridiagonal_solve

def create_grid(a, b, h):
    """Создаёт равномерную сетку на [a,b] с шагом h."""
    N = int((b - a) / h)
    x = np.linspace(a, b, N + 1)
    return x, N

def build_matrix_and_rhs(task_config, x, h):
    """Строит матрицу системы и правую часть для дифференциального уравнения."""
    N = len(x) - 1
    f = task_config['f']
    order = task_config['order']

    A = np.zeros((N+1, N+1))
    b = np.zeros(N+1)

    if order == 4:
        _build_4th_order_matrix(A, b, x, h, f, N)
    elif order == 5:
        _build_5th_order_matrix(A, b, x, h, f, N)
    elif order == 6:
        _build_6th_order_matrix(A, b, x, h, f, N)

    return A, b

def _build_4th_order_matrix(A, b, x, h, f, N):
    """Строит матрицу для уравнения 4-го порядка."""
    for i in range(2, N-1):
        A[i, i-2] = 1
        A[i, i-1] = -4
        A[i, i] = 6
        A[i, i+1] = -4
        A[i, i+2] = 1
        b[i] = h**4 * f(x[i])
    
    # Граничные условия
    A[0, 0] = 1
    A[1, 0:3] = [1, -2, 1]
    A[N-1, N-2:N+1] = [1, -2, 1]
    A[N, N] = 1

def _build_5th_order_matrix(A, b, x, h, f, N):
    """Строит матрицу для уравнения 5-го порядка."""
    for i in range(3, N-3):
        A[i, i-3] = 1
        A[i, i-2] = -5
        A[i, i-1] = 10
        A[i, i+1] = -10
        A[i, i+2] = 5
        A[i, i+3] = -1
        b[i] = h**5 * f(x[i])
    
    # Граничные условия
    A[0, 0] = 1
    A[1, 0:2] = [-1/h, 1/h]
    A[2, 0:3] = [1/h**2, -2/h**2, 1/h**2]
    b[0:3] = 0
    
    A[N-2, N-2:N+1] = [1/h**2, -2/h**2, 1/h**2]
    A[N-1, N-1:] = [-1/h, 1/h]
    A[N, N] = 1
    b[N-2:N+1] = [0, np.exp(1), np.exp(1)]

def _build_6th_order_matrix(A, b, x, h, f, N):
    """Строит матрицу для уравнения 6-го порядка."""
    for i in range(3, N-3):
        A[i, i-3] = -1
        A[i, i-2] = 6
        A[i, i-1] = -15
        A[i, i] = 20
        A[i, i+1] = -15
        A[i, i+2] = 6
        A[i, i+3] = -1
        b[i] = 0
    
    # Граничные условия
    A[0, 0] = 1
    A[1, 0:2] = [-1/h, 1/h]
    A[2, 0:3] = [1/h**2, -2/h**2, 1/h**2]
    b[0:3] = 0
    
    A[N-2, N-2:N+1] = [1/h**2, -2/h**2, 1/h**2]
    A[N-1, N-1:] = [-1/h, 1/h]
    A[N, N] = 1
    b[N-2:N+1] = 0

def solve_equation(order, task_config):
    """
    Решает дифференциальное уравнение заданного порядка.
    
    Аргументы:
        order (int): Порядок дифференциального уравнения (4, 5 или 6)
        task_config (dict): Конфигурация для конкретной задачи
        
    Возвращает:
        tuple: (x, u_numeric, u_exact), где x - точки сетки,
               u_numeric - численное решение,
               u_exact - точное решение
    """
    # Фиксированные параметры
    a = task_config['a']
    b = task_config['b']
    h = 0.01  # Фиксированный шаг

    x, N = create_grid(a, b, h)
    
    # Получаем точное решение
    u_exact = task_config['u_exact'](x)

    # Строим и решаем систему
    try:
        if order == 4:
            A, rhs = build_matrix_and_rhs(task_config, x, h)
            u_numeric = solve_tridiagonal(A, rhs)
        else:
            m = 6 if order == 5 else 7
            A_blocks = [np.zeros((m, m)) for _ in range(N)]
            B_blocks = []
            C_blocks = [np.zeros((m, m)) for _ in range(N)]
            F_blocks = []
            
            # Формируем систему
            for i in range(N + 1):
                if i == 0 or i == N:
                    B = np.eye(m)
                    F = np.zeros(m)
                else:
                    B = _build_block_matrix(order, m, h)
                    F = np.zeros(m)
                    if 2 <= i < N - 2:
                        F[m//2] = task_config['f'](x[i])
                
                B_blocks.append(B)
                F_blocks.append(F)
            
            # Настраиваем граничные блоки
            for i in range(N):
                if i > 0 and i < N - 3:
                    A_blocks[i] = np.zeros((m, m))
                    A_blocks[i][m//2, m//2] = 1
                    C_blocks[i] = np.zeros((m, m))
                    C_blocks[i][m//2, m//2] = 1
            
            u_numeric = matrix_tridiagonal_solve(A_blocks, B_blocks, C_blocks, F_blocks)
            u_numeric = u_numeric.reshape(-1, m)[:, m//2]
            
    except np.linalg.LinAlgError as e:
        print(f"Ошибка: {str(e)}")
        return None, None, None

    return x, u_numeric, u_exact

def _build_block_matrix(order, m, h):
    """Строит блочную матрицу для заданного порядка."""
    if order == 5:
        return np.vstack([
            np.eye(m)[j] if j != m//2 else 
            np.array([[1, -3, 3, -3, 3, -1]]) / (2 * h**5)
            for j in range(m)
        ])
    elif order == 6:
        return np.vstack([
            np.eye(m)[j] if j != m//2 else 
            np.array([[-1, 6, -15, 20, -15, 6, -1]]) / h**6
            for j in range(m)
        ])
    else:
        raise ValueError(f"Неподдерживаемый порядок: {order}") 
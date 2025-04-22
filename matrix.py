import numpy as np
import matplotlib.pyplot as plt
from math import sin, pi, exp

def solve_block_tridiag(R, L, M, F):
    """
    Решение блочно-трёхдиагональной системы R[i]*x[i-1] + L[i]*x[i] + M[i]*x[i+1] = F[i].
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
    Решает систему с блочно-трехдиагональной матрицей методом матричной прогонки.
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

def create_grid(a, b, h):
    N = int((b - a) / h)
    x = np.linspace(a, b, N + 1)
    return x, N

def build_matrix_and_rhs(task_key, x, h):
    N = len(x) - 1
    f = TASKS[task_key]['f']
    order = TASKS[task_key]['order']

    A = np.zeros((N+1, N+1))
    b = np.zeros(N+1)

    if order == 4:
        for i in range(2, N-1):
            A[i, i-2] = 1
            A[i, i-1] = -4
            A[i, i] = 6
            A[i, i+1] = -4
            A[i, i+2] = 1
            b[i] = h**4 * f(x[i])
        A[0, 0] = 1
        A[1, 0:3] = [1, -2, 1]
        A[N-1, N-2:N+1] = [1, -2, 1]
        A[N, N] = 1

    elif order == 5:
        for i in range(3, N-3):
            A[i, i-3] = 1
            A[i, i-2] = -5
            A[i, i-1] = 10
            A[i, i+1] = -10
            A[i, i+2] = 5
            A[i, i+3] = -1
            b[i] = h**5 * f(x[i])
        # Граничные условия: u(0)=0, u'(0)=0, u''(0)=0
        A[0, 0] = 1
        A[1, 0:2] = [-1/h, 1/h]
        A[2, 0:3] = [1/h**2, -2/h**2, 1/h**2]
        b[0:3] = 0
        # u''(1)=0, u'(1)=e, u(1)=e
        A[N-2, N-2:N+1] = [1/h**2, -2/h**2, 1/h**2]
        A[N-1, N-1:] = [-1/h, 1/h]
        A[N, N] = 1
        b[N-2:N+1] = [0, exp(1), exp(1)]

    elif order == 6:
        for i in range(3, N-3):
            A[i, i-3] = -1
            A[i, i-2] = 6
            A[i, i-1] = -15
            A[i, i] = 20
            A[i, i+1] = -15
            A[i, i+2] = 6
            A[i, i+3] = -1
            b[i] = 0
        # u(0)=0, u'(0)=0, u''(0)=0
        A[0, 0] = 1
        A[1, 0:2] = [-1/h, 1/h]
        A[2, 0:3] = [1/h**2, -2/h**2, 1/h**2]
        b[0:3] = 0
        # u''(1)=0, u'(1)=0, u(1)=0
        A[N-2, N-2:N+1] = [1/h**2, -2/h**2, 1/h**2]
        A[N-1, N-1:] = [-1/h, 1/h]
        A[N, N] = 1
        b[N-2:N+1] = 0

    return A, b

def solve_tridiagonal(A, b):
    if np.linalg.matrix_rank(A) < A.shape[0]:
        raise np.linalg.LinAlgError("Матрица вырождена (неполный ранг). Проверьте граничные условия.")
    return np.linalg.solve(A, b)

def solve_equation(order):
    # Фиксированные параметры
    a = 0
    if order == 4:
        b = np.pi
        h = 0.01
    elif order == 5:
        b = 1
        h = 0.01
    elif order == 6:
        b = 1
        h = 0.01
    else:
        raise ValueError("Неверный порядок уравнения. Выберите 4, 5 или 6.")

    n_points = int((b - a) / h) + 1
    x = np.linspace(a, b, n_points)

    if order == 4:
        m = 5  # Размерность блока (5 точек для четвертой производной)
        u_exact = np.sin(x)
    elif order == 5:
        m = 6  # 6 точек для пятой производной
        u_exact = np.exp(x) - 1 - x - x**2 / 2
    elif order == 6:
        m = 7  # 7 точек для шестой производной
        u_exact = np.zeros_like(x)

    A_blocks = [np.zeros((m, m)) for _ in range(n_points - 1)]
    B_blocks = []
    C_blocks = [np.zeros((m, m)) for _ in range(n_points - 1)]
    F_blocks = []

    # Формирование системы
    for i in range(n_points):
        if i == 0 or i == n_points - 1:
            B = np.eye(m)
            F = np.zeros(m)
        else:
            if order == 4:
                B = np.array([
                    [1, -4, 6, -4, 1]
                ]) / h**4
            elif order == 5:
                B = np.array([
                    [1, -3, 3, -3, 3, -1]
                ]) / (2 * h**5)
            elif order == 6:
                B = np.array([
                    [-1, 6, -15, 20, -15, 6, -1]
                ]) / h**6
            B = np.vstack([np.eye(m)[j] if j != m//2 else B for j in range(m)])
            F = np.zeros(m)
            if 2 <= i < n_points - 2:
                if order == 4:
                    F[m//2] = np.sin(x[i])
                elif order == 5:
                    F[m//2] = np.exp(x[i])

        B_blocks.append(B)
        F_blocks.append(F)

    # Граничные условия
    for i in range(n_points - 1):
        if i > 0 and i < n_points - 3:
            A_blocks[i] = np.zeros((m, m))
            A_blocks[i][m//2, m//2] = 1
            C_blocks[i] = np.zeros((m, m))
            C_blocks[i][m//2, m//2] = 1

    try:
        if order == 4:
            A, rhs = build_matrix_and_rhs('4', x, h)
            u_numeric = solve_tridiagonal(A, rhs)
        else:
            u_numeric = matrix_tridiagonal_solve(A_blocks, B_blocks, C_blocks, F_blocks)
    except np.linalg.LinAlgError:
        print("Ошибка: невозможно обратить матрицу")
        return

    if order == 4:
        u_numeric = u_numeric
    else:
        u_numeric = u_numeric.reshape(-1, m)[:, m//2]

    # Вывод численных значений решения
   ## print(f"\nЧисленное решение для уравнения {order}-го порядка:")
  ##  for i, (x_val, u_val) in enumerate(zip(x, u_numeric)):
   ##     print(f"x[{i}] = {x_val:.4f}, u[{i}] = {u_val:.4f}")

    # Вывод аналитического решения для сравнения
 ##   print(f"\nАналитическое решение для уравнения {order}-го порядка:")
 ##   for i, (x_val, u_val) in enumerate(zip(x, u_exact)):
  ##      print(f"x[{i}] = {x_val:.4f}, u[{i}] = {u_val:.4f}")

    error = np.max(np.abs(u_exact - u_numeric))
    print(f"\nОшибка для {order}-го порядка:", error)

    # Визуализация результатов
    plt.figure()
    plt.plot(x, u_exact, label="Аналитическое")
    plt.plot(x, u_numeric, 'o', markersize=3, label="Численное")
    plt.title(f"Уравнение {order}-го порядка")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.grid(True)
    plt.savefig("myplot.png")

# Настройки задач
TASKS = {
    '4': {
        'a': 0,
        'b': pi,
        'f': lambda x: np.sin(x),
        'u_exact': lambda x: np.sin(x),
        'order': 4,
        'boundary_conditions': '4th'
    },
    '5': {
        'a': 0,
        'b': 1,
        'f': lambda x: np.exp(x),
        'u_exact': lambda x: np.exp(x) - 1 - x - x**2/2,
        'order': 5,
        'boundary_conditions': '5th'
    },
    '6': {
        'a': 0,
        'b': 1,
        'f': lambda x: 0*x,
        'u_exact': lambda x: 0*x,
        'order': 6,
        'boundary_conditions': '6th'
    }
}

if __name__ == "__main__":
    print("Выберите уравнение для решения:")
    print("1. Четвёртый порядок")
    print("2. Пятый порядок")
    print("3. Шестой порядок")

    choice = input("Введите номер (1/2/3): ")
    order_map = {"1": 4, "2": 5, "3": 6}
    order = order_map.get(choice, None)

    if order is None:
        print("Неверный выбор.")
    else:
        solve_equation(order)

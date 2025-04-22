import matplotlib.pyplot as plt
import numpy as np

def plot_solution(x, u_numeric, u_exact, order, save_path=None):
    """
    Строит график численного и точного решений.
    
    Аргументы:
        x (np.ndarray): Точки сетки
        u_numeric (np.ndarray): Численное решение
        u_exact (np.ndarray): Точное решение
        order (int): Порядок дифференциального уравнения
        save_path (str, optional): Путь для сохранения графика. Если None, график отображается.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x, u_exact, label="Точное решение")
    plt.plot(x, u_numeric, 'o', markersize=3, label="Численное решение")
    plt.title(f"Дифференциальное уравнение {order}-го порядка")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def print_solution_info(x, u_numeric, u_exact, order):
    """
    Выводит информацию о численном решении.
    
    Аргументы:
        x (np.ndarray): Точки сетки
        u_numeric (np.ndarray): Численное решение
        u_exact (np.ndarray): Точное решение
        order (int): Порядок дифференциального уравнения
    """
    error = np.max(np.abs(u_exact - u_numeric))
    print(f"\nРешение для уравнения {order}-го порядка:")
    print(f"Максимальная погрешность: {error:.2e}")
    
    # Выводим несколько точек для примера
    print("\nПримеры точек (x, численное, точное):")
    step = len(x) // 5  # Показываем 5 точек
    for i in range(0, len(x), step):
        print(f"x[{i}] = {x[i]:.4f}, u_числ = {u_numeric[i]:.4f}, u_точн = {u_exact[i]:.4f}") 
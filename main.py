from numerical_methods import solve_equation, TASKS, plot_solution, print_solution_info

def main():
    print("Выберите уравнение для решения:")
    print("1. Четвёртого порядка")
    print("2. Пятого порядка")
    print("3. Шестого порядка")

    choice = input("Введите номер (1/2/3): ")
    order_map = {"1": 4, "2": 5, "3": 6}
    order = order_map.get(choice)

    if order is None:
        print("Неверный выбор.")
        return

    # Получаем конфигурацию задачи
    task_key = str(order)
    task_config = TASKS[task_key]

    # Решаем уравнение
    x, u_numeric, u_exact = solve_equation(order, task_config)
    
    if x is None:  # Произошла ошибка при решении
        return

    # Выводим информацию о решении
    print_solution_info(x, u_numeric, u_exact, order)
    
    # Строим график решения
    plot_solution(x, u_numeric, u_exact, order, "график_решения.png")

if __name__ == "__main__":
    main() 
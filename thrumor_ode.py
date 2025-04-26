import numpy as np  # Для численных вычислений и работы с массивами
import matplotlib.pyplot as plt  # Для построения графиков
import seaborn as sns  # Для улучшенной визуализации данных
import tkinter as tk  # Для создания графического интерфейса
from tkinter import ttk  # Улучшенные виджеты для tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # Для встраивания графиков в tkinter

# Экспериментальные данные (дни наблюдений)
exp_days = np.array([0, 3, 6, 9, 12, 15, 18])
exp_days_pbs = np.array([0, 3, 6, 9, 12, 15, 18])

# Экспериментальные значения объема опухоли для разных групп
exp_tumor_pbs = np.array([100, 166, 272, 414, 630, 892, 1550])  # Контрольная группа (PBS)
exp_tumor_dc_cik = np.array([100, 110, 130, 190, 260, 410, 710])  # Группа DC-CIK
exp_tumor_cik = np.array([100, 140, 210, 300, 470, 680, 1130])  # Группа CIK
exp_tumor_ag_dc_cik = np.array([100, 120, 130, 170, 240, 340, 480])  # Группа AG-DC-CIK

def validate_input(value):
    """Проверка ввода - разрешает только числа (включая десятичные и отрицательные)"""
    return value.replace(".", "", 1).replace("-", "", 1).isdigit() or value == ""

def tumor_immune_model(t, V, I, r, K, p, s, alpha, d):
    """
    Модель взаимодействия опухоли и иммунной системы.
    Система дифференциальных уравнений:
    dV/dt = r*V*(1 - V/K) - p*I*V  # Изменение объема опухоли
    dI/dt = s + alpha*I*V - d*I     # Изменение количества иммунных клеток
    """
    dV_dt = r * V * (1 - V / K) - p * I * V
    dI_dt = s + alpha * I * V - d * I
    return np.array([dV_dt, dI_dt])

def runge_kutta_4(f, V0, I0, t0, t_end, h, params):
    """
    Реализация метода Рунге-Кутта 4-го порядка для решения системы ОДУ.
    f - функция, описывающая систему уравнений
    V0, I0 - начальные условия
    t0, t_end - начальное и конечное время
    h - шаг интегрирования
    params - параметры модели
    """
    t_values = np.arange(t0, t_end + h, h)  # Массив временных точек
    V_values, I_values = np.zeros(len(t_values)), np.zeros(len(t_values))  # Массивы для решений
    V_values[0], I_values[0] = V0, I0  # Установка начальных условий

    # Итеративное вычисление решения
    for i in range(1, len(t_values)):
        t, V, I = t_values[i-1], V_values[i-1], I_values[i-1]
        # Вычисление коэффициентов метода Рунге-Кутта
        k1 = h * f(t, V, I, *params)
        k2 = h * f(t + h/2, V + k1[0]/2, I + k1[1]/2, *params)
        k3 = h * f(t + h/2, V + k2[0]/2, I + k2[1]/2, *params)
        k4 = h * f(t + h, V + k3[0], I + k3[1], *params)
        # Обновление значений
        V_values[i] = V + (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6
        I_values[i] = I + (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6

    return t_values, V_values, I_values

def show_mae_window(mae_data):
    """
    Создает окно с подробным анализом ошибок моделирования (MAE - Mean Absolute Error)
    mae_data - словарь с данными об ошибках для каждой группы
    """
    # Проверяем, существует ли уже окно и не было ли оно закрыто
    if hasattr(root, 'mae_window') and root.mae_window.winfo_exists():
        root.mae_window.lift()  # Поднимаем существующее окно на передний план
        root.mae_window.focus_set()  # Даем фокус окну
        return
    
    # Создаем новое окно
    root.mae_window = tk.Toplevel()
    root.mae_window.title("Анализ ошибок моделирования")
    root.mae_window.geometry("750x450")
    
    # Настройка стиля таблицы
    style = ttk.Style()
    style.configure("Treeview.Heading", font=('Arial', 10, 'bold'))
    
    # Создание таблицы для отображения данных
    tree = ttk.Treeview(root.mae_window, columns=("group", "mae_mm", "mae_percent", "mae_normalized", "max_value"), 
                       show="headings", height=10)
    
    # Настройка заголовков таблицы
    tree.heading("group", text="Группа", anchor=tk.CENTER)
    tree.heading("mae_mm", text="MAE (мм³)", anchor=tk.CENTER)
    tree.heading("mae_percent", text="MAE (%)", anchor=tk.CENTER)
    tree.heading("mae_normalized", text="MAE/K (%)", anchor=tk.CENTER)
    tree.heading("max_value", text="Макс. объем", anchor=tk.CENTER)
    
    # Настройка столбцов таблицы
    tree.column("group", width=150, anchor=tk.CENTER)
    tree.column("mae_mm", width=120, anchor=tk.CENTER)
    tree.column("mae_percent", width=120, anchor=tk.CENTER)
    tree.column("mae_normalized", width=120, anchor=tk.CENTER)
    tree.column("max_value", width=120, anchor=tk.CENTER)
    
    # Заполнение таблицы данными
    for group, data in mae_data.items():
        tree.insert("", tk.END, values=(
            group,
            f"{data['mae']:.2f}",
            f"{data['mae_percent']:.1f}%",
            f"{data['mae_normalized']:.1f}%",
            f"{data['max_value']:.0f}"
        ))
    
    # Цветовая маркировка строк в зависимости от величины ошибки
    for item in tree.get_children():
        mae_norm = float(tree.item(item)['values'][3].rstrip('%'))
        mae_percent = float(tree.item(item)['values'][2].rstrip('%'))
        
        if mae_norm < 5:
            tree.tag_configure('excellent', background='#d4edda')
            tree.item(item, tags=('excellent',))
        elif 5 <= mae_norm < 10:
            tree.tag_configure('good', background='#cce5ff')
            tree.item(item, tags=('good',))
        elif 10 <= mae_norm < 15:
            tree.tag_configure('acceptable', background='#fff3cd')
            tree.item(item, tags=('acceptable',))
        else:
            tree.tag_configure('poor', background='#f8d7da')
            tree.item(item, tags=('poor',))
        
        if mae_percent > 20:
            tree.tag_configure('high_error', background='#ffdddd')
            tree.item(item, tags=(tree.item(item)['tags'][0], 'high_error'))
    
    # Добавление скроллбара
    scrollbar = ttk.Scrollbar(root.mae_window, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    tree.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
    
    # Добавление пояснения по интерпретации ошибок
    info_frame = ttk.Frame(root.mae_window)
    info_frame.pack(fill=tk.X, padx=10, pady=5)
    
    ttk.Label(info_frame, text="Интерпретация MAE/K (%):", font=('Arial', 9, 'bold')).pack(anchor='w')
    ttk.Label(info_frame, text="<5% - Отличная точность", foreground='#155724', font=('Arial', 9)).pack(anchor='w')
    ttk.Label(info_frame, text="5-10% - Хорошая точность", foreground='#004085', font=('Arial', 9)).pack(anchor='w')
    ttk.Label(info_frame, text="10-15% - Приемлемая точность", foreground='#856404', font=('Arial', 9)).pack(anchor='w')
    ttk.Label(info_frame, text=">15% - Высокая ошибка", foreground='#721c24', font=('Arial', 9)).pack(anchor='w')
    
    ttk.Label(info_frame, text="Интерпретация MAE (%):", font=('Arial', 9, 'bold')).pack(anchor='w', pady=(5,0))
    ttk.Label(info_frame, text="0-5% - Очень высокая точность", font=('Arial', 9)).pack(anchor='w')
    ttk.Label(info_frame, text="5-10% - Хорошая точность", font=('Arial', 9)).pack(anchor='w')
    ttk.Label(info_frame, text="10-20% - Умеренная точность", font=('Arial', 9)).pack(anchor='w')
    ttk.Label(info_frame, text=">20% - Плохая точность", foreground='#721c24', font=('Arial', 9)).pack(anchor='w')
    
    # Кнопка закрытия окна
    ttk.Button(root.mae_window, text="Закрыть", 
              command=lambda: root.mae_window.destroy()).pack(pady=10)
    
    # Обработчик закрытия окна
    root.mae_window.protocol("WM_DELETE_WINDOW", lambda: root.mae_window.destroy())
    
    # Центрирование окна на экране
    root.mae_window.update_idletasks()
    width = root.mae_window.winfo_width()
    height = root.mae_window.winfo_height()
    x = (root.mae_window.winfo_screenwidth() // 2) - (width // 2)
    y = (root.mae_window.winfo_screenheight() // 2) - (height // 2)
    root.mae_window.geometry(f'{width}x{height}+{x}+{y}')

def update_plot():
    """Обновление графика на основе текущих параметров"""
    # Получение параметров из интерфейса
    params = [float(entry_params[param].get()) for param in default_values.keys()]
    V0 = float(entry_initial["V0"].get())
    I0 = float(entry_initial["I0"].get())
    t_end = float(entry_initial["t_end"].get())
    h = float(entry_initial["h"].get())
    
    # Параметры отображения графика
    time_step = float(display_params["time_step"].get())
    tumor_step = float(display_params["tumor_step"].get())
    
    # Решение системы уравнений
    t_values, V_values, I_values = runge_kutta_4(tumor_immune_model, V0, I0, 0, t_end, h, params)

    # Очистка графика перед обновлением
    ax.clear()
    
    # Построение графиков модельных данных
    if show_tumor.get():
        sns.lineplot(x=t_values, y=V_values, label="Модель: размер опухоли (мм³)", color='r', linewidth=2, ax=ax)
    
    if show_immune.get():
        sns.lineplot(x=t_values, y=I_values, label="Модель: иммунные клетки (I)", color='b', linewidth=2, ax=ax)

    # Настройка шкалы времени
    ax.set_xticks(np.arange(0, t_end + time_step, time_step))
    
    # Определение максимального значения для масштабирования оси Y
    y_max = max(V_values) if show_tumor.get() else 0
    if show_pbs.get(): y_max = max(y_max, max(exp_tumor_pbs))
    if show_dc_cik.get(): y_max = max(y_max, max(exp_tumor_dc_cik))
    if show_cik.get(): y_max = max(y_max, max(exp_tumor_cik))
    if show_ag_dc_cik.get(): y_max = max(y_max, max(exp_tumor_ag_dc_cik))
    
    # Настройка шкалы объема опухоли
    ax.set_yticks(np.arange(0, y_max * 1.1 + tumor_step, tumor_step))
    ax.set_ylim(0, y_max * 1.1 if y_max > 0 else 100)

    # Расчет ошибок моделирования (MAE)
    mae_data = {}
    K = float(entry_params["K"].get())  # Получаем K из параметров модели

    # Обработка данных для каждой группы
    if show_pbs.get():
        ax.scatter(exp_days_pbs, exp_tumor_pbs, label="PBS (контроль)", color='black', marker='o', s=40)
        model_pbs = np.interp(exp_days_pbs, t_values, V_values)  # Интерполяция модельных значений на экспериментальные точки
        pbs_errors = np.abs(model_pbs - exp_tumor_pbs)  # Абсолютные ошибки
        pbs_mae = np.mean(pbs_errors)  # Средняя абсолютная ошибка
        pbs_mae_percent = 100 * pbs_mae / np.mean(exp_tumor_pbs) if np.mean(exp_tumor_pbs) > 0 else 0  # Ошибка в процентах
        pbs_mae_normalized = 100 * pbs_mae / K  # Нормализованная ошибка
        mae_data["PBS"] = {
            "mae": pbs_mae,
            "mae_percent": pbs_mae_percent,
            "mae_normalized": pbs_mae_normalized,
            "max_value": max(exp_tumor_pbs)
        }

    if show_dc_cik.get():
        ax.scatter(exp_days, exp_tumor_dc_cik, label="DC-CIK", color='purple', marker='o', s=40)
        model_dc_cik = np.interp(exp_days, t_values, V_values)
        dc_cik_errors = np.abs(model_dc_cik - exp_tumor_dc_cik)
        dc_cik_mae = np.mean(dc_cik_errors)
        dc_cik_mae_percent = 100 * dc_cik_mae / np.mean(exp_tumor_dc_cik) if np.mean(exp_tumor_dc_cik) > 0 else 0
        dc_cik_mae_normalized = 100 * dc_cik_mae / K
        mae_data["DC-CIK"] = {
            "mae": dc_cik_mae,
            "mae_percent": dc_cik_mae_percent,
            "mae_normalized": dc_cik_mae_normalized,
            "max_value": max(exp_tumor_dc_cik)
        }

    if show_cik.get():
        ax.scatter(exp_days, exp_tumor_cik, label="CIK", color='green', marker='o', s=40)
        model_cik = np.interp(exp_days, t_values, V_values)
        cik_errors = np.abs(model_cik - exp_tumor_cik)
        cik_mae = np.mean(cik_errors)
        cik_mae_percent = 100 * cik_mae / np.mean(exp_tumor_cik) if np.mean(exp_tumor_cik) > 0 else 0
        cik_mae_normalized = 100 * cik_mae / K
        mae_data["CIK"] = {
            "mae": cik_mae,
            "mae_percent": cik_mae_percent,
            "mae_normalized": cik_mae_normalized,
            "max_value": max(exp_tumor_cik)
        }

    if show_ag_dc_cik.get():
        ax.scatter(exp_days, exp_tumor_ag_dc_cik, label="AG-DC-CIK", color='orange', marker='o', s=40)
        model_ag_dc_cik = np.interp(exp_days, t_values, V_values)
        ag_dc_cik_errors = np.abs(model_ag_dc_cik - exp_tumor_ag_dc_cik)
        ag_dc_cik_mae = np.mean(ag_dc_cik_errors)
        ag_dc_cik_mae_percent = 100 * ag_dc_cik_mae / np.mean(exp_tumor_ag_dc_cik) if np.mean(exp_tumor_ag_dc_cik) > 0 else 0
        ag_dc_cik_mae_normalized = 100 * ag_dc_cik_mae / K
        mae_data["AG-DC-CIK"] = {
            "mae": ag_dc_cik_mae,
            "mae_percent": ag_dc_cik_mae_percent,
            "mae_normalized": ag_dc_cik_mae_normalized,
            "max_value": max(exp_tumor_ag_dc_cik)
        }

    # Отображение линий выживаемости
    if show_pbs_survival.get():
        ax.axvline(x=22, color='black', linestyle=':', alpha=0.7, label='PBS выживаемость')
    if show_dc_cik_survival.get():
        ax.axvline(x=34, color='purple', linestyle=':', alpha=0.7, label='DC-CIK выживаемость')
    if show_cik_survival.get():
        ax.axvline(x=26, color='green', linestyle=':', alpha=0.7, label='CIK выживаемость')
    if show_ag_dc_cik_survival.get():
        ax.axvline(x=44, color='orange', linestyle=':', alpha=0.7, label='AG-DC-CIK выживаемость')

    # Настройка осей и заголовка
    ax.set_xlabel("Time (day)", fontsize=12)
    ax.set_ylabel("Tumor volume (mm³)", fontsize=12)
    ax.set_title("Динамика роста опухоли и иммунного ответа", fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # Обновление блока с информацией об ошибках
    if hasattr(root, 'mae_frame'):
        root.mae_frame.destroy()
    
    if mae_data:
        root.mae_frame = ttk.LabelFrame(frame_bottom, text="Ошибки моделирования")
        root.mae_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5, expand=False)
        
        mae_display = tk.Text(root.mae_frame, height=7, width=40, font=('Courier New', 10))
        mae_display.pack(padx=5, pady=5)
        
        mae_display.insert(tk.END, "Группа      MAE(мм³)   MAE(%)   MAE/K(%)\n", "header")
        mae_display.insert(tk.END, "----------------------------------------\n")
        
        for group in ['PBS', 'DC-CIK', 'CIK', 'AG-DC-CIK']:
            if group in mae_data:
                data = mae_data[group]
                line = f"{group:<12}{data['mae']:>8.2f}{data['mae_percent']:>8.1f}%{data['mae_normalized']:>8.1f}%\n"
                
                # Цветовое кодирование в зависимости от величины ошибки
                if data['mae_normalized'] < 5:
                    tag = "excellent"
                elif 5 <= data['mae_normalized'] < 10:
                    tag = "good"
                elif 10 <= data['mae_normalized'] < 15:
                    tag = "acceptable"
                else:
                    tag = "poor"
                
                if data['mae_percent'] > 20:
                    tag += " high_error"
                
                mae_display.insert(tk.END, line, tag)
        
        # Настройка стилей для разных уровней ошибок
        mae_display.tag_configure("header", font=('Courier New', 10, 'bold'))
        mae_display.tag_configure("excellent", foreground='#155724')  # темно-зеленый
        mae_display.tag_configure("good", foreground='#004085')  # темно-синий
        mae_display.tag_configure("acceptable", foreground='#856404')  # темно-желтый
        mae_display.tag_configure("poor", foreground='#721c24')  # темно-красный
        mae_display.tag_configure("high_error", background='#ffdddd')  # светло-красный фон
        mae_display.config(state=tk.DISABLED)
        
        # Кнопка для подробного анализа ошибок
        ttk.Button(root.mae_frame, text="Подробный анализ", 
                 command=lambda: show_mae_window(mae_data)).pack(pady=5)

    # Обновление блока с описанием параметров
    desc_text_widget.config(state=tk.NORMAL)
    desc_text_widget.delete(1.0, tk.END)
    
    desc_text_widget.insert(tk.END, "Параметры модели:\n", "bold")
    desc_text_widget.insert(tk.END, """r — скорость роста опухоли
K — предельный размер опухоли (мм³)
p — скорость уничтожения опухоли иммунитетом
s — скорость появления иммунных клеток
α — коэффициент активации иммунитета
d — скорость гибели иммунных клеток

""")
    
    desc_text_widget.insert(tk.END, "Начальные условия:\n", "bold")
    desc_text_widget.insert(tk.END, """V0 — начальный объем опухоли (мм³)
I0 — количество иммунных клеток
t_end — время моделирования (дни)
h — шаг интегрирования""")
    
    desc_text_widget.tag_configure("bold", font=("Arial", 12, "bold"))
    desc_text_widget.config(state=tk.DISABLED)

    # Перерисовка графика
    canvas.draw()

def reset_parameters():
    """Сброс параметров к значениям по умолчанию"""
    for param, value in default_values.items():
        entry_params[param].delete(0, tk.END)
        entry_params[param].insert(0, value)

    for param, value in initial_values.items():
        entry_initial[param].delete(0, tk.END)
        entry_initial[param].insert(0, value)
    
    display_params["time_step"].delete(0, tk.END)
    display_params["time_step"].insert(0, "3")
    
    display_params["tumor_step"].delete(0, tk.END)
    display_params["tumor_step"].insert(0, "500")

# Создание главного окна приложения
root = tk.Tk()
root.title("Моделирование роста опухоли")
root.state("zoomed")  # Открытие в полноэкранном режиме

# Верхняя часть интерфейса (график)
frame_top = ttk.Frame(root)
frame_top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Создание фигуры matplotlib
fig, ax = plt.subplots(figsize=(8, 4))
canvas = FigureCanvasTkAgg(fig, master=frame_top)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Панель инструментов для графика
toolbar_frame = ttk.Frame(frame_top)
toolbar_frame.pack(fill=tk.X, pady=(0, 5))

toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
toolbar.update()

# Нижняя часть интерфейса (параметры и управление)
frame_bottom = ttk.Frame(root)
frame_bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

# Левая часть нижнего фрейма (параметры + кнопки)
frame_left = ttk.Frame(frame_bottom)
frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Контейнер для параметров
frame_params_container = ttk.Frame(frame_left)
frame_params_container.pack(fill=tk.BOTH, expand=True)

# Фрейм для параметров модели
frame_params = ttk.LabelFrame(frame_params_container, text="Параметры модели")
frame_params.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5, expand=True)

# Создание полей ввода для параметров модели
entry_params = {}
default_values = {
    "r": "0.200",  # скорость роста опухоли
    "K": "2500",   # предельный размер опухоли
    "p": "0.01",   # скорость уничтожения опухоли иммунитетом
    "s": "5",      # скорость появления иммунных клеток
    "alpha": "0.001",  # коэффициент активации иммунитета
    "d": "0.1"     # скорость гибели иммунных клеток
}

for param, value in default_values.items():
    row = ttk.Frame(frame_params)
    row.pack(fill=tk.X, padx=5, pady=2)

    label = ttk.Label(row, text=f"{param}:", width=10)
    label.pack(side=tk.LEFT)

    entry = ttk.Entry(row, validate="key", 
                     validatecommand=(root.register(validate_input), "%P"),
                     width=15)
    entry.insert(0, value)
    entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    entry_params[param] = entry

# Фрейм для начальных условий и параметров отображения
frame_initial = ttk.LabelFrame(frame_params_container, text="🔢 Начальные условия и отображение")
frame_initial.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5, expand=True)

# Создание полей ввода для начальных условий
entry_initial = {}
initial_values = {
    "V0": "100",   # начальный объем опухоли
    "I0": "10",    # начальное количество иммунных клеток
    "t_end": "20", # время моделирования
    "h": "0.5"     # шаг интегрирования
}

for param, value in initial_values.items():
    row = ttk.Frame(frame_initial)
    row.pack(fill=tk.X, padx=5, pady=2)

    label = ttk.Label(row, text=f"{param}:", width=10)
    label.pack(side=tk.LEFT)

    entry = ttk.Entry(row, validate="key", 
                     validatecommand=(root.register(validate_input), "%P"),
                     width=15)
    entry.insert(0, value)
    entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    entry_initial[param] = entry

# Разделительная линия
ttk.Separator(frame_initial, orient='horizontal').pack(fill='x', padx=5, pady=5)

# Параметры отображения графика
display_params = {}

# Шаг шкалы времени
row_time_step = ttk.Frame(frame_initial)
row_time_step.pack(fill=tk.X, padx=5, pady=2)

ttk.Label(row_time_step, text="Шаг времени:", width=10).pack(side=tk.LEFT)
time_step_entry = ttk.Entry(row_time_step, validate="key", 
                          validatecommand=(root.register(validate_input), "%P"),
                          width=15)
time_step_entry.insert(0, "3")
time_step_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
display_params["time_step"] = time_step_entry

# Шаг шкалы объема опухоли
row_tumor_step = ttk.Frame(frame_initial)
row_tumor_step.pack(fill=tk.X, padx=5, pady=2)

ttk.Label(row_tumor_step, text="Шаг Tumor:", width=10).pack(side=tk.LEFT)
tumor_step_entry = ttk.Entry(row_tumor_step, validate="key", 
                           validatecommand=(root.register(validate_input), "%P"),
                           width=15)
tumor_step_entry.insert(0, "500")
tumor_step_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
display_params["tumor_step"] = tumor_step_entry

# Фрейм для кнопок управления
btn_frame = ttk.Frame(frame_left)
btn_frame.pack(fill=tk.X, pady=(0, 5), padx=5)

btn_apply = ttk.Button(btn_frame, text="ПРИМЕНИТЬ", command=update_plot)
btn_apply.pack(side=tk.LEFT, expand=True, padx=5)

btn_reset = ttk.Button(btn_frame, text="СБРОС", command=reset_parameters)
btn_reset.pack(side=tk.RIGHT, expand=True, padx=5)

# Правая часть нижнего фрейма (описание параметров)
frame_description = ttk.LabelFrame(frame_bottom, text="Описание параметров")
frame_description.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=5)

# Текстовое поле с описанием параметров
desc_text_widget = tk.Text(frame_description, wrap=tk.WORD, font=("Arial", 12), padx=10, pady=10, height=15, width=40)
desc_text_widget.pack()

desc_text_widget.insert(tk.END, "Параметры модели:\n", "bold")
desc_text_widget.insert(tk.END, """r — скорость роста опухоли
K — предельный размер опухоли (мм³)
p — скорость уничтожения опухоли иммунитетом
s — скорость появления иммунных клеток
α — коэффициент активации иммунитета
d — скорость гибели иммунных клеток

""")

desc_text_widget.insert(tk.END, "Начальные условия:\n", "bold")
desc_text_widget.insert(tk.END, """V0 — начальный объем опухоли (мм³)
I0 — количество иммунных клеток
t_end — время моделирования (дни)
h — шаг интегрирования""")

desc_text_widget.tag_configure("bold", font=("Arial", 12, "bold"))
desc_text_widget.config(state=tk.DISABLED)

# Фрейм для чекбоксов модельных данных
frame_checkboxes_model = ttk.LabelFrame(frame_bottom, text="Отображение модельных данных")
frame_checkboxes_model.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)

# Переменные и чекбоксы для управления отображением модельных данных
show_tumor = tk.BooleanVar(value=True)
show_immune = tk.BooleanVar(value=True)

chk_tumor = ttk.Checkbutton(frame_checkboxes_model, text="Показывать опухоль", variable=show_tumor)
chk_tumor.pack(anchor="w")

chk_immune = ttk.Checkbutton(frame_checkboxes_model, text="Показывать иммунные клетки", variable=show_immune)
chk_immune.pack(anchor="w")

# Фрейм для чекбоксов экспериментальных данных
frame_checkboxes_exp = ttk.LabelFrame(frame_bottom, text="Отображение экспериментальных данных")
frame_checkboxes_exp.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)

# Переменные для управления отображением экспериментальных данных
show_pbs = tk.BooleanVar(value=True)
show_dc_cik = tk.BooleanVar(value=True)
show_cik = tk.BooleanVar(value=True)
show_ag_dc_cik = tk.BooleanVar(value=True)
show_pbs_survival = tk.BooleanVar(value=False)
show_dc_cik_survival = tk.BooleanVar(value=False)
show_cik_survival = tk.BooleanVar(value=False)
show_ag_dc_cik_survival = tk.BooleanVar(value=False)

def create_group_frame(parent, text, var_data, var_survival, survival_text):
    """Создает фрейм с чекбоксами для группы данных"""
    frame = ttk.Frame(parent)
    frame.pack(fill=tk.X, pady=2)
    ttk.Checkbutton(frame, text=text, variable=var_data).pack(side=tk.LEFT, padx=5)
    ttk.Checkbutton(frame, text=survival_text, variable=var_survival).pack(side=tk.LEFT)
    return frame

# Создание чекбоксов для каждой группы данных
create_group_frame(frame_checkboxes_exp, "PBS (контроль)", show_pbs, show_pbs_survival, "Выживаемость (22 дн)")
create_group_frame(frame_checkboxes_exp, "DC-CIK", show_dc_cik, show_dc_cik_survival, "Выживаемость (34 дн)")
create_group_frame(frame_checkboxes_exp, "CIK", show_cik, show_cik_survival, "Выживаемость (26 дн)")
create_group_frame(frame_checkboxes_exp, "AG-DC-CIK", show_ag_dc_cik, show_ag_dc_cik_survival, "Выживаемость (44 дн)")

# Первоначальное обновление графика
update_plot()
# Запуск главного цикла приложения
root.mainloop()

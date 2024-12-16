import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats  # Импортируем SciPy для статистических тестов

# Загрузка данных из файла
file_path = 'Bitcoin.csv'  # Укажите корректный путь к файлу

# Попытка загрузить данные с запятой как разделитель
data = pd.read_csv(file_path, sep=',', skiprows=1)  # Пробуем разделитель ",", пропускаем первую строку с заголовком

# Проверка структуры данных
print("Первые строки файла:")
print(data.head())
print("\nСтруктура данных:")
print(data.info())  # Вывод информации о DataFrame (покажет количество столбцов и их типы)

# Переименовываем столбцы для удобства
data.columns = ['Week', 'Interest']

# Преобразуем столбец "Week" в формат даты
try:
    data['Week'] = pd.to_datetime(data['Week'], errors='raise')
except Exception as e:
    print(f"Ошибка при преобразовании столбца 'Week' в дату: {e}")
    data['Week'] = pd.to_datetime(data['Week'], errors='coerce')  # Пропуск некорректных дат

# Преобразуем столбец "Interest" в числовой тип
data['Interest'] = pd.to_numeric(data['Interest'], errors='coerce')

# Устанавливаем индекс по столбцу "Week"
data.set_index('Week', inplace=True)

# Проверка структуры данных после обработки
print("\nДанные после обработки:")
print(data.head())

# Применение теста Шапиро-Уилка для проверки нормальности данных
stat, p_value = stats.shapiro(data['Interest'].dropna())  # Применяем тест к столбцу 'Interest'
print("\nРезультат теста Шапиро-Уилка для проверки нормальности:")
print(f"Статистика: {stat:.4f}, P-значение: {p_value:.4f}")
if p_value > 0.05:
    print("Данные могут следовать нормальному распределению (не отклоняем гипотезу H0).")
else:
    print("Данные не следуют нормальному распределению (отклоняем гипотезу H0).")

# Визуализация временного ряда (таймлайн) - График 1
plt.figure(figsize=(14, 6))
sns.lineplot(data=data, x=data.index, y='Interest', color='blue', marker='o', label='Interest over time')
plt.title('Интерес к запросу в России (Google Trends)', fontsize=16)
plt.xlabel('Неделя', fontsize=14)
plt.ylabel('Популярность', fontsize=14)
plt.xticks(rotation=45)  # Поворот меток по оси X для лучшей читаемости
plt.legend()
plt.grid(True)
plt.tight_layout()  # Чтобы график не выходил за пределы

# Проверим, что данных достаточно для декомпозиции
# Если данные слишком короткие, декомпозиция может не сработать
if len(data) >= 52:
    print("\nДанные достаточно длинные для декомпозиции. Выполняем декомпозицию...")
    # Декомпозиция временного ряда
    try:
        result = seasonal_decompose(data['Interest'], model='additive', period=52)
        # Открываем новый график
        plt.figure(figsize=(14, 8))
        result.plot()
        plt.suptitle('Декомпозиция временного ряда', fontsize=16)
        plt.tight_layout()
    except Exception as e:
        print(f"Ошибка при декомпозиции: {e}")
else:
    print("Данных недостаточно для декомпозиции. Требуется минимум 52 недели.")

# Построение автокорреляционной и частичной автокорреляционной функции - График 3
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plot_acf(data['Interest'], lags=52, ax=plt.gca(), title='ACF: Автокорреляционная функция')
plt.subplot(2, 1, 2)
plot_pacf(data['Interest'], lags=52, ax=plt.gca(), title='PACF: Частичная автокорреляционная функция')
plt.tight_layout()

# Показываем все графики сразу
plt.show()

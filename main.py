import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QTableWidget, 
                             QTableWidgetItem, QVBoxLayout, QWidget, QPushButton, 
                             QFileDialog, QComboBox, QLabel, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

class CorrelationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализ корреляций")
        self.setGeometry(100, 100, 1000, 800)
        
        self.data = None
        self.var_types = {}
        
        self.initUI()
    
    def initUI(self):
        # Создаем вкладки
        self.tabs = QTabWidget()
        
        # Вкладка 1: Загрузка данных и выбор типов
        self.tab1 = QWidget()
        self.tab1_layout = QVBoxLayout()
        self.table_data = QTableWidget()
        self.btn_load = QPushButton("Загрузить данные (CSV)")
        self.btn_load.clicked.connect(self.load_data)
        self.btn_process = QPushButton("Рассчитать корреляции")
        self.btn_process.clicked.connect(self.process_data)
        self.label_info = QLabel("Укажите тип каждой переменной:")
        self.tab1_layout.addWidget(self.btn_load)
        self.tab1_layout.addWidget(self.label_info)
        self.tab1_layout.addWidget(self.table_data)
        self.tab1_layout.addWidget(self.btn_process)
        self.tab1.setLayout(self.tab1_layout)
        
        # Вкладка 2: Парные корреляции
        self.tab2 = QWidget()
        self.tab2_layout = QVBoxLayout()
        self.table_pairwise = QTableWidget()
        self.label_pairwise = QLabel("Парные корреляции (значимые p < 0.05 выделены красным):")
        self.tab2_layout.addWidget(self.label_pairwise)
        self.tab2_layout.addWidget(self.table_pairwise)
        self.tab2.setLayout(self.tab2_layout)
        
        # Вкладка 3: Частные корреляции
        self.tab3 = QWidget()
        self.tab3_layout = QVBoxLayout()
        self.table_partial = QTableWidget()
        self.label_partial = QLabel("Частные корреляции:")
        self.tab3_layout.addWidget(self.label_partial)
        self.tab3_layout.addWidget(self.table_partial)
        self.tab3.setLayout(self.tab3_layout)
        
        # Вкладка 4: Множественные корреляции
        self.tab4 = QWidget()
        self.tab4_layout = QVBoxLayout()
        self.table_multiple = QTableWidget()
        self.label_multiple = QLabel("Множественные корреляции (R²):")
        self.tab4_layout.addWidget(self.label_multiple)
        self.tab4_layout.addWidget(self.table_multiple)
        self.tab4.setLayout(self.tab4_layout)
        
        # Добавляем вкладки
        self.tabs.addTab(self.tab1, "Данные")
        self.tabs.addTab(self.tab2, "Парные корреляции")
        self.tabs.addTab(self.tab3, "Частные корреляции")
        self.tabs.addTab(self.tab4, "Множественные корреляции")
        
        self.setCentralWidget(self.tabs)
    
    def load_data(self):
        """Загрузка CSV-файла с данными."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Открыть файл", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                self.update_table_data()
                QMessageBox.information(self, "Успех", "Данные успешно загружены!")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл:\n{str(e)}")
    
    def update_table_data(self):
        """Обновление таблицы с данными и выбором типов переменных."""
        if self.data is not None:
            self.table_data.setRowCount(len(self.data.columns))
            self.table_data.setColumnCount(2)
            self.table_data.setHorizontalHeaderLabels(["Переменная", "Тип"])
            self.table_data.setColumnWidth(0, 200)
            self.table_data.setColumnWidth(1, 150)
            
            for i, col in enumerate(self.data.columns):
                self.table_data.setItem(i, 0, QTableWidgetItem(col))
                
                combo = QComboBox()
                combo.addItems(["Количественный", "Качественный", "Порядковый"])
                combo.setCurrentIndex(0)
                combo.currentTextChanged.connect(lambda text, col=col: self.update_var_type(col, text))
                self.table_data.setCellWidget(i, 1, combo)
                
                # Сохраняем тип переменной по умолчанию
                self.var_types[col] = "Количественный"
    
    def update_var_type(self, col, var_type):
        """Обновление типа переменной."""
        self.var_types[col] = var_type
    
    def process_data(self):
        """Основной метод обработки данных и расчета корреляций."""
        if self.data is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите данные!")
            return
        
        # Проверка на пропущенные значения
        if self.data.isnull().any().any():
            reply = QMessageBox.question(self, "Пропущенные значения", 
                                      "В данных есть пропуски. Хотите удалить строки с пропусками?", 
                                      QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.data = self.data.dropna()
        
        # Рассчитываем все виды корреляций
        self.calculate_pairwise()
        self.calculate_partial()
        self.calculate_multiple()
        
        QMessageBox.information(self, "Готово", "Расчет корреляций завершен!")
    
    def is_normal(self, data, threshold=0.05):
        """Проверка на нормальность распределения с помощью теста Шапиро-Уилка."""
        if len(data) > 5000:  # Тест Шапиро работает только для n < 5000
            sample = data.sample(5000)
        else:
            sample = data
        stat, p = stats.shapiro(sample)
        return p > threshold
    
    def calculate_correlation(self, var1, var2):
        """
        Вычисляет корреляцию между двумя переменными с учетом их типов.
        Возвращает: (коэффициент, p-value, название метода)
        """
        type1 = self.var_types[var1]
        type2 = self.var_types[var2]
        x = self.data[var1].dropna()
        y = self.data[var2].dropna()

        # Количественная vs Количественная
        if type1 == "Количественный" and type2 == "Количественный":
            if self.is_normal(x) and self.is_normal(y):
                corr, p_value = stats.pearsonr(x, y)
                method = "Пирсон"
            else:
                corr, p_value = stats.spearmanr(x, y)
                method = "Спирмен"
        
        # Количественная vs Бинарная
        elif ((type1 == "Количественный" and type2 == "Качественный" and len(y.unique()) == 2) or
              (type2 == "Количественный" and type1 == "Качественный" and len(x.unique()) == 2)):
            if type1 == "Качественный":
                x, y = y, x  # Чтобы x была количественной, y — бинарной
            corr, p_value = stats.pointbiserialr(x, y), 0.0
            method = "Точечная бисериальная"
        
        # Количественная vs Категориальная (≥3 уровней)
        elif ((type1 == "Количественный" and type2 == "Качественный" and len(y.unique()) > 2) or
              (type2 == "Количественный" and type1 == "Качественный" and len(x.unique()) > 2)):
            if type1 == "Качественный":
                x, y = y, x
            # ANOVA F-value как мера связи
            groups = [x[y == cat] for cat in y.unique() if len(x[y == cat]) > 1]
            if len(groups) < 2:
                return np.nan, 1.0, "Невозможно вычислить"
            f_stat, p_value = stats.f_oneway(*groups)
            eta_squared = f_stat / (f_stat + (len(x) - len(groups)))
            corr = np.sqrt(eta_squared)
            method = "ANOVA (η²)"
        
        # Категориальная vs Категориальная
        elif type1 == "Качественный" and type2 == "Качественный":
            contingency = pd.crosstab(x, y).values
            if contingency.size == 0:
                return np.nan, 1.0, "Невозможно вычислить"
            chi2 = stats.chi2_contingency(contingency)[0]
            n = contingency.sum()
            phi2 = chi2 / n
            r, k = contingency.shape
            corr = np.sqrt(phi2 / min((k-1), (r-1)))
            p_value = stats.chi2_contingency(contingency)[1]
            method = "Крамера"
        
        # Порядковая vs Порядковая
        elif type1 == "Порядковый" and type2 == "Порядковый":
            corr, p_value = stats.kendalltau(x, y)
            method = "Кендалла"
        
        # Порядковая vs Количественная/Качественная
        else:
            corr, p_value = stats.spearmanr(x, y)
            method = "Спирмен"
        
        return corr, p_value, method
    
    def calculate_pairwise(self):
        """Расчет и отображение парных корреляций."""
        if self.data is None:
            return
        
        cols = self.data.columns
        n = len(cols)
        self.table_pairwise.setRowCount(n)
        self.table_pairwise.setColumnCount(n)
        self.table_pairwise.setHorizontalHeaderLabels(cols)
        self.table_pairwise.setVerticalHeaderLabels(cols)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    item = QTableWidgetItem("1.0 (—)")
                    item.setBackground(QColor(240, 240, 240))
                    self.table_pairwise.setItem(i, j, item)
                else:
                    var1 = cols[i]
                    var2 = cols[j]
                    try:
                        corr, p_value, method = self.calculate_correlation(var1, var2)
                        if np.isnan(corr):
                            item = QTableWidgetItem("—")
                        else:
                            item = QTableWidgetItem(f"{corr:.3f} ({method})")
                            if p_value < 0.05:  # Выделяем значимые
                                item.setBackground(QColor(255, 200, 200))
                    except Exception as e:
                        item = QTableWidgetItem("Ошибка")
                        print(f"Ошибка при расчете {var1} и {var2}: {str(e)}")
                    
                    self.table_pairwise.setItem(i, j, item)
        
        self.table_pairwise.resizeColumnsToContents()
    
    def calculate_partial(self):
        """Расчет частных корреляций (заглушка)."""
        self.table_partial.setRowCount(1)
        self.table_partial.setColumnCount(1)
        self.table_partial.setItem(0, 0, QTableWidgetItem("Реализация в разработке"))
    
    def calculate_multiple(self):
        """Расчет множественных корреляций (заглушка)."""
        self.table_multiple.setRowCount(1)
        self.table_multiple.setColumnCount(1)
        self.table_multiple.setItem(0, 0, QTableWidgetItem("Реализация в разработке"))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CorrelationApp()
    window.show()
    sys.exit(app.exec_())

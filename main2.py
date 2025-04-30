import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QTableWidget, 
                             QTableWidgetItem, QVBoxLayout, QWidget, QPushButton, 
                             QFileDialog, QComboBox, QLabel, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
import scipy.stats as stats

class CorrelationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализ корреляций (матричные методы)")
        self.setGeometry(100, 100, 1200, 900)
        
        self.data = None
        self.var_types = {}
        
        self.initUI()
    
    def initUI(self):
        self.tabs = QTabWidget()
        
        # Вкладка 1: Загрузка данных
        self.tab1 = QWidget()
        self.setup_tab1()
        
        # Вкладка 2: Парные корреляции
        self.tab2 = QWidget()
        self.setup_tab2()
        
        # Вкладка 3: Частные корреляции
        self.tab3 = QWidget()
        self.setup_tab3()
        
        # Вкладка 4: Множественные корреляции
        self.tab4 = QWidget()
        self.setup_tab4()
        
        self.tabs.addTab(self.tab1, "Данные")
        self.tabs.addTab(self.tab2, "Парные корреляции")
        self.tabs.addTab(self.tab3, "Частные корреляции")
        self.tabs.addTab(self.tab4, "Множественные R²")
        self.setCentralWidget(self.tabs)
    
    def setup_tab1(self):
        layout = QVBoxLayout()
        self.table_data = QTableWidget()
        self.btn_load = QPushButton("Загрузить данные (CSV)")
        self.btn_load.clicked.connect(self.load_data)
        self.btn_process = QPushButton("Рассчитать корреляции")
        self.btn_process.clicked.connect(self.process_data)
        
        layout.addWidget(self.btn_load)
        layout.addWidget(QLabel("Укажите тип каждой переменной:"))
        layout.addWidget(self.table_data)
        layout.addWidget(self.btn_process)
        self.tab1.setLayout(layout)
    
    def setup_tab2(self):
        layout = QVBoxLayout()
        self.table_pairwise = QTableWidget()
        layout.addWidget(QLabel("Парные корреляции (p < 0.05 выделены):"))
        layout.addWidget(self.table_pairwise)
        self.tab2.setLayout(layout)
    
    def setup_tab3(self):
        layout = QVBoxLayout()
        self.table_partial = QTableWidget()
        layout.addWidget(QLabel("Частные корреляции (матричный метод):"))
        layout.addWidget(self.table_partial)
        self.tab3.setLayout(layout)
    
    def setup_tab4(self):
        layout = QVBoxLayout()
        self.table_multiple = QTableWidget()
        layout.addWidget(QLabel("Множественные R² (значимые p < 0.05):"))
        layout.addWidget(self.table_multiple)
        self.tab4.setLayout(layout)
    
    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Открыть файл", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                self.update_table_data()
                QMessageBox.information(self, "Успех", "Данные загружены!")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки:\n{str(e)}")
    
    def update_table_data(self):
        if self.data is not None:
            self.table_data.setRowCount(len(self.data.columns))
            self.table_data.setColumnCount(2)
            self.table_data.setHorizontalHeaderLabels(["Переменная", "Тип"])
            self.table_data.setColumnWidth(0, 250)
            self.table_data.setColumnWidth(1, 150)
            
            for i, col in enumerate(self.data.columns):
                self.table_data.setItem(i, 0, QTableWidgetItem(col))
                
                combo = QComboBox()
                combo.addItems(["Количественный", "Качественный", "Порядковый"])
                combo.setCurrentIndex(0)
                combo.currentTextChanged.connect(lambda text, col=col: self.update_var_type(col, text))
                self.table_data.setCellWidget(i, 1, combo)
                self.var_types[col] = "Количественный"
    
    def update_var_type(self, col, var_type):
        self.var_types[col] = var_type
    
    def process_data(self):
        if self.data is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите данные!")
            return
        
        if self.data.isnull().any().any():
            reply = QMessageBox.question(self, "Пропуски", 
                                       "Удалить строки с пропусками?", 
                                       QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.data = self.data.dropna()
        
        self.calculate_pairwise()
        self.calculate_partial()
        self.calculate_multiple()
        QMessageBox.information(self, "Готово", "Расчеты завершены!")
    
    def is_normal(self, data, threshold=0.05):
        if len(data) > 5000:
            data = data.sample(5000)
        stat, p = stats.shapiro(data)
        return p > threshold
    
    def calculate_correlation(self, var1, var2):
        type1 = self.var_types[var1]
        type2 = self.var_types[var2]
        x = self.data[var1].dropna()
        y = self.data[var2].dropna()
        n = len(x)

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
                x, y = y, x
            corr = stats.pointbiserialr(x, y)[0]
            p_value = self.calculate_significance(corr, n)
            method = "Точечная бисериальная"
        
        # Количественная vs Категориальная
        elif ((type1 == "Количественный" and type2 == "Качественный" and len(y.unique()) > 2) or
              (type2 == "Количественный" and type1 == "Качественный" and len(x.unique()) > 2)):
            if type1 == "Качественный":
                x, y = y, x
            groups = [x[y == cat] for cat in y.unique() if len(x[y == cat]) > 1]
            if len(groups) < 2:
                return np.nan, 1.0, "Невозможно вычислить"
            f_stat, p_value = stats.f_oneway(*groups)
            eta_squared = f_stat / (f_stat + (len(x) - len(groups)))
            corr = np.sqrt(eta_squared)
            p_value = self.calculate_significance(corr, n)
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
        
        else:
            corr, p_value = stats.spearmanr(x, y)
            method = "Спирмен"
        
        return corr, p_value, method
    
    def calculate_significance(self, corr, n, method='pearson'):
        """Проверка значимости корреляции через t-тест Стьюдента"""
        if np.isnan(corr):
            return 1.0
        
        if method == 'pearson':
            df = n - 2
            t_value = corr * np.sqrt(df / (1 - corr**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_value), df))
        else:
            z = np.arctanh(corr)
            se = 1 / np.sqrt(n - 3)
            p_value = 2 * (1 - stats.norm.cdf(abs(z / se)))
        
        return p_value
    
    def calculate_pairwise(self):
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
                else:
                    var1, var2 = cols[i], cols[j]
                    try:
                        corr, p_value, method = self.calculate_correlation(var1, var2)
                        if np.isnan(corr):
                            item = QTableWidgetItem("—")
                        else:
                            item = QTableWidgetItem(f"{corr:.3f} ({method})")
                            if p_value < 0.05:
                                item.setBackground(QColor(255, 200, 200))
                    except Exception as e:
                        item = QTableWidgetItem("Ошибка")
                        print(f"Ошибка при расчете {var1} и {var2}: {str(e)}")
                
                self.table_pairwise.setItem(i, j, item)
        
        self.table_pairwise.resizeColumnsToContents()
    
    def calculate_partial(self):
        """Точный расчет частных корреляций через обратную ковариационную матрицу"""
        if self.data is None:
            return

        quant_cols = [col for col in self.data.columns if self.var_types[col] == "Количественный"]
        if len(quant_cols) < 2:
            self.table_partial.setRowCount(1)
            self.table_partial.setColumnCount(1)
            self.table_partial.setItem(0, 0, QTableWidgetItem("Требуется ≥2 количественных переменных"))
            return

        # Матричный расчет
        X = self.data[quant_cols].values
        X_centered = X - X.mean(axis=0)
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        try:
            precision_matrix = np.linalg.pinv(cov_matrix)
            diag = np.diag(precision_matrix)
            partial_corrs = -precision_matrix / np.sqrt(np.outer(diag, diag))
            np.fill_diagonal(partial_corrs, 1.0)
        except np.linalg.LinAlgError:
            QMessageBox.warning(self, "Ошибка", "Не удалось вычислить частные корреляции")
            return

        # Отображение результатов
        n = len(quant_cols)
        self.table_partial.setRowCount(n)
        self.table_partial.setColumnCount(n)
        self.table_partial.setHorizontalHeaderLabels(quant_cols)
        self.table_partial.setVerticalHeaderLabels(quant_cols)

        for i in range(n):
            for j in range(n):
                corr = partial_corrs[i, j]
                item = QTableWidgetItem(f"{corr:.3f}")
                
                # Проверка значимости
                df = len(self.data) - len(quant_cols)
                t_val = corr * np.sqrt(df / (1 - corr**2))
                p_val = 2 * (1 - stats.t.cdf(abs(t_val), df))
                
                if p_val < 0.05:
                    item.setBackground(QColor(255, 200, 200))
                
                self.table_partial.setItem(i, j, item)

        self.table_partial.resizeColumnsToContents()
    
    def calculate_multiple(self):
        """Расчет множественных R² через проекционные матрицы"""
        if self.data is None:
            return

        quant_cols = [col for col in self.data.columns if self.var_types[col] == "Количественный"]
        if len(quant_cols) < 2:
            self.table_multiple.setRowCount(1)
            self.table_multiple.setColumnCount(1)
            self.table_multiple.setItem(0, 0, QTableWidgetItem("Требуется ≥2 количественных переменных"))
            return

        X = self.data[quant_cols].values
        X_centered = X - X.mean(axis=0)
        n_vars = X.shape[1]

        # Матричный расчет R²
        r_squared = np.zeros(n_vars)
        p_values = np.zeros(n_vars)

        for i in range(n_vars):
            y = X_centered[:, i]
            X_other = np.delete(X_centered, i, axis=1)
            
            # Проекционная матрица
            H = X_other @ np.linalg.pinv(X_other.T @ X_other) @ X_other.T
            y_pred = H @ y
            ss_total = np.sum(y**2)
            ss_res = np.sum((y - y_pred)**2)
            r_squared[i] = 1 - (ss_res / ss_total)
            
            # F-тест
            n = X.shape[0]
            p = X_other.shape[1]
            f_val = (r_squared[i] / p) / ((1 - r_squared[i]) / (n - p - 1))
            p_values[i] = 1 - stats.f.cdf(f_val, p, n - p - 1)

        # Отображение
        self.table_multiple.setRowCount(n_vars)
        self.table_multiple.setColumnCount(2)
        self.table_multiple.setHorizontalHeaderLabels(["Переменная", "R²"])
        self.table_multiple.setColumnWidth(0, 200)
        self.table_multiple.setColumnWidth(1, 100)
        
        for i in range(n_vars):
            item_var = QTableWidgetItem(quant_cols[i])
            item_r2 = QTableWidgetItem(f"{r_squared[i]:.3f}")
            
            if p_values[i] < 0.05:
                item_r2.setBackground(QColor(255, 200, 200))
            
            self.table_multiple.setItem(i, 0, item_var)
            self.table_multiple.setItem(i, 1, item_r2)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CorrelationApp()
    window.show()
    sys.exit(app.exec_())

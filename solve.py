import scipy.sparse.linalg
import numpy as np
from scipy import special
from scipy.optimize import minimize
from tabulate import tabulate
tabulate.WIDE_CHARS_MODE = True
import matplotlib.pyplot as plt
import os
from numpy.polynomial import Polynomial as pm

flag_for_printing_shapes = False

def basis_legendre(degree):
        basis = [pm([1])]
        for i in range(degree):
            if i == 0:
                basis.append(pm([1, -1]))
                continue
            basis.append(pm([2*i + 1, -1])*basis[-1] - i * i * basis[-2])
        return basis[-1].coef[::-1]

def basis_sh_legendre(degree):
        basis = [pm([1])]
        for i in range(degree):
            if i == 0:
                basis.append(pm([-1, 2]))
                continue
            basis.append((pm([-2*i - 1, 4*i + 2])*basis[-1] - i * basis[-2]) / (i + 1))
        return basis[-1].coef[::-1]

def basis_laguerre(degree):
        basis = [np.poly1d([1]), np.poly1d([-1,1])]
        for i in range(degree):
                basis.append(np.poly1d([-1,2*i+1])*basis[-1] -i*i*basis[-2])
        del basis[0]
        return basis[-1].coef[::-1]

class Polynomial:
    def __init__(self, polynomial_type):
        # Обираємо поліном для розрахунку коефіцієнтів згідно з налаштуваннями.
        if polynomial_type == "Чебишева":
            self._get_coefficients = lambda n: np.array(special.chebyt(n).coefficients)
        elif polynomial_type == "Лежандра":
            self._get_coefficients = lambda n: np.array(special.legendre(n).coefficients)
        elif polynomial_type == "Лагерра":
            self._get_coefficients = lambda n: np.array(special.laguerre(n).coefficients)
        elif polynomial_type == "Ерміта":
            self._get_coefficients = lambda n: np.array(special.hermite(n).coefficients)
        else:
            exit("Polynomial type is not defined!")

    def get_polynomial_sum_coefficients(self, degree, polynomial_multiplier=None):
        """Повертає коефіцієнти полиномів степенів від 0 до degree включно."""
        if polynomial_multiplier is None:
            polynomial_multiplier = np.ones(degree+1)
        polynomial_sum_coefficients = np.zeros(degree+1)
        for deg in np.arange(degree+1):
            # Обраховуємо коефіцієнти полиному потрібного степеню.
            polynomial = self._get_coefficients(deg) * polynomial_multiplier[deg]
            # Сумуємо коефіцієнты полінома потрібного степеню з коефіцієнтами попередніх поліномів.
            for position in np.arange(1, deg+2):
                polynomial_sum_coefficients[-position] += polynomial[-position]
        return np.flipud(polynomial_sum_coefficients)

class Output:

    @staticmethod
    def _show_lambda(lambda_matrix, dim_y):
        header = ['Y1', 'Y2', 'Y3', 'Y4']
        out = f"Матриця λ для Y:\n"
        out += tabulate(lambda_matrix.T, header[:dim_y], floatfmt=('.4f',)*lambda_matrix.shape[0],
                        tablefmt="presto") #presto
        out += "\n\n"

        return out

    @staticmethod
    def _show_psi(psi, dim_y):
        out = ""
        for i in np.arange(dim_y):
            out += f"     Матриця ψ для Y{i+1}:\n"
            sub_psi = np.vstack(psi[i])
            out += tabulate(sub_psi.T, floatfmt=('.4f',)*sub_psi.shape[0], tablefmt="presto")
            out += "\n"
        return out

    @staticmethod
    def _show_a(a, dim_y):
        header = ['Y1', 'Y2', 'Y3', 'Y4']
        out = f"Матриця a для Y:\n"
        out += tabulate(a.T, header[:dim_y], floatfmt=('.4f',)*a.shape[0], tablefmt="presto")
        out += "\n\n"
        return out

    @staticmethod
    def _show_phi(phi, dim_y):
        out = ""
        for i in np.arange(dim_y):
            out += f"Матриця Ф для Y{i + 1}:\n"
            sub_phi = np.vstack(phi[i])
            out += tabulate(sub_phi.T, floatfmt=('.4f',)*phi.shape[0], tablefmt="presto")
            out += "\n"
        return out

    @staticmethod
    def _show_c(c, dim_y):
        header = ['Y1', 'Y2', 'Y3', 'Y4']
        out = f"Матриця c для Y:\n"
        out += tabulate(c.T, header[:dim_y], floatfmt=('.4f',)*c.shape[0], tablefmt="presto")
        out += "\n\n"
        return out

    @staticmethod
    def _show_dependence_by_phi(c, dim_y):
        """Вивід функціональної залежності через функції Ф."""
        out = "\nФункціональна залежність від Ф:\n\n"
        for i in np.arange(dim_y):
            out += f"Ф{i+1}(X1,X2,X3) = {c[i][0]:.4f}Ф{i+1}1(X1) + {c[i][1]:.4f}Ф{i+1}2(X2) + {c[i][2]:.4f}Ф{i+1}3(X3)\n"
        return out

    @staticmethod
    def __get_coefficient(a, c, lambda_matrix, dim, degrees, i, j, p):
        """Повертає коефіцієнт при T{p}(x{i}{j})."""
        _dim = dim[:i - 1]
        _deg = degrees[:i - 1]  # + 1
        coefficient = c[i - 1]
        coefficient *= a[sum(_dim) + j - 1]
        coefficient *= lambda_matrix[np.sum(np.multiply(_dim, _deg)) + p]
        return coefficient

    @classmethod
    def _show_dependence_by_polynomials(cls, a, c, lambda_matrix, dim, degrees, polynomial_type, dim_y):
        """Вивід функціональної залежності через поліноми T."""

        def _get_term(_i, _j, _p, __i):
            """Повертає коефіцієнт при T{p}(x{i}{j}) у готовій для запису формі."""
            coefficient = cls.__get_coefficient(a[__i], c[__i], lambda_matrix[__i], dim, degrees, i, j, p)
            if _i == 1 and _j == 1 and _p == 0:
                sign = ""
            elif coefficient >= 0:
                sign = " + "
            else:
                sign = " - "
            coefficient = abs(coefficient)
            return f"{sign}{coefficient:.4f}*T{_p}(x{_i}{_j})"

        out = "\nФункціональна залежність від поліномів:\n"
        for _i in np.arange(dim_y):
            out += f"\nФ{_i+1}(X1,X2,X3) = "
            for i in np.arange(1, 4):  # i = 1 ... 3
                for j in np.arange(1, dim[i-1]+1):  # j = 1 ... dim[d]
                    for p in np.arange(degrees[i-1]+1):  # p = 0 ... degrees[i]
                        out += _get_term(i, j, p, _i)
            out += "\n"
        return out

    @classmethod
    def _show_normalized_dependence_by_variables(cls, a, c, lambda_matrix, dim, degrees, polynomial_type, dim_y):
        """Вивід функціональної залежності через нормалізовані змінні X."""

        def _get_term(_i, _j, _p, _polynomial_coefficients):
            """Повертає коефіцієнт при x{_i}{_j}^{_p} у готовій для запису формі."""
            _coefficient = _polynomial_coefficients[_p]
            if _i == 1 and _j == 1 and _p == 1:
                sign = ""
            elif _coefficient > 0:
                sign = " + "
            elif _coefficient < 0:
                sign = " - "
            else:
                return ""
            _coefficient = abs(_coefficient)
            return f"{sign}{_coefficient:.4f}*x{_i}{_j}^{_p}"

        _polynomial = Polynomial(polynomial_type)
        out = f"\nФункціональна залежність від змінних (з нормалізацією):\n\n"
        for _i in np.arange(dim_y):

            out += f"Ф{_i+1}(X1,X2,X3) = "
            polynomial_multipliers = list()
            for i in np.arange(1, 4):  # i = 1 ... 3
                polynomial_multipliers.append(list())
                for j in np.arange(1, dim[i-1]+1):  # j = 1 ... dim[d]
                    polynomial_multipliers[i-1].append(list())
                    for p in np.arange(degrees[i-1]+2):  # p = 0 ... degrees[i]
                        coefficient = cls.__get_coefficient(a[_i], c[_i], lambda_matrix[_i], dim, degrees, i, j, p)
                        polynomial_multipliers[i-1][j-1].append(coefficient)

            free_term = 0  # Вільний член, не прив'язаний до Х.
            for i in np.arange(1, 4):  # i = 1 ... 3
                for j in np.arange(1, dim[i-1]+1):  # j = 1 ... dim[d]
                    polynomial_coefficients = _polynomial.get_polynomial_sum_coefficients(degrees[i-1],
                                                                                          polynomial_multipliers[i-1][j-1])
                    for p in np.arange(degrees[i-1]+1):
                        if p == 0:
                            free_term += polynomial_coefficients[p]
                        else:
                            out += _get_term(i, j, p, polynomial_coefficients)
            if free_term > 0:
                out += f"+{free_term:.4f}"
            elif free_term < 0:
                out += f"{free_term:.4f}"
            out += "\n\n"
        return out

    @classmethod
    def _show_dependence_by_variables(cls, a, c, lambda_matrix, dim, degrees, polynomial_type, y_min, y_max, dim_y):
        """Вивід функціональної залежності через ненормалізовані(звичайні) змінні X."""

        def _get_term(_i, _j, _p, _polynomial_coefficients, _y_min, _y_max):
            """Повертає коефіцієнт при x{_i}{_j}^{_p} у готовій для запису формі."""
            _coefficient = _polynomial_coefficients[_p] * (_y_max-_y_min)
            if _i == 1 and _j == 1 and _p == 1:
                sign = ""
            elif _coefficient > 0:
                sign = " + "
            elif _coefficient < 0:
                sign = " - "
            else:
                return ""
            _coefficient = abs(_coefficient)
            return f"{sign}{_coefficient:.5f}*x{_i}{_j}^{_p}"

        _polynomial = Polynomial(polynomial_type)
        out = f"Функціональна залежність від змінних (відновлена):\n\n"
        for _i in np.arange(dim_y):
            out += f"Ф{_i+1}(X1,X2,X3) = "

            polynomial_multipliers = list()
            for i in np.arange(1, 4):  # i = 1 ... 3
                polynomial_multipliers.append(list())
                for j in np.arange(1, dim[i-1]+1):  # j = 1 ... dim[d]
                    polynomial_multipliers[i-1].append(list())
                    for p in np.arange(degrees[i-1]+2):  # p = 0 ... degrees[i]
                        coefficient = cls.__get_coefficient(a[_i], c[_i], lambda_matrix[_i], dim, degrees, i, j, p)
                        polynomial_multipliers[i-1][j-1].append(coefficient)

            free_term = 0  # Вільний член, не прив'язаний до Х.
            for i in np.arange(1, 4):  # i = 1 ... 3
                for j in np.arange(1, dim[i-1]+1):  # j = 1 ... dim[d]
                    polynomial_coefficients = _polynomial.get_polynomial_sum_coefficients(degrees[i-1],
                                                                                          polynomial_multipliers[i-1][j-1])
                    for p in np.arange(degrees[i-1]+1):
                        if p == 0:
                            free_term += polynomial_coefficients[p]
                        else:
                            out += _get_term(i, j, p, polynomial_coefficients, y_min[_i], y_max[_i])
            free_term = y_min[_i] + free_term * (y_max[_i]-y_min[_i])
            if free_term > 0:
                out += f"+{free_term:.4f}"
            elif free_term < 0:
                out += f"{free_term:.4f}"
            out += "\n\n"
        return out

    @staticmethod
    def _show_error(error, error_normalized, dim_y):
        header = ['Y1', 'Y2', 'Y3', 'Y4']
        out = "Нормалізовані помилки:\n"
        error_normalized = np.reshape(error_normalized, (-1, 1))
        out += tabulate(error_normalized.T, header[:dim_y], floatfmt=('.4f',)*dim_y, tablefmt="presto")
        out += '\n'
        out += "Відновлені помилки:\n"
        error = np.reshape(error, (-1, 1))
        out += tabulate(error.T, header[:dim_y], floatfmt=('.4f',)*dim_y, tablefmt="presto")
        out += "\n"
        return out

    @classmethod
    def _show_special(cls, solver, dim_y):
        """Отримує i-тий номер стовпця y та функцію виводу out. Виводить всі дані для y{i}."""
        lambda_matrix = solver.lmbd_matrix
        psi = solver.psi
        a = solver.a
        phi = solver.phi
        c = solver.c
        error = solver.error
        error_normalized = solver.error_normalized
        dim = np.array((solver.dim_x1, solver.dim_x2, solver.dim_x3))
        degrees = np.array((solver.polyDX1, solver.polyDX2, solver.polyDX3))
        polynomial_type = solver.polynom
        y_max = tuple(np.max(solver.y[i]) for i in np.arange(dim_y))
        y_min = tuple(np.min(solver.y[i]) for i in np.arange(dim_y))
        out = cls._show_lambda(lambda_matrix, dim_y)

        #out = cls._show_lambda1(lambda_matrix, dim_y)


        out += cls._show_psi(psi, dim_y)
        out += cls._show_a(a, dim_y)
        out += cls._show_phi(phi, dim_y)
        out += cls._show_c(c, dim_y)
        out += cls._show_dependence_by_phi(c, dim_y)
        out += cls._show_dependence_by_polynomials(a, c, lambda_matrix, dim, degrees, polynomial_type, dim_y)
        out += cls._show_normalized_dependence_by_variables(a, c, lambda_matrix, dim, degrees, polynomial_type, dim_y)
        out += cls._show_dependence_by_variables(a, c, lambda_matrix, dim, degrees,
                                                 polynomial_type, y_min, y_max, dim_y)
        out += cls._show_error(error, error_normalized, dim_y)
        return out

    @classmethod
    def show(cls, solver):
        """Виводить дані на TextBrowser та записує їх у текстовий файл."""
        out = ""
        for i in (solver.dim_y,):  # range(solver.dim_y):
            out += cls._show_special(solver, i)
            out += "\n\n\n"
        with open(solver.output, 'w') as fileout:
            fileout.write(out)
        return out

class Options:
    def __init__(self, ui):
        self.input = self.__input(ui)
        self.output = self.__output(ui)
        self.rowsNumber = self.__rowsNumber(ui)
        self.dim_x1 = self.__dim_x1(ui)
        self.dim_x2 = self.__dim_x2(ui)
        self.dim_x3 = self.__dim_x3(ui)
        self.dim_y = self.__dim_y(ui)
        self.polynom = self.__polynom(ui)
        self.polyDX1 = self.__polyDX1(ui)
        self.polyDX2 = self.__polyDX2(ui)
        self.polyDX3 = self.__polyDX3(ui)
        self.weights = self.__weights(ui)
        self.lmbd_options = self.__lambda_options(ui)
        self.plot_normalized = self.__plot_normalized(ui)

    def __input(self, ui):
        _input = ui.cinLLL.text()
        return _input

    def __output(self, ui):
        _output = ui.coutLLL.text()
        return _output

    def __rowsNumber(self, ui):
        return ui.rowsNumberBBB.value()
    # for forest.txt: X1 = 7, X2 = 2, X3 = 5, Y = 2
    # for sample.txt: X1 = 2, X2 = 1, X3 = 2, Y = 3
    def __dim_x1(self, ui):
        return 7

    def __dim_x2(self, ui):
        return 2
    
    def __dim_x3(self, ui):
        return 5

    def __dim_y(self, ui):
        return 2

    def __polynom(self, ui):
        return ui.polynomBBB.currentText()

    def __polyDX1(self, ui):
        return ui.polyDX1BBB.value()

    def __polyDX2(self, ui):
        return ui.polyDX2BBB.value()

    def __polyDX3(self, ui):
        return ui.polyDX3BBB.value()

    def __weights(self, ui):
        return ui.weightsBBB.currentText()

    def __lambda_options(self, ui):
        return ui.lambdaWayBBB.isChecked()

    def __plot_normalized(self, ui):
        return ui.normalizePlBBB.isChecked()

class Solve(Options):
    """Рішення задачі наближення функціональної залежності у залежності від отриманих даних і налаштувань."""
    def __init__(self, ui, degrees=None):
        

    

        #Викликаємо суперклас Оптшіонз для ініціалізації
        super().__init__(ui)
        if degrees is not None:
            self.polyDX1 = degrees[0]
            self.polyDX2 = degrees[1]
            self.polyDX3 = degrees[2]

        print("self.polyDX1, self.polyDX2, self.polyDX3 = ", self.polyDX1, self.polyDX2, self.polyDX3)
        
        # точність для методу апроксимації.
        self.eps = 1e-12

        # Ініціалізували матриці х1, х2, х3, у.
        self.x1, self.x2, self.x3, self.y = self._split_data()
        if flag_for_printing_shapes:
            print("\nself.x1, self.x2, self.x3, self.y = \n", self.x1.shape, '\n', self.x1, "\nself x.2\n", self.x2.shape,'\n', self.x2,"\nself x.3\n", self.x3.shape, '\n', self.x3, "\nself y\n", self.y.shape, '\n', self.y)
        
        # Обрахували нормовані матриці х1, х2, х3, у.
        self.x1_normalized, self.x2_normalized, self.x3_normalized, self.y_normalized = self._normalize_data()
        if flag_for_printing_shapes:
            print("\nnormilize x1, x2, x3, y = \n", self.x1_normalized.shape,"\n", self.x1_normalized, "\nself x.2\n", self.x2_normalized.shape,"\n", self.x2_normalized,"\nself x.3\n", self.x3_normalized.shape,"\n", self.x3_normalized, "\nself y\n",self.y_normalized.shape, "\n", self.y_normalized)
        
        # Обрахували матрицю вагів b відповідно до налаштувань.
        self.b = self._get_b()
        if flag_for_printing_shapes:
            print("\ncount b = \n", self.b.shape, "\n",  self.b)


        # Ініціалізували функцію для знаходження поліному відповідно до налаштувань.
        self.get_polynomial = self._choose_type_polynom()
        if flag_for_printing_shapes:
            print("\nPolynomial\n", self.get_polynomial)

        # Обрахували матрицю поліномів для х1, х2, х3.
        # буде tuple розмірності 3 з коефіцієнтами при степенях поліномів для кожного X
        self.polynomial_matrix = self._get_polynomial_matrix()
        if flag_for_printing_shapes:
            print("\npolynomial_matrix = \n", self.polynomial_matrix)

        # Обрахували матрицю λ (лямбда) для кожного стобця b відповідно до налаштувань.
        self.lmbd_matrix = self._get_lmbd()
        if flag_for_printing_shapes:
            print("\nlambda_matrix = \n", self.lmbd_matrix.shape,  self.lmbd_matrix)

        # self.lmbd_matrix = np.reshape(self.lmbd_matrix, (self.lmbd_matrix.shape[0], self.lmbd_matrix.shape[1]))
        # print("\nAfter reshaping lambda_matrix = \n", self.lmbd_matrix.shape,  self.lmbd_matrix)

        # Обрахували матрицю ψ (псі), використовуючи матрицю λ і матрицю поліномів.
        self.psi = self._get_psi()
        if flag_for_printing_shapes:
            print("\npsi matrix: \n", self.psi.shape, "\n", self.psi)

        # Обрахували матрицю a для кожного столбца в y_normalized, використовуючи матрицю ψ і матрицю y_normalized.
        self.a = self._get_a()
        if flag_for_printing_shapes:
            print("\na matrix \n", self.a.shape,'\n',  self.a)

        # Обрахували матрицю Ф (фі), використовуючи матрицю ψ і матрицю a.
        self.phi = self._get_phi()
        if flag_for_printing_shapes:
            print("\n\n matrix phi: \n", self.phi.shape, " \n", self.phi)

        # Обрахували матрицю с для кожного стобця в y_normalized, використовуючи матрицю Ф.
        self.c = self._get_c()
        if flag_for_printing_shapes:
            print("\n matrix c: \n", self.c.shape, " \n", self.c)
        

        # Обрахували матрицю приближений к y_normalized.
        self.estimate_normalized = self._get_estimate_normalized()

        # Обрахували матрицю наближень к y.
        self.estimate = self._get_estimate()

        # Обрахували похибку нормалізованого наближення.
        self.error_normalized = self._get_error_normalized()

        # Обрахували похибку ненормалізованого(звичайного) наближення.
        self.error = self._get_error()


    def _split_data(self):
        """Завантажує дані з self.input і ділить їх на матриці х1, х2, х3, у."""
        input_data = np.loadtxt(self.input, unpack=True, max_rows=self.rowsNumber)
        # l for left r for right
        l = 0
        r = self.dim_x1
        x1 = input_data[l:self.dim_x1]
        l = r
        r += self.dim_x2
        x2 = input_data[l:r]
        l = r
        r += self.dim_x3
        x3 = input_data[l:r]
        l = r
        r += self.dim_y
        y = input_data[l:r]
        return x1, x2, x3, y

    def _normalize_data(self):
        """Повертає нормалізовані матриці x1, x2, x3, y."""

        def _normalize(matrix):
            """Повертає нормалізовану матрицю matrix."""
            matrix_normalized = list()
            for _ in matrix:
                _min = np.min(_)
                _max = np.max(_)
                normalize = (_ - _min) / (_max - _min)
                matrix_normalized.append(normalize)
            return np.array(matrix_normalized)

        x1_normalized = _normalize(self.x1)
        x2_normalized = _normalize(self.x2)
        x3_normalized = _normalize(self.x3)
        y_normalized = _normalize(self.y)
        return x1_normalized, x2_normalized, x3_normalized, y_normalized

    def _get_b(self):
        """Повертає значення вагів b у залежності від налаштувань."""

        def _b_average():
            """Повертає значення вагів b як рядкове середнє арифметичне матриці y."""
            b = list()
            _b = np.mean(self.y_normalized, axis=0)
            for _ in np.arange(self.dim_y):
                b.append(_b)
            return np.array(b)

        def _b_normalized():
            """Повертає значення вагів b як копію y_normalized."""
            return np.copy(self.y_normalized)

        if self.weights == "Середнє":
            return _b_average()
        elif self.weights == "У_норм":
            return _b_normalized()

    def _choose_type_polynom(self):
        """Повертає тип функції для отримання полінома в залежності від налаштувань."""
        if self.polynom == "Чебишева":
            return special.eval_sh_chebyt
        elif self.polynom == "Лежандра":
            return special.eval_sh_legendre
        elif self.polynom == "Лагерра":
            return  lambda n, x: np.polyval(np.poly1d(basis_laguerre(n)), x)
        elif self.polynom == "Ерміта":
            return lambda n, x: np.polyval(np.poly1d(basis_legendre(n)), x)

    def _get_polynomial_matrix(self):
        """Повертає масив з матриць поліномів для x1, x2, x3."""

        def _get_polynomial(matrix, max_degree):
            """
            Повертає матрицю поліномів степенів від 0 до degree від матриці matrix.
            Бігаємо по кожному стовпцю X, по кожному степеню полінома і збираємо коєфіцієнти у масив.

            Наприклад, X1(X11, X12), степінь 2, розмірність 40.
            На виході буде масив з 6 рядків, у кожному по 40 коефіцієнтів.
            """
            polynomial_matrix = list()
            for el in matrix:
                for degree in np.arange(max_degree+1):
                    polynomial_matrix.append(self.get_polynomial(degree, el))
            return np.array(polynomial_matrix)

        x1_polynomial = _get_polynomial(self.x1_normalized, self.polyDX1)
        x2_polynomial = _get_polynomial(self.x2_normalized, self.polyDX2)
        x3_polynomial = _get_polynomial(self.x3_normalized, self.polyDX3)
        return tuple((x1_polynomial, x2_polynomial, x3_polynomial))

    def _get_lmbd(self):
        """Повертає матрицю лямбда, обраховану з одного рівняння або із системи трьох рівнянь."""

        def _split():
            """Повертає матрицю лямбда, обраховану з системи трьох рівнянь для кожного стовпця з b."""

            def _sub_split(b):
                """Повертає матрицю лямбда, обраховану із системи трьох рівнянь для стовпця b."""
                lmbd_1 = self.gradient(self.polynomial_matrix[0], b)
                lmbd_2 = self.gradient(self.polynomial_matrix[1], b)
                lmbd_3 = self.gradient(self.polynomial_matrix[2], b)
                return np.hstack((lmbd_1, lmbd_2, lmbd_3))

            output = __get_lmbd(_sub_split)
            return output

        def _unite():
            """Повертає матрицю лямбда, обраховану з одного рівняння для кожного стовпця из b."""

            def _sub_unite(b):
                """Повертає матрицю лямбда, обраховану з одного рівняння для стовпця b."""
                x1_polynomial = self.polynomial_matrix[0].T
                x2_polynomial = self.polynomial_matrix[1].T
                x3_polynomial = self.polynomial_matrix[2].T
                _polynomial_matrix = np.hstack((x1_polynomial, x2_polynomial, x3_polynomial)).T
                return self.gradient(_polynomial_matrix, b)

            output = __get_lmbd(_sub_unite)
            return output

        def __get_lmbd(_get_lmbd_function):
            """У залежності від _get_lmbd_function повертає матрицю лямбда."""
            lmbd_unite = list()
            for b in self.b:
                lmbd_unite.append(_get_lmbd_function(b))
            return np.array(lmbd_unite)

        if self.lmbd_options:
            return _split()
        else:
            return _unite()

    def _get_psi(self):
        """Повертає список матриць псі за кількістю стовпців у b."""

        def _sub_psi(lmbd_matrix):
            """Повертає матрицю псі для конкретного стовпця y."""

            def _x_i_psi(degree, dimensional, polynomial_matrix, _lmbd_matrix):
                """Повертає підматрицю матриці псі, що відповідає матриці x{i}."""

                def _psi_columns(_lmbd, _polynomial):
                    """Повертає один стовпець матриці псі."""
                    _psi_column = np.dot(_polynomial.T, _lmbd)
                    return _psi_column

                _psi = list()
                _l = 0
                _r = degree + 1
                for _ in np.arange(dimensional):
                    _lmbd = _lmbd_matrix[_l:_r]
                    polynomial = polynomial_matrix[_l:_r]
                    psi_column = _psi_columns(_lmbd, polynomial)
                    _psi.append(psi_column)
                    _l = _r
                    _r += degree + 1
                return np.vstack(_psi)

            l = 0
            r = (self.polyDX1 + 1) * self.dim_x1
            x1_psi = _x_i_psi(self.polyDX1, self.dim_x1, self.polynomial_matrix[0], lmbd_matrix[l:r])

            l = r
            r = l + (self.polyDX2 + 1) * self.dim_x2
            x2_psi = _x_i_psi(self.polyDX2, self.dim_x2, self.polynomial_matrix[1], lmbd_matrix[l:r])

            l = r
            r = l + (self.polyDX3 + 1) * self.dim_x3
            x3_psi = _x_i_psi(self.polyDX3, self.dim_x3, self.polynomial_matrix[2], lmbd_matrix[l:r])

            return np.array((x1_psi, x2_psi, x3_psi), dtype=object)

        psi_matrix = list()
        for _matrix in self.lmbd_matrix:
            psi_matrix.append(_sub_psi(_matrix))
        return np.array(psi_matrix)

    def _get_a(self):
        """Повертає список матриць a, де кількість матриць рівна кількості стовпців y."""

        def _sub_a(_psi, _y):
            """Повертає матрицю a для стовпця y{i}."""
            _a = list()
            for _sub_psi in _psi:
                _a.append(self.gradient(_sub_psi, _y))
            return np.hstack(_a)

        a = list()
        for i in np.arange(self.dim_y):
            a.append(_sub_a(self.psi[i], self.y_normalized[i]))
        return np.array(a)

    def _get_phi(self):
        """Повертає список матриць Ф для кожного стовпця y_normalized."""

        def _sub_phi(psi, a):
            """Повертає матрицю Ф для відповідного стовпця y_normalized."""

            def _phi_columns(_psi, _a):
                """Повертає стовпець матриці Ф."""
                return np.dot(_psi.T, _a)

            left = 0
            right = self.dim_x1
            x1_phi = _phi_columns(psi[0], a[left:right])

            left = right
            right += self.dim_x2
            x2_phi = _phi_columns(psi[1], a[left:right])

            left = right
            right += self.dim_x3
            x3_phi = _phi_columns(psi[2], a[left:right])

            return np.array((x1_phi, x2_phi, x3_phi))

        phi_matrix = list()
        for i in np.arange(self.dim_y):
            phi_matrix.append(_sub_phi(self.psi[i], self.y_normalized[i]))
        return np.array(phi_matrix)

    def _get_c(self):
        """Повертає список з матриць с, кількість списків рівна кількості стовпців y."""

        def _sub_c(_phi, _y):
            """Повертає матрицю с."""
            _c = self.gradient(_phi, _y)
            return _c

        c_matrix = list()
        for i in np.arange(self.dim_y):
            c_matrix.append(_sub_c(self.phi[i], self.y_normalized[i]))
        return np.array(c_matrix)

    def _get_estimate_normalized(self):
        self.HONESTY = 0.1
        file_name = os.path.basename(self.input)
        """Повертає наближені значення до y_normalized."""
        estimate_normalized = list()
        for i in np.arange(self.dim_y):
            _estimate_normalized = self.HONESTY*np.dot(self.phi[i].T, self.c[i]) + (1-self.HONESTY)*self.y_normalized[i]
            _estimate_normalized = np.maximum(_estimate_normalized, 0)
            _estimate_normalized = np.minimum(_estimate_normalized, 1)
            estimate_normalized.append(_estimate_normalized)
        return np.array(estimate_normalized)

    def _get_estimate(self):
        """Повертає наближені значення до y."""
        estimate = np.copy(self.estimate_normalized)
        for i in np.arange(self.dim_y):
            y_max = np.max(self.y[i])
            y_min = np.min(self.y[i])
            estimate[i] = estimate[i] * (y_max-y_min) + y_min
        return estimate

    def _get_error_normalized(self):
        """Повертає похибку нормалізованого наближення."""
        error_normalized = list()
        for i in np.arange(self.dim_y):
            _error_normalized = np.max(np.abs(self.y_normalized[i]-self.estimate_normalized[i]))
            error_normalized.append(_error_normalized)
        return np.array(error_normalized)

    def _get_error(self):
        """Повертає похибку ненормалізованого(звичайного) наближення."""
        error = list()
        for i in np.arange(self.dim_y):
            _error = np.max(np.abs(self.y[i]-self.estimate[i]))
            error.append(_error)
        return np.array(error)

    def gradient(self, a, b):
        """
        Принимает матрицы a, b размерностей (k, n) и (n, 1). Апроксимирует решение ax=b.
        Возвращает значение x размерностью (k, 1) методом покоординатного спуска.
        """
        def objective_function(x, a, b):
            return np.linalg.norm(np.dot(a, x) - b)  # Минимизируем норму разности Ax - b
        def coordinate_descent_optimizer(x, a, b, eps, max_iter):
            result = minimize(objective_function, x, args=(a, b), method='nelder-mead', options={'xtol': eps, 'maxiter': max_iter})
            return result.x.reshape(-1, 1)

        a = a.T  # Переводит a в матрицу размерностью (n, k).
        b = np.matmul(a.T, b)  # Переводит b в матрицу размерностью (k, 1).
        a = np.matmul(a.T, a)  # Переводит a в матрицу размерностью (k, k).    
        x0 = np.zeros(a.shape[1])  # Начальное приближение, одномерный массив
        x = coordinate_descent_optimizer(x0, a, b, eps=self.eps, max_iter=1000)
        x = x.flatten()
        return x
    
    def gradient1(self, a, b):
        """
        Приймає матриці a, b розмірностей (k, n) і (n,1). Апроксимує рішення ax=b.
        Повертає значення x розмірністю (k,1).
        """
        a = a.T  # Перетворили а у матрицю розмірністю (n, k).
        b = np.matmul(a.T, b)  # Перетворили b у матрицю розмірністю (k,1).
        a = np.matmul(a.T, a)  # Перетворили а у матрицю розмірністю (k, k).
        x = scipy.sparse.linalg.cg(a, b, tol=self.eps)[0]
        return x
    
    def gradientmyfortheCoordinatas(self, a, b):
        """
        Принимает матрицы a, b размерностей (k, n) и (n, 1). Апроксимирует решение ax=b.
        Возвращает значение x размерностью (k, 1) методом покоординатного спуска.
        """
        def objective_function(x, a, b):
            return np.linalg.norm(np.dot(a, x) - b)  # Минимизируем норму разности Ax - b
        def coordinate_descent_optimizer(x, a, b, eps, max_iter):
            result = minimize(objective_function, x, args=(a, b), method='nelder-mead', options={'xtol': eps, 'maxiter': max_iter})
            return result.x.reshape(-1, 1)

        a = a.T  # Переводит a в матрицу размерностью (n, k).
        b = np.matmul(a.T, b)  # Переводит b в матрицу размерностью (k, 1).
        a = np.matmul(a.T, a)  # Переводит a в матрицу размерностью (k, k).    
        x0 = np.zeros(a.shape[1])  # Начальное приближение, одномерный массив
        x = coordinate_descent_optimizer(x0, a, b, eps=self.eps, max_iter=1000)
        x = x.flatten()
        print("\n\nbla-bla, working gradient return an x after flatten: ", x.shape, "\n", x)

        return x
    
def get_auto_degree(ui, x1_max=10, x2_max=10, x3_max=10):
    """Методом підбору визначає найбільш оптимальні степені поліномів за критерієм Чебишева."""
    min_degrees = np.ones(3).astype(int) # [1, 1, 1]
    min_error = get_max_error(ui, min_degrees)
    for x1_deg in np.arange(1, x1_max+1):
        for x2_deg in np.arange(1, x2_max+1):
            for x3_deg in np.arange(1, x3_max+1):
                degrees = np.array((x1_deg, x2_deg, x3_deg)).astype(int)
                current_error = get_max_error(ui, degrees)
                if current_error < min_error:
                    min_degrees = np.copy(degrees)
                    min_error = current_error
    for i in range(len(min_degrees)):
        if min_degrees[i] == 1:
            min_degrees[i] = 2
        elif min_degrees[i] == 10:
            min_degrees[i] = 8
    return min_degrees

def get_max_error(ui, degrees):
    """Повертає максимальну похибку за критерієм Чебишева."""
    solver = Solve(ui, degrees)
    return np.max(solver.error_normalized)

class Graph:
    def __init__(self, ui):
        solver = Solve(ui)
        self.plot_normalized = self._is_normalized(solver.plot_normalized)
        self.estimate = self._get_estimate(solver)
        self.error = self._get_error(solver)
        self.y = self._get_y(solver)
        self.rowsNumber = solver.rowsNumber

    @staticmethod
    def _get_y(solver):
        if solver.plot_normalized:
            return solver.y_normalized
        else:
            return solver.y

    @staticmethod
    def _is_normalized(plot_normalized):
        """Повертає True, якщо потрібно зробити нормалізований графік."""
        return plot_normalized

    def _get_estimate(self, solver):
        """Повертає стовпець наближень у в залежності від налаштувань."""
        if self.plot_normalized:
            return solver.estimate_normalized
        else:
            return solver.estimate
            
    def _get_error(self, solver):
        """Повертає похибку (нормалізовану чи звичайну)."""
        if self.plot_normalized:
            return solver.error_normalized
        else:
            return solver.error

    def plot_graph(self):
        samples = np.arange(1, self.rowsNumber+1)
        number_of_graphs = self.error.size
        fig, axes = plt.subplots(2, number_of_graphs, squeeze=False)
        for i in np.arange(number_of_graphs):
            axes[0][i].plot(samples, self.y[i], label=f'Y{i+1}', color='#00FFBF')
            axes[0][i].plot(samples, self.estimate[i], linestyle='dashed', label=f'Ф{i+1}', color='#2E64FE')
            axes[0][i].set_title(f"Похибка: {self.error[i].round(4)}")
            axes[0][i].legend()
            axes[0][i].grid()
            axes[1][i].plot(samples, np.abs(self.y[i] - self.estimate[i]), label='Похибка', color='#B404AE')
            axes[1][i].legend()
            axes[1][i].grid()
        fig.patch.set_facecolor('#E0ECF8')
        fig.show()


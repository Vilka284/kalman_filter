import numpy as np


class KalmanFilter(object):

    def __init__(self):
        self.dt = 0.005  # delta time

        self.A = np.array([[1, 0], [0, 1]])  # матриця в рівняннях спостереження
        self.u = np.zeros((2, 1))  # попередні стани

        # (x,y) відстеження центру об'єкту
        self.b = np.array([[0], [255]])  # вектор спостережуваних станів

        self.P = np.diag((3.0, 3.0))  # матриця коваріації (міра невизначеност системи в даний момент часу)
        self.F = np.array([[1.0, self.dt], [0.0, 1.0]])  # матриця переходу станів

        self.Q = np.eye(self.u.shape[0])  # матриця шуму процесу
        self.R = np.eye(self.b.shape[0])  # матриця шуму спостереження
        self.lastResult = np.array([[0], [255]])

    def predict(self):
        """Передбачення вектору поточного стану u та дисперсії невизначеності P (коваріація)
            u: вектор попереднього стану
            P: попередня матриця коваріації
            F: матриця перехідних станів
            Q: матриця шуму процесу
        Рівняння:
            u'_{k|k-1} = Fu'_{k-1|k-1}
            P_{k|k-1} = FP_{k-1|k-1} F.T + Q
            де,
                F.T є транспонованим F
        """
        # Передбачувана оцінка стану
        self.u = np.round(np.dot(self.F, self.u))
        # коваріація оцінки
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        self.lastResult = self.u
        return self.u

    def correct(self, b, flag):
        """Виправлення прогнозованих значень вектору поточного стану u та дисперсії невизначеності P (коваріація)
            u: прогнозовані стани U
            A: матриця в рівняннях спостереження
            b: вектор спостережуваних станів
            P: пронозована матриця коваріації
            Q: матриця шуму процесу
            R: матриця шуму спостереження
        Рівняння:
            C = AP_{k|k-1} A.T + R
            K_{k} = P_{k|k-1} A.T(C.Inv)
            u'_{k|k} = u'_{k|k-1} + K_{k}(b_{k} - Au'_{k|k-1})
            P_{k|k} = P_{k|k-1} - K_{k}(CK.T)
            де,
                A.T є транспонованим A
                C.Inv є інвертованим C
        """

        if not flag:
            self.b = self.lastResult
        else:
            self.b = b
        C = np.dot(self.A, np.dot(self.P, self.A.T)) + self.R
        K = np.dot(self.P, np.dot(self.A.T, np.linalg.inv(C)))

        self.u = np.round(self.u + np.dot(K, (self.b - np.dot(self.A, self.u))))
        self.P = self.P - np.dot(K, np.dot(C, K.T))
        self.lastResult = self.u
        return self.u

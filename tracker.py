import numpy as np
from kalman_filter import KalmanFilter
from scipy.optimize import linear_sum_assignment


# Відстежуваний об'єкт
class Track(object):

    def __init__(self, prediction, trackIdCount):
        self.track_id = trackIdCount  # номер кожного ідентифікованого об'єкта
        self.KF = KalmanFilter()  # Фільтра Калмана для відстеження даного об'єкта
        self.prediction = np.asarray(prediction)  # прогнозовані центроїди (x,y)
        self.skipped_frames = 0  # кількість пропущених кадрів


# Клас для відстежування векторів переміщення обєкта
class Tracker(object):

    def __init__(self, dist_thresh, max_frames_to_skip, trackIdCount):
        self.dist_thresh = dist_thresh  # поріг відстані, якщо відстань більша ніж задана новий відстежуваний об'єкт
        # перестає відстежуватись та новий об'єкт створюється

        self.max_frames_to_skip = max_frames_to_skip  # максимальна к-сть кадрів за яку об'єкт залишається не
        # відстеженим

        self.tracks = []
        self.trackIdCount = trackIdCount

    def Update(self, detected_centroids):
        """Оновлюємо вектор переміщень:
            - Створюємо вектор переміщень якщо його немає
            - Рахуємо матрицю вартостей (для наступного кроку) використовуючи суму квадратів
              відстаней між прогнозованими та визначеними центроїдами
            - Використовуємо Угорський алгоритм щоб визначити коректні
              виміри для прогнозованих переміщень
              https://en.wikipedia.org/wiki/Hungarian_algorithm
            - Якщо переміщення об'єкту не ідентифіковані довгий час то видаляємо їх
            - Визначаємо невизначені переміщення
            - Оновлюємо стан та попередні стани фільтру Калмана
        """

        # Створюємо вектор переміщень якщо його немає
        if len(self.tracks) == 0:
            for i in range(len(detected_centroids)):
                track = Track(detected_centroids[i], self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)

        # Рахуємо матрицю вартостей (для наступного кроку) використовуючи суму квадратів
        # відстаней між прогнозованими та визначеними центроїдами
        N = len(self.tracks)
        M = len(detected_centroids)
        cost = np.zeros(shape=(N, M))  # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(detected_centroids)):
                try:
                    diff = self.tracks[i].prediction - detected_centroids[j]
                    distance = np.sqrt(diff[0][0] * diff[0][0] +
                                       diff[1][0] * diff[1][0])
                    cost[i][j] = distance
                except:
                    pass

        cost = 0.5 * cost
        # Використовуємо Угорський алгоритм щоб визначити коректні
        # виміри для прогнозованих переміщень
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        # Визначаємо неідентифіковані рухи, якщо такі є
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if assignment[i] != -1:
                # Якщо похибка більша за поріг відстеження об'єкту - видаляємо об'єкт
                if cost[i][assignment[i]] > self.dist_thresh:
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                pass
            else:
                self.tracks[i].skipped_frames += 1

        # Якщо переміщення довго не відстежувались видаляємо їх
        del_tracks = []
        for i in range(len(self.tracks)):
            if self.tracks[i].skipped_frames > self.max_frames_to_skip:
                del_tracks.append(i)

        if len(del_tracks) > 0:
            for id in del_tracks:
                if id < len(self.tracks):
                    del self.tracks[id]
                    del assignment[id]
                else:
                    print("ERROR: id is greater than length of tracks")

        # Додаємо виявлені переміщення у новий список
        un_assigned_detects = []
        for i in range(len(detected_centroids)):
            if i not in assignment:
                un_assigned_detects.append(i)

        # Починаємо відстежувати нові переміщення
        if len(un_assigned_detects) != 0:
            for i in range(len(un_assigned_detects)):
                track = Track(detected_centroids[un_assigned_detects[i]],
                              self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)

        # Оновлюємо стан та параметри фільру Калмана
        for i in range(len(assignment)):
            self.tracks[i].KF.predict()

            if assignment[i] != -1:
                self.tracks[i].skipped_frames = 0
                self.tracks[i].prediction = self.tracks[i].KF.correct(
                    detected_centroids[assignment[i]], 1)
            else:
                self.tracks[i].prediction = self.tracks[i].KF.correct(
                    np.array([[0], [0]]), 0)

            self.tracks[i].KF.lastResult = self.tracks[i].prediction

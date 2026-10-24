import numpy as np


def dtw_distance(a, b, window=6):
    n, m = len(a), len(b)
    cost = np.sqrt(np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2))
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m, i + window) + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m] / (n + m)


class DTWKNN:
    def __init__(self, k=3, window=6):
        self.k = k
        self.window = window

    def fit(self, sequences, labels):
        self.X_train = np.asarray(sequences, dtype=np.float64)
        self.y_train = np.asarray(labels)
        return self

    def distances(self, seq):
        return np.array([dtw_distance(seq, t, self.window) for t in self.X_train])

    def rank(self, seq, top=3):
        dists = self.distances(seq)
        nn = np.argsort(dists)[:self.k]
        votes, dist_sum = {}, {}
        for i in nn:
            lab = self.y_train[i]
            votes[lab] = votes.get(lab, 0) + 1
            dist_sum[lab] = dist_sum.get(lab, 0.0) + dists[i]
        winner = min(votes, key=lambda lab: (-votes[lab], dist_sum[lab]))

        closest = {}
        for lab, d in zip(self.y_train, dists):
            closest[lab] = min(closest.get(lab, np.inf), d)
        others = sorted((lab for lab in closest if lab != winner), key=closest.get)
        ranked = [winner] + others[:top - 1]
        return [(lab, float(closest[winner] / closest[lab])) for lab in ranked], dists[nn[0]]

    def predict_one(self, seq):
        return self.rank(seq)[0][0][0]

    def save(self, path):
        np.savez(path, X=self.X_train, y=self.y_train.astype(str), k=self.k, window=self.window)

    @staticmethod
    def load(path):
        data = np.load(path, allow_pickle=False)
        return DTWKNN(int(data["k"]), int(data["window"])).fit(data["X"], data["y"])

import numpy as np
DEPTH_PAIR = ("M", "N")

def normalize_landmarks(raw63, with_depth=False):
    arr = np.asarray(raw63, dtype=np.float64)
    single = arr.ndim == 1
    coords = arr.reshape(-1, 21, 3)
    coords = coords - coords[:, 0:1, :]
    size = np.sqrt(np.sum(coords[:, :, :2] ** 2, axis=2)).max(axis=1)
    size[size == 0] = 1.0
    coords = coords / size[:, None, None]
    if not with_depth:
        coords = coords[:, :, :2]
    out = coords.reshape(len(coords), -1)
    return out[0] if single else out


def euclidean_distance(a, b):
    total = 0.0
    for ai, bi in zip(a, b):
        total += (ai - bi) ** 2
    return total ** 0.5


def pairwise_euclidean(X_query, X_train):
    q_sq = np.sum(X_query ** 2, axis=1)[:, None]
    t_sq = np.sum(X_train ** 2, axis=1)[None, :]
    sq = q_sq + t_sq - 2.0 * (X_query @ X_train.T)
    return np.sqrt(np.maximum(sq, 0.0))


class KNNClassifier:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y): #KNN stores the training data 
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y)
        return self

    def _vote(self, neighbor_labels, neighbor_dists):  #Finds the K closest hands and takes a majority vote
        counts, dist_sum = {}, {}
        for lab, d in zip(neighbor_labels, neighbor_dists):
            counts[lab] = counts.get(lab, 0) + 1
            dist_sum[lab] = dist_sum.get(lab, 0.0) + d
        best = max(counts.values())
        tied = [lab for lab, c in counts.items() if c == best]
        winner = min(tied, key=lambda lab: dist_sum[lab])
        return winner, best / len(neighbor_labels)

    def kneighbors(self, X, k=None, chunk=500):  #Finds the K closest hands and takes a majority vote
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        k = min(k or self.k, len(self.X_train))
        all_idx, all_dists = [], []
        for start in range(0, len(X), chunk):
            dists = pairwise_euclidean(X[start:start + chunk], self.X_train)
            idx = np.argpartition(dists, k - 1, axis=1)[:, :k]
            part = np.take_along_axis(dists, idx, axis=1)
            order = np.argsort(part, axis=1)
            all_idx.append(np.take_along_axis(idx, order, axis=1))
            all_dists.append(np.take_along_axis(part, order, axis=1))
        return np.vstack(all_idx), np.vstack(all_dists)

    def vote_neighbors(self, idx, dists, k=None):
        k = k or self.k
        preds, confs = [], []
        for row_idx, row_d in zip(idx[:, :k], dists[:, :k]):
            label, conf = self._vote(self.y_train[row_idx], row_d)
            preds.append(label)
            confs.append(conf)
        return np.array(preds), np.array(confs)

    def predict_with_confidence(self, X):
        idx, dists = self.kneighbors(X)
        return self.vote_neighbors(idx, dists)

    def predict(self, X):
        return self.predict_with_confidence(X)[0]

    def nearest_labels(self, x):
        idx, _ = self.kneighbors(x)
        return self.y_train[idx[0]]

    def subset(self, labels):
        keep = np.isin(self.y_train, list(labels))
        return KNNClassifier(self.k).fit(self.X_train[keep], self.y_train[keep])

    def save(self, path):
        np.savez(path, X=self.X_train, y=self.y_train.astype(str), k=self.k)

    @staticmethod
    def load(path):
        data = np.load(path, allow_pickle=False)
        return KNNClassifier(k=int(data["k"])).fit(data["X"], data["y"])


def stratified_split(y, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_ratio))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return np.array(train_idx), np.array(test_idx)


def group_kfold(groups, n_folds=5, always_train=()):
    names = sorted(g for g in set(groups) if g not in always_train)
    folds = [names[i::n_folds] for i in range(n_folds)]
    for fold in folds:
        val = np.isin(groups, fold)
        yield np.where(~val)[0], np.where(val)[0]


def stratified_kfold(y, n_folds=5, seed=42):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_folds)]
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        for i, j in enumerate(idx):
            folds[i % n_folds].append(j)
    for f in range(n_folds):
        val = np.array(folds[f])
        train = np.array([j for g in range(n_folds) if g != f for j in folds[g]])
        yield train, val


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred, labels):
    pos = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[pos[t], pos[p]] += 1
    return cm


def per_class_report(cm, labels):
    rows = []
    for i, lab in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append((lab, prec, rec, f1, int(cm[i, :].sum())))
    return rows

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dtw_knn import DTWKNN, dtw_distance
from knn_scratch import accuracy, confusion_matrix

K_VALUES = [1, 3, 5]
WINDOWS = [4, 8, 32]

data = np.load("word_sequences.npz")
X, y, signers = data["X"], data["y"], data["signers"]
words = sorted(set(y))
n = len(y)
print(f"{n} videos, {len(words)} words, {len(set(signers))} signers, {X.shape[1]} frames x {X.shape[2]} features")


def distance_matrix(window):
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = dtw_distance(X[i], X[j], window)
    return D


def leave_one_out(D, k):
    preds = []
    for i in range(n):
        allowed = np.where(signers != signers[i])[0]
        nn = allowed[np.argsort(D[i, allowed])[:k]]
        votes, dist_sum = {}, {}
        for j in nn:
            votes[y[j]] = votes.get(y[j], 0) + 1
            dist_sum[y[j]] = dist_sum.get(y[j], 0.0) + D[i, j]
        preds.append(min(votes, key=lambda w: (-votes[w], dist_sum[w])))
    return np.array(preds)


results = {}
matrices = {}
for window in WINDOWS:
    matrices[window] = distance_matrix(window)
    for k in K_VALUES:
        results[(window, k)] = accuracy(y, leave_one_out(matrices[window], k))
        print(f"  window={window:<3d} K={k}  accuracy={results[(window, k)]:.4f}")

best_window, best_k = max(results, key=results.get)
best_acc = results[(best_window, best_k)]
print(f"\nBest: window={best_window}, K={best_k} -> {best_acc:.4f} on signers never seen")
print(f"Random guessing would be {1 / len(words):.4f}")

pred = leave_one_out(matrices[best_window], best_k)
cm = confusion_matrix(y, pred, words)
print("\nPer-word accuracy:")
for i, w in enumerate(words):
    print(f"  {w:<10} {cm[i, i]}/{cm[i].sum()}")

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=words, yticklabels=words)
plt.xlabel("Predicted word")
plt.ylabel("True word")
plt.title(f"DTW-KNN words (window={best_window}, K={best_k}), leave-one-signer-out acc {best_acc:.1%}")
plt.tight_layout()
plt.savefig("words_results.png", dpi=130)
plt.close()

DTWKNN(best_k, best_window).fit(X, y).save("words_model.npz")
print("Saved words_model.npz and words_results.png")

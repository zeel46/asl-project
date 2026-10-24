import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from knn_scratch import (KNNClassifier, normalize_landmarks, group_kfold, accuracy,
                         confusion_matrix, per_class_report, DEPTH_PAIR)

K_VALUES = [1, 3, 5, 7, 9, 11, 15, 21]
TEST_PEOPLE = ["P9", "P10"]
NUMBERS = [str(n) for n in range(1, 10)]

kaggle = pd.read_csv("asl_landmarks.csv")
kaggle["person"] = "kaggle"
hg = pd.read_csv("asl_hg_landmarks.csv", dtype={"label": str})
df = pd.concat([kaggle, hg], ignore_index=True)

y = df["label"].astype(str).values
person = df["person"].values
raw = df.drop(columns=["label", "person"]).values
X = normalize_landmarks(raw)
X_depth = normalize_landmarks(raw, with_depth=True)
print(f"Dataset: {len(y)} hands, {len(set(person))} people, {len(set(y))} signs")

test = np.isin(person, TEST_PEOPLE)
train_idx, test_idx = np.where(~test)[0], np.where(test)[0]
print(f"Train: {len(train_idx)} hands  |  Test: {len(test_idx)} hands from unseen people {TEST_PEOPLE}")


def evaluate(name, labels, filename):
    tr = train_idx[np.isin(y[train_idx], labels)]
    te = test_idx[np.isin(y[test_idx], labels)]
    print(f"\n===== {name} ({len(labels)} signs) =====")

    cv_acc = {k: [] for k in K_VALUES}
    for f_tr, f_va in group_kfold(person[tr], n_folds=4, always_train={"kaggle"}):
        model = KNNClassifier().fit(X[tr][f_tr], y[tr][f_tr])
        idx, dists = model.kneighbors(X[tr][f_va], max(K_VALUES))
        for k in K_VALUES:
            cv_acc[k].append(accuracy(y[tr][f_va], model.vote_neighbors(idx, dists, k)[0]))
    model = KNNClassifier().fit(X[tr], y[tr])
    idx, dists = model.kneighbors(X[tr], max(K_VALUES))
    train_acc = {k: accuracy(y[tr], model.vote_neighbors(idx, dists, k)[0]) for k in K_VALUES}

    print("K tuning (cross-validation by person: each fold hides 2 people):")
    for k in K_VALUES:
        print(f"  K={k:<3d} train={train_acc[k]:.4f}  cv={np.mean(cv_acc[k]):.4f}")
    best_k = max(K_VALUES, key=lambda k: np.mean(cv_acc[k]))
    print(f"Best K = {best_k}")

    model.k = best_k
    pred = model.predict(X[te])
    acc = accuracy(y[te], pred)
    print(f"TEST accuracy on unseen people: {acc:.4f}")

    if DEPTH_PAIR[0] in labels:
        pair_tr = tr[np.isin(y[tr], DEPTH_PAIR)]
        pair_model = KNNClassifier(5).fit(X_depth[pair_tr], y[pair_tr])
        fix = np.isin(pred, DEPTH_PAIR)
        pred[fix] = pair_model.predict(X_depth[te][fix])
        acc = accuracy(y[te], pred)
        print(f"TEST accuracy with the {'/'.join(DEPTH_PAIR)} depth specialist: {acc:.4f}")

    old = tr[person[tr] == "kaggle"]
    if len(old):
        old_model = KNNClassifier(best_k).fit(X[old], y[old])
        print(f"Old model (Kaggle only, 1 person) on the same unseen people: "
              f"{accuracy(y[te], old_model.predict(X[te])):.4f}")

    cm = confusion_matrix(y[te], pred, labels)
    print("Per-sign results:")
    for lab, p, r, f1, n in per_class_report(cm, labels):
        print(f"  {lab:<3}precision={p:.3f}  recall={r:.3f}  f1={f1:.3f}  n={n}")
    off = sorted(((cm[i, j], labels[i], labels[j]) for i in range(len(labels))
                  for j in range(len(labels)) if i != j and cm[i, j]), reverse=True)[:8]
    print("Most common confusions:", ", ".join(f"{t}->{p} ({c})" for c, t, p in off))

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [1, 1.4]})
    axes[0].plot(K_VALUES, [train_acc[k] for k in K_VALUES], "o-", label="Training accuracy")
    axes[0].plot(K_VALUES, [np.mean(cv_acc[k]) for k in K_VALUES], "s-", label="CV accuracy (unseen people)")
    axes[0].axvline(best_k, color="gray", ls="--", label=f"Best K = {best_k}")
    axes[0].set_xlabel("K (number of neighbours)")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"{name}: accuracy vs K")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title(f"{name}: confusion matrix on unseen people (acc {acc:.1%})")
    plt.tight_layout()
    plt.savefig(filename, dpi=130)
    plt.close()
    return best_k


letters = sorted(l for l in set(y) if l.isalpha())
k_letters = evaluate("Letters", letters, "knn_letters_results.png")
k_numbers = evaluate("Numbers", NUMBERS, "knn_numbers_results.png")

final = KNNClassifier(k_letters).fit(X, y)
final.save("knn_model.npz")
pair = np.isin(y, DEPTH_PAIR)
KNNClassifier(5).fit(X_depth[pair], y[pair]).save("knn_depth_pair_model.npz")
print(f"\nSaved knn_model.npz (all {len(y)} hands), knn_depth_pair_model.npz, "
      "knn_letters_results.png, knn_numbers_results.png")

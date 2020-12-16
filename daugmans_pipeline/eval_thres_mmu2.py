#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import numpy as np
from matplotlib import pyplot as plt


#------------------------------------------------------------------------------
#   Parameters
#------------------------------------------------------------------------------
DIST_MAT_FILE = "dist_mat_mmu2.npy"
THRESHOLDS = np.linspace(start=0.0, stop=1.0, num=100)
NUM_IMAGES = 5


#------------------------------------------------------------------------------
#   Main execution
#------------------------------------------------------------------------------
dist_mat = np.load(DIST_MAT_FILE)

ground_truth = np.zeros_like(dist_mat, dtype=int)
for i in range(ground_truth.shape[0]):
    for j in range(ground_truth.shape[1]):
        if i//NUM_IMAGES == j//NUM_IMAGES:
            ground_truth[i, j] = 1

accuracies, precisions, recalls, fscores = [], [], [], []
for threshold in THRESHOLDS:
    decision_map = (dist_mat<=threshold).astype(int)
    accuracy = (decision_map==ground_truth).sum() / ground_truth.size
    precision = (ground_truth*decision_map).sum() / decision_map.sum()
    recall = (ground_truth*decision_map).sum() / ground_truth.sum()
    fscore = 2*precision*recall / (precision+recall)
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    fscores.append(fscore)

print("Max fscore:", max(fscores))
print("Best threshold:", THRESHOLDS[fscores.index(max(fscores))])

plt.figure()
plt.plot(THRESHOLDS, accuracies, "-or")
plt.plot(THRESHOLDS, precisions, "-vb")
plt.plot(THRESHOLDS, recalls, "-*g")
plt.plot(THRESHOLDS, fscores, "-sc")
plt.legend(["accuracy", "precision", "recall", "fscore"])
plt.grid(True)
plt.show()
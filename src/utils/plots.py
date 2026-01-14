import matplotlib.pyplot as plt
import seaborn as sns




def plot_confusion(cm, labels, path):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.savefig(path)
    plt.close()
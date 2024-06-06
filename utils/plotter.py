import matplotlib.pyplot as plt


def plot_loss_training(loss_values, path):
    epochs = range(1, len(loss_values) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss_values, 'b', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper center')
    plt.savefig(path)
    plt.show()


def plot_loss_training_log(loss_values, path):
    epochs = range(1, len(loss_values) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss_values, 'b', label='Training loss')
    plt.yscale('log')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper center')
    plt.savefig(path)
    plt.show()

def plot_roc_curve(fpr, tpr, auc, path):
    plt.plot(fpr, tpr, color='orange', label=f'AUROC: {auc:.4f}')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig(path)
    plt.show()


def plot_pr_curve(precision, recall, path):
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(path)
    plt.show()


def pr_auc(y_test, y_pred):

    return auc_score

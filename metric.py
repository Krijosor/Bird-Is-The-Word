import torch

# Small number to prevent division by 0
epsilon = 1e-8

"""
Calculates Precision
- TP = True Positive
- FP = False Positive
"""
def precision(TP, FP):
    return TP / (TP + FP + epsilon)

"""
Calculates Recall
- TP = True Positive
- FP = False Positive
"""
def recall(TP, FN):
    return TP / (TP + FN + epsilon)

"""
Calculates weighted F1 Score
- predicted = Model Predictions
- true_labels = Ground Truth
The calculation is weighted so that the wrong codes are penalized more than missing ones
"""
def weighted_f1(predicted, target, n_classes):

    # TODO: Make sure One-hot encoding is correct for use case
    # One-Hot encode the parameters
    predicted = torch.nn.functional.one_hot(predicted, n_classes)
    target = torch.nn.functional.one_hot(target, n_classes)

    # Calculate TP, FP, FN, TN:
    TP = (predicted & target).sum(dim=0).float()
    FP = (predicted & (~target)).sum(dim=0).float()
    FN = ((~predicted) & target).sum(dim=0).float()

    return (1.25*(precision(TP=TP, FP=FP)*recall(TP=TP, FN=FN))) / (0.25*(precision(TP=TP, FP=FP) + recall(TP=TP, FN=FN)) + epsilon)


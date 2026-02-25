import warnings
from typing import Dict

import numpy as np
import torch

# Import necessary metrics from sklearn for evaluation
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")


class ClassificationBenchmark:
    def __init__(self, dataloader, model):
        self.target = []
        self.pred = []
        self.pred_proba = []
        self.incorretly_classified_texts = []
        model.eval()
        # Determine device from model parameters; default to cpu
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        with torch.no_grad():
            for batch in dataloader:
                # NEW BERT DATALOADER HANDLING
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                y_batch = batch[2].to(device)
                #texts_batch = batch[3]

                # NEW BERT MODEL
                logits, _ = model(input_ids, attention_mask)

                # Move tensors to CPU and detach before converting to numpy
                self.target.extend(y_batch.detach().cpu().numpy())
                self.pred.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
                self.pred_proba.extend(logits.detach().cpu().numpy())
                
                # Collect incorrectly classified texts
              #  incorrect_mask = (torch.argmax(logits, dim=1) != y_batch).cpu().numpy()
              #  self.incorretly_classified_texts.extend([{'text': text, 'predicted': pred.item(), 'actual': actual.item()} for text, incorrect, pred, actual in zip(texts_batch, incorrect_mask, torch.argmax(logits, dim=1), y_batch) if incorrect])
                

        self.target = np.array(self.target)
        self.pred = np.array(self.pred)
        self.pred_proba = np.array(self.pred_proba)

    def accuracy(self):
        # Compute the accuracy: proportion of correct predictions.
        return np.mean(self.target == self.pred)

    def precision(self, average="macro"):
        # Compute precision: TP / (TP + FP), averaged across classes.
        return precision_score(self.target, self.pred, average=average)

    def recall(self, average="macro"):
        # Compute recall: TP / (TP + FN), averaged across classes.
        return recall_score(self.target, self.pred, average=average)

    def F1(self, average="macro"):
        # Compute F1 score: harmonic mean of precision and recall, averaged across classes.
        return f1_score(self.target, self.pred, average=average)

    def informedness(self):
        # Compute informedness (Youden's J statistic): mean of (recall + specificity) - 1.
        mc_matrix = multilabel_confusion_matrix(self.target, self.pred)

        recall = mc_matrix[:, 1, 1] / (mc_matrix[:, 1, 1] + mc_matrix[:, 1, 0])
        specificity = mc_matrix[:, 0, 0] / (mc_matrix[:, 0, 0] + mc_matrix[:, 0, 1])

        return np.mean(recall + specificity) - 1

    def markedness(self):
        # Compute markedness: mean of (precision + NPV) - 1.
        # NPV is Negative Predictive Value.
        mc_matrix = multilabel_confusion_matrix(self.target, self.pred)

        precision = mc_matrix[:, 1, 1] / (mc_matrix[:, 1, 1] + mc_matrix[:, 0, 1])
        npv = mc_matrix[:, 0, 0] / (mc_matrix[:, 0, 0] + mc_matrix[:, 1, 0])

        return np.mean(precision + npv) - 1

    def matthews(self):
        # Compute Matthews Correlation Coefficient (MCC).
        return matthews_corrcoef(self.target, self.pred)

    def confusion_matrix(self):
        # Compute confusion matrix.
        return confusion_matrix(self.target, self.pred)


def evaluate_classification(dataloader, model) -> Dict[str, float]:
    # Evaluate a classification model using the ClassificationBenchmark class.
    bench = ClassificationBenchmark(dataloader, model)

    return {
        "accuracy": bench.accuracy(),
        "precision": bench.precision(),
        "recall": bench.recall(),
        "f1": bench.F1(),
        "informedness": bench.informedness(),
        "markedness": bench.markedness(),
        "matthews_corrcoef": bench.matthews(),
        "confusion_matrix": bench.confusion_matrix()  # Convert to list for JSON serialization
    }
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)
import pandas as pd
import pickle
import seaborn as sns


class Evaluator():

    def visualize_confusion_matrix(self, y_true: List[int], y_pred: List[int], labels: List[str], save_dir: str, suffix: str) -> None:
        """
        This function visualizes the confusion matrix for the given true and predicted labels.
        It saves the confusion matrix as an image and a pickle file in the specified directory.
        
        Args:
        * y_true (List[int]): a list of true labels.
        * y_pred (List[int]): a list of predicted labels.
        * labels (List[str]): a list of labels corresponding to the true and predicted labels.
        * save_dir (str): a directory to save the confusion matrix image and pickle file.
        * suffix (str): a suffix to append to the saved file names.
        
        Returns:
        * None
        """
        # cm_labels = [f's{i+1}: {labels[i]}' for i in range(len(labels))]
        cm_labels = [f's{i+1}' for i in range(len(labels))]
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))).astype(float)
        cm_df = pd.DataFrame(cm, index=cm_labels, columns=cm_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, annot=False, cmap='Blues')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.tick_params(axis='y', rotation=0)
        plt.tight_layout()
        plt.savefig(save_dir / f'cm_{suffix}.png')
        plt.close()
 
        #cm /= cm.sum(axis=1, keepdims=True)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
        with open(save_dir / f'cm_{suffix}.pkl', 'wb') as f:
            pickle.dump(cm, f)

        cm_df = pd.DataFrame(cm, index=cm_labels, columns=cm_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='.2f')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.tick_params(axis='y', rotation=0)
        plt.tight_layout()
        plt.savefig(save_dir / f'cm_{suffix}_normalized.png')
        plt.close()
    
    def frame_level_metrics(self, y_true: List[int], y_pred: List[int], save_fp: Optional[str]=None) -> Tuple[float, float]:
        """
        This function computes time frame-level metrics for a given set of true and predicted labels.
        The metrics include accuracy, recall, precision, and F1 score for all classes combined, which are returned as a tuple.

        Args:
        * y_true_series (List[int]): a list of true labels.
        * y_pred_series (List[int]): a list of predicted labels.
        * save_fp (str): a file path to save the metrics. If None, the metrics are not saved.

        Returns:
        * all_accuracy (float): a float value of the overall accuracy score.
        * all_f1 (float): a float value of the overall macro F1 score.
        """

        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        if save_fp is not None:
            with open(save_fp, 'w') as f:
                f.write(f'Number of Frames: {len(y_true)}\n')
                f.write(f'Accuracy: {accuracy}\n')
                f.write(f'Precision: {precision}\n')
                f.write(f'Recall: {recall}\n')
                f.write(f'F1: {f1}\n')

        return accuracy, f1

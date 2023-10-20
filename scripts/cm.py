import numpy as np

class ConfusionMatrix:
    def __init__(self):
        self.classes = [1, 0, 2]  # [True, False, Neutral]
        self.confusion_matrix = np.zeros((3, 3))
        
    def update(self, y_true, y_pred):
        # Convert string labels to numbers
        y_true = { 'T': 1, 'F': 0, 'N': 2 }[y_true]
        y_pred = { 'T': 1, 'F': 0, 'N': 2 }[y_pred]

        idx_true = self.classes.index(y_true)
        idx_pred = self.classes.index(y_pred)
        self.confusion_matrix[idx_true][idx_pred] += 1
    
    def precision(self, label):
        idx = self.classes.index(label)
        column_sum = np.sum(self.confusion_matrix[:, idx])
        if column_sum == 0:
            return 0
        return self.confusion_matrix[idx, idx] / column_sum
    
    def recall(self, label):
        idx = self.classes.index(label)
        row_sum = np.sum(self.confusion_matrix[idx, :])
        if row_sum == 0:
            return 0
        return self.confusion_matrix[idx, idx] / row_sum
    
    def f1_score(self, label):
        p = self.precision(label)
        r = self.recall(label)
        if p + r == 0:
            return 0
        return 2 * (p * r) / (p + r)
    
    def micro_precision(self):
        total_tp = np.trace(self.confusion_matrix)
        total_fp = np.sum(np.sum(self.confusion_matrix, axis=0) - np.diagonal(self.confusion_matrix))
        return total_tp / (total_tp + total_fp)

    def micro_recall(self):
        total_tp = np.trace(self.confusion_matrix)
        total_fn = np.sum(np.sum(self.confusion_matrix, axis=1) - np.diagonal(self.confusion_matrix))
        return total_tp / (total_tp + total_fn)

    def micro_f1_score(self):
        micro_prec = self.micro_precision()
        micro_rec = self.micro_recall()
        if micro_prec + micro_rec == 0:  # Avoid dividing by zero
            return 0
        return 2 * (micro_prec * micro_rec) / (micro_prec + micro_rec)

    def summary(self):
        metrics = {}
        for label in self.classes:
            label_str = str(label)
            metrics[f"precision_{label_str}"] = self.precision(label)
            metrics[f"recall_{label_str}"] = self.recall(label)
            metrics[f"f1_{label_str}"] = self.f1_score(label)

        return metrics
    
    def table(self):
        # Print the column header
        print("Predicted -->".rjust(12), end='')
        for cls in self.classes:
            print(f'{cls}'.center(8), end='')
        print("\n" + "-"*38)
        
        # Print the rows
        for idx, cls in enumerate(self.classes):
            print(f"Actual {cls} |".rjust(12), end='')
            for value in self.confusion_matrix[idx]:
                print(f'{int(value)}'.center(8), end='')
            print()

if __name__ == "__main__":
    cm = ConfusionMatrix()

    cm.confusion_matrix[0][0] = 65  
    cm.confusion_matrix[0][1] = 47
    cm.confusion_matrix[0][2] = 317

    cm.confusion_matrix[1][0] = 46
    cm.confusion_matrix[1][1] = 76
    cm.confusion_matrix[1][2] = 308

    cm.confusion_matrix[2][0] = 20
    cm.confusion_matrix[2][1] = 59
    cm.confusion_matrix[2][2] = 262

    cm.table()

    print(cm.summary())
    print("Micro precision: " + str(cm.micro_precision()))
    print("Micro recall: " + str(cm.micro_recall()))
    print("Micro F1 score: " + str(cm.micro_f1_score()))
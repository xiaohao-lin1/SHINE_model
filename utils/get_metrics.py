from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def get_metrics(y_pred, y_true):
    # print(f'y_pred: {y_pred}')
    # print(f'y_true: {y_true}')
    acc = round(accuracy_score(y_true, y_pred), 4)
    f1= round(f1_score(y_true, y_pred), 4)
    cm = confusion_matrix(y_true, y_pred) #if classes is None else confusion_matrix(y_true, y_pred, labels=classes)
    return acc, f1, cm #





if __name__ == '__main__':
    #
    # get_metrics()
    pass

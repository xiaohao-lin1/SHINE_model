# import pandas as pd
# import numpy as np
#
#
# def get_average_confusion_matrix(cm_name, dataframe):
#     # Use the sum_confusion_matrices function
#     cms_string = dataframe.loc[cm_name].values
#     total_cm = sum_confusion_matrices(cms_string)
#
#     # Calculate the average
#     average_cm = total_cm / len(cms_string)
#
#     return average_cm
#
#
# # Using the function:
#
#
#
# def sum_confusion_matrices(cms_string):
#     # Convert string to 2D array
#     def string_to_matrix(s):
#         clean_string = s.replace('[', '').replace(']', '').strip()
#         return np.array([list(map(int, row.split())) for row in clean_string.split('\n')])
#
#     # Initialize an empty matrix for the total sum
#     # total_cm = np.zeros_like(string_to_matrix(cms_string[0]))
#     total_cm = np.empty_like(cms_string[0])
#     cms_string[-2] =cms_string[-1]
#     cms_string = cms_string[:-1]
#     # Iterate over the string representations and sum them up
#     for cm_str in cms_string:
#
#
#         #numpy.core._exceptions.UFuncTypeError: Cannot cast ufunc 'add' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'
#         #what should i do?
#
#
#         if type(cm_str) == 'float64':
#             continue
#         else:
#             total_cm += cm_str
#
#         # total_cm += string_to_matrix(cm_str)
#
#     return total_cm
#
# def get_average_confusion_matrix_for_train_val_test(dataframe):
#     train_cm_average = get_average_confusion_matrix('train_cm', dataframe)
#     val_cm_average = get_average_confusion_matrix('val_cm', dataframe)
#     test_cm_average = get_average_confusion_matrix('test_cm', dataframe)
#     return train_cm_average, val_cm_average, test_cm_average
#
#
# if __name__ == '__main__':
#     file = '../results/DCN_new_subject_dependent_10_fold_cross_validation_ku54_62chan_None/exp_data/sub0_DCN_new_subject_dependent_10_fold_cross_validation_ku54_62chan_None_session1.csv'
#     dataframe = pd.read_csv(file, index_col=0)
#     average_cm =  get_average_confusion_matrix(cm_name='train_cm', dataframe=dataframe)
#
#     print(average_cm)# test get_average_confusion_matrix
#
import pandas as pd
import numpy as np

def string_to_matrix(s):
    clean_string = s.replace('[', '').replace(']', '').strip()
    # Convert each row to a list of floats
    return np.array([list(map(float, row.split())) for row in clean_string.split('\n')])

def sum_confusion_matrices(cms_string):
    # Initialize an empty matrix for the total sum
    total_cm = None

    for cm_str in cms_string:
        if isinstance(cm_str, str):  # Check if cm_str is a string
            cm_matrix = string_to_matrix(cm_str)  # Convert to matrix
            if total_cm is None:
                total_cm = cm_matrix  # Initialize the total_cm with the first matrix
            else:
                total_cm += cm_matrix  # Add matrices
        else:
            continue

    return total_cm

#todo: debug the confusion matrix
def get_average_confusion_matrix(cm_name, dataframe):
    cms_string = dataframe.loc[cm_name].values
    total_cm = sum_confusion_matrices(cms_string)
    average_cm = total_cm / len(cms_string)
    return average_cm

def get_average_confusion_matrix_for_train_val_test(dataframe):
    train_cm_average = get_average_confusion_matrix('train_cm', dataframe)
    val_cm_average = get_average_confusion_matrix('val_cm', dataframe)
    test_cm_average = get_average_confusion_matrix('test_cm', dataframe)
    return train_cm_average, val_cm_average, test_cm_average

if __name__ == '__main__':
    file = '../results/DCN_new_subject_dependent_10_fold_cross_validation_ku54_62chan_None/exp_data/sub0_DCN_new_subject_dependent_10_fold_cross_validation_ku54_62chan_None_session1.csv'

    dataframe = pd.read_csv(file, index_col=0)
    # get the df except last column
    average_cm = get_average_confusion_matrix(cm_name='train_cm', dataframe=dataframe)

    print(average_cm)  # Test get_average_confusion_matrix

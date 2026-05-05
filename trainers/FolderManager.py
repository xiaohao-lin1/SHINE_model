import re
import os
import glob

import numpy as np
import pandas as pd

from utils.get_average_confusion_matrix import get_average_confusion_matrix
import datetime

import ast

import re
import numpy as np

def average_confusion_matrices(series):
    """
    series: pd.Series of string-encoded confusion matrices like
            '[[783 567]\n [612 701]]'
    """
    cms = []

    for v in series.dropna():
        # extract all integers
        numbers = list(map(int, re.findall(r"-?\d+", v)))

        # infer square matrix size
        n = int(len(numbers) ** 0.5)
        cm = np.array(numbers).reshape(n, n)

        cms.append(cm)

    return np.mean(cms, axis=0).round(2)



def export_to_latex_table(combined_all_file, method_name, dataset_name):
    '''

    :param dataset_name:
    :param combined_all:
    :return:
    '''
    # 1. set up the column names: Method, Mean (SD), F! (SD), Median, Range (MAx - Min), Parameters
    # 2  using the combined_all_file, get the values for each column
    # 3. save to a csv file

    # 1
    column_names = ['Method', 'Mean Acc (\u00B1SD)', 'F1 (\u00B1SD)', 'Median', 'Dataset']
    df = pd.DataFrame(columns=column_names)
    df['Method'] = [method_name]
    # 2
    # combined_all_file = pd.read_csv(combined_all_file, index_col=0)
    # df['Mean Acc (\u00B1SD)'] = [f'{combined_all_file["test_acc"]:.2f} ({combined_all_file["mean_sd"]:.2f})']
    # df['F1 (\u00B1SD)'] = [f'{combined_all_file["test_f1"]:.2f} ({combined_all_file["f1_sd"]:.2f})']
    df['Mean Acc \u00B1SD (%)'] = [f'{combined_all_file["test_acc"]:.2f} \u00B1 {combined_all_file["mean_sd"]:.2f}']
    df['F1 \u00B1SD (%)'] = [f'{combined_all_file["test_f1"]:.2f} \u00B1 {combined_all_file["f1_sd"]:.2f}']
    df['Median'] = [f'{combined_all_file["median"]:.2f}']
    df['25th Percentile'] = [f'{combined_all_file["25th Percentile"]:.2f}']
    df['75th Percentile'] = [f'{combined_all_file["75th Percentile"]:.2f}']
    df['Interquartile Range'] = [f'{combined_all_file["Interquartile Range"]:.2f}']
    # df['Range (Max - Min)'] = [f'{combined_all_file["Interquantile Range"]:.2f} ({combined_all_file["75th Percentile"]:.2f} - {combined_all_file["25th Percentile"]:.2f})']
    # df['Range (Max - Min)'] = [f'{combined_all_file["range"]:.2f} ({combined_all_file["max"]:.2f} - {combined_all_file["min"]:.2f})']
    # df['Parameters'] = [f'{int(combined_all_file["params"])}']
    df['Dataset'] = dataset_name
    # df = df.sort_values(by='Mean Accuracy (\u00B1SD)', ascending=True)

    # sort the df in acsending order of mean accuracy

    return df


class FolderManager:
    def __init__(self, folder_name, results_path):
        self.folder_name = folder_name
        self.full_folder_path = os.path.join(results_path, folder_name)

        self.LOSS_CURVES = self.get_saving_paths(folder_name_of_the_data_saved='loss_curves')
        self.EXP_DATA = self.get_saving_paths(folder_name_of_the_data_saved='exp_data')
        self.MODELS_WEIGHTS = self.get_saving_paths(folder_name_of_the_data_saved='model_weights')
        self.LOGGER_FOLDER = self.get_saving_paths(folder_name_of_the_data_saved='logger')
        self.ACC_LOSS = self.get_saving_paths(folder_name_of_the_data_saved='acc_and_loss')

        self.FINETUNE_LOSS_CURVES = self.get_saving_paths(folder_name_of_the_data_saved='finetune_loss_curves')
        self.FINETUNE_DATA = self.get_saving_paths(folder_name_of_the_data_saved='finetune_data')
        self.FINETUNE_MODELS_WEIGHTS = self.get_saving_paths(folder_name_of_the_data_saved='finetune_model_weights')
        self.FINETUNE_LOGGER_FOLDER = self.get_saving_paths(folder_name_of_the_data_saved='finetune_logger')

    def get_saving_paths(self, folder_name_of_the_data_saved):
        path = os.path.join(self.full_folder_path, folder_name_of_the_data_saved)
        self.get_safe_dir(path)
        return path

    def get_safe_dir(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def save_results_to_csv(self, fold, train_results_pdseries, test_results_pdseries, folder_name, exp_data):
        '''
        :param fold:
        :param train_results_pdseries:
        :param test_results_pdseries:
        :param folder_name:
        :param exp_data:
        :return:
        '''

        combined_results = pd.DataFrame(pd.concat([train_results_pdseries, test_results_pdseries]))
        sub_save_name = f'sub{fold}'
        sub_save_path = os.path.join(exp_data, sub_save_name)
        combined_results.to_csv(f'{sub_save_path}.csv')

    def get_average_results_over_10folds(self, dataframe:pd.DataFrame):
        dataframe_numeric = dataframe.apply(pd.to_numeric, errors='coerce')
        dataframe_row_means = dataframe_numeric.mean(axis=1, numeric_only=True).round(4)
        return dataframe_row_means

    def get_results_from_specific_fold(self, dataframe:pd.DataFrame, fold_no=0):
        specific_fold_results = dataframe.iloc[:, fold_no]
        specific_fold_results = specific_fold_results.apply(pd.to_numeric, errors='coerce')
        specific_fold_results = specific_fold_results.round(4)
        return specific_fold_results


    def combine_all_csv_files(self, folder_path, suffix, more_than_1_session=False, session_no=None):
        # List all files in the folder with the pattern 'sub*_*.csv'
        all_files = glob.glob(os.path.join(folder_path, f'sub*.csv'))
        if more_than_1_session:
            assert session_no is not None
            all_files = [file for file in all_files if f'session{session_no}' in file]

        def get_file_number(file_path):
            file_name = os.path.basename(file_path)
            match = re.search(r'sub(\d+)', file_name)
            return int(match.group(1)) if match else 0

        sorted_files = sorted(all_files, key=get_file_number)

        # Initialize an empty DataFrame to store concatenated columns
        print(f'Processing file for Fold 1')
        all_df = pd.read_csv(sorted_files[0], header=None)
        # Loop through each file and read the contents, then concatenate the columns
        for i, file in enumerate(sorted_files[1:]):
            print(f'Processing file for {file}')
            df = pd.read_csv(file, header=None).iloc[:, -1]
            all_df = pd.concat([all_df, df], axis=1)

        # Save the concatenated columns to 'all_*.csv'
        all_csv_file_name = f'all.csv' if not more_than_1_session else f'all_session{session_no}_{suffix}.csv'
        all_suffix_path = os.path.join(folder_path, all_csv_file_name)
        all_df.iloc[0] = list(range(len(sorted_files) + 1))
        all_df.columns = list(range(len(sorted_files) + 1))
        all_df.to_csv(all_suffix_path, index=False, header=False)

        df = all_df
        df.columns = df.iloc[0]

        # Drop the first row
        df.drop(df.index[0], axis=0, inplace=True)

        # Reset the index
        df.reset_index(drop=True, inplace=True)

        df = df.T
        df.columns = df.iloc[0]

        # Drop the first row
        df.drop(df.index[0], axis=0, inplace=True)

        # Reset the index
        df.reset_index(drop=True, inplace=True)

        all_df_numeric = df.apply(pd.to_numeric, errors='ignore')
        #get test acc only
        test_acc_only_df = pd.DataFrame()
        test_acc_only_df['test_acc'] = all_df_numeric['test_acc']
        test_acc_only_df.to_csv(os.path.join(folder_path, 'test_acc_only.csv'), index=False, header=True)
        #get test f1 only
        test_f1_only_df = pd.DataFrame()
        test_f1_only_df['test_f1'] = all_df_numeric['test_f1']
        test_f1_only_df.to_csv(os.path.join(folder_path, 'test_f1_only.csv'), index=False, header=True)

        mean_acc = all_df_numeric['test_acc'].mean()
        mean_sd = all_df_numeric['test_acc'].std()
        f1_sd = all_df_numeric['test_f1'].std()
        median = all_df_numeric['test_acc'].median()
        _25th_percentile = all_df_numeric['test_acc'].quantile(0.25)
        _75th_percentile = all_df_numeric['test_acc'].quantile(0.75)
        iqr = _75th_percentile - _25th_percentile
        # wilcoxon =
        # max = all_df_numeric['test_acc'].max()
        # min = all_df_numeric['test_acc'].min()
        # range_ = max - min


        # combined_all = pd.DataFrame()
        numeric_df = all_df_numeric.select_dtypes(include=[np.number])

        combined_all = numeric_df.mean().round(4)
        combined_all['test_acc'] = mean_acc
        combined_all['mean_sd'] = mean_sd
        combined_all['f1_sd'] = f1_sd
        combined_all['median'] = median
        combined_all['25th Percentile'] = _25th_percentile
        combined_all['75th Percentile'] = _75th_percentile
        combined_all['Interquartile Range'] = iqr
        # combined_all['max'] = max
        # combined_all['min'] = min
        # combined_all['range'] = range_


        combined_all['epoch'] = combined_all['epoch']/100
        if 'params' in combined_all.index:
            combined_all['params']  = combined_all['params']/100
        else:
            combined_all['params'] = 0
        # combined_all['params'] /= 100 if 'params' in combined_all.index else 0

        combined_all *= 100

        # Save the mean values to 'combined_all.csv'
        combined_all_csv_file_name = f'combined_all.csv' if not more_than_1_session else f'combined_all_session{session_no}.csv'
        combined_all_path = os.path.join(folder_path, combined_all_csv_file_name)

        combined_all.to_csv(combined_all_path, index=True, header=True)
        print(f'{"-"*10}{folder_path}{"-"*10}')
        print(combined_all)

        method_name = '_'.join(folder_path.split('/')[2].split('_')[:next(
            (i for i, part in enumerate(folder_path.split('/')[2].split('_')) if part == 'subject'), None)])
        dataset_name = '_'.join(
            part for part in folder_path.split('/')[2].split('_') if part.startswith(('Cho2017', 'ku54')))

        latex_df = export_to_latex_table(combined_all, method_name, dataset_name)
        latex_df.to_csv(os.path.join(folder_path, 'latex_table.csv'), index=False, header=True)


    def combine_latex_tables_from_various_methods(self, folder_path_list, save_path):
        for folder_path in folder_path_list:
            latex_table_path = os.path.join('../results/', folder_path, 'exp_data', 'latex_table.csv')
            df = pd.read_csv(latex_table_path)
            if folder_path == folder_path_list[0]:
                combined_df = df
            else:
                combined_df = pd.concat([combined_df, df])
        save_path = os.path.join(save_path, 'combined_latex_table.csv')
        combined_df.to_csv(save_path, index=False, header=True)
        return combined_df

    def combine_all_csv_files_for_sub_dependent_cv(self, folder_path, more_than_1_session=False, session_no=None, nfolds=None):

        # List all files in the folder with the pattern 'sub*_*.csv'
        #
        if os.path.exists(folder_path):
            print('Path exist')
        else:
            print('Path does not exist')
        all_files = glob.glob(os.path.join(folder_path, f'sub*.csv'))
        all_files = sorted(all_files, key=sort_key)

        if more_than_1_session:
            assert session_no is not None
            all_files = [file for file in all_files if f'session{session_no}' in file]

        # for each file, get the column named 'average_over_all_folds' and concatenate them together
        all_data = []
        for file in all_files:
            print(file)
            df = pd.read_csv(file, index_col=0)  # Assuming first column is the index column
            print('-------------df----------------')
            print(df)
            if 'average_over_all_folds' not in df.columns:
                # Convert the DataFrame to numeric, turning non-numeric values into NaN
                df_numeric = df.apply(pd.to_numeric, errors='coerce')

                # Drop the column '10'
                df_numeric = df_numeric.drop(columns=[str(nfolds)])

                # Calculate the mean
                df_numeric['average_over_all_folds'] = df_numeric.mean(axis=1)

                df['average_over_all_folds'] = df_numeric['average_over_all_folds']
                df.to_csv(file)

            all_data.append(df['average_over_all_folds'])

        concatenated_data = pd.concat(all_data, axis=1)
        concatenated_data.columns = list(range(1, len(all_files) + 1))
        concatenated_data.to_csv(os.path.join(folder_path, f'all_session_{session_no}.csv'))

        combined_results = self.get_average_results_over_10folds(concatenated_data)
        combined_results.to_csv(os.path.join(folder_path, f'combined_all_session_{session_no}.csv'))

        return concatenated_data, combined_results

    import os
    import glob
    import pandas as pd
    import numpy as np

    # def combine_all_csv_files_for_sub_dependent_finetune(self, folder_path, seeds_list, train_percent_list):
    #     """
    #     Combine per-subject CSVs (sub0.csv, sub1.csv, ...)
    #     Each CSV has columns: '1', '2', '3' (seed numbers)
    #
    #     Outputs:
    #       - all_subjects_seed1.csv
    #       - all_subjects_seed2.csv
    #       - all_subjects_seed3.csv
    #       - avg_all_seeds.csv
    #     """
    #
    #     assert os.path.exists(folder_path), f"Path does not exist: {folder_path}"
    #
    #     # --------------------------------------------------
    #     # Load all subject CSVs
    #     # --------------------------------------------------
    #     files = sorted(glob.glob(os.path.join(folder_path, "sub*.csv")))
    #     if len(files) == 0:
    #         raise RuntimeError("No sub*.csv files found")
    #
    #     subject_dfs = {}
    #     for f in files:
    #         subject_id = os.path.splitext(os.path.basename(f))[0]  # sub0, sub1, ...
    #         df = pd.read_csv(f, index_col=0)
    #
    #         # keep only numeric values (drops model_name, cm, etc.)
    #         df = df.apply(pd.to_numeric, errors="coerce")
    #
    #         subject_dfs[subject_id] = df
    #
    #     # --------------------------------------------------
    #     # Build per-seed aggregated CSVs
    #     # --------------------------------------------------
    #     seed_avg_values = pd.DataFrame(
    #         index=train_percent_list,
    #         columns=[str(s) for s in seeds_list]
    #     )
    #
    #     for train_percent in train_percent_list:
    #         for seed in seeds_list:
    #             seed = str(seed)
    #             per_subject_cols = []
    #
    #             for sub_id, df in subject_dfs.items():
    #                 if seed not in df.columns:
    #                     raise RuntimeError(f"Seed {seed} missing in {sub_id}")
    #
    #                 col = df[seed].rename(sub_id)
    #                 per_subject_cols.append(col)
    #
    #             # concat subjects horizontally
    #             seed_df = pd.concat(per_subject_cols, axis=1)
    #
    #             # row-wise avg across subjects
    #             seed_df["avg"] = seed_df.mean(axis=1)
    #
    #             # save
    #             out_path = os.path.join(folder_path, f"all_subjects_seed{seed}.csv")
    #             seed_df.to_csv(out_path)
    #
    #             # CORRECT indexing
    #             seed_avg_values.loc[train_percent, seed] = seed_df.loc['test_acc', 'avg']
    #
    #     # --------------------------------------------------
    #     # Build avg_all_seeds.csv
    #     # --------------------------------------------------
    #     # seed_avg_values['avg'] =
    #     seed_avg_values["avg"] = seed_avg_values[["1", "2", "3"]].mean(axis=1)
    #     seed_avg_values["sd"] = seed_avg_values[["1", "2", "3"]].std(axis=1)
    #     seed_avg_values.to_csv(
    #         os.path.join(folder_path, "avg_all_seeds.csv"), index=True
    #     )
    #
    #     return {
    #         "seed_averages": seed_avg_values,
    #         # "avg_all_seeds": avg_all_seeds_df,
    #     }
    def get_aggregate_results(self, name_of_result, train_percent_list, seeds_list, subject_dfs, folder_path):        # --------------------------------------------------
        # Build avg_all_seeds.csv (example: test_acc only)
        # --------------------------------------------------
        seed_aggregate_result = pd.DataFrame(
            index=train_percent_list,
            columns=[str(s) for s in seeds_list],
            dtype=float,
        )

        for seed in seeds_list:
            seed = str(seed)
            col_name = f"{seed}_{name_of_result}"

            per_subject_cols = []
            for df in subject_dfs.values():
                per_subject_cols.append(df[col_name])

            tmp = pd.concat(per_subject_cols, axis=1)
            seed_aggregate_result[seed] = tmp.mean(axis=1)

        seed_aggregate_result["avg"] = seed_aggregate_result.mean(axis=1)
        seed_aggregate_result["sd"] = seed_aggregate_result.std(axis=1)

        seed_aggregate_result.to_csv(
            os.path.join(folder_path, f"avg_{name_of_result}_all_seeds.csv"), index=True
        )

        return seed_aggregate_result


    def combine_all_csv_files_for_sub_dependent_finetune(
            self,
            folder_path,
            seeds_list,
            train_percent_list,
    ):
        METRICS_TO_AGG = [
            "test_acc",
            "train_acc",
            "val_acc",
            "test_f1",
            "train_f1",
            "val_f1",
        ]

        assert os.path.exists(folder_path), f"Path does not exist: {folder_path}"

        # --------------------------------------------------
        # Load all subject CSVs
        # --------------------------------------------------
        files = sorted(glob.glob(os.path.join(folder_path, "sub*.csv")))
        if len(files) == 0:
            raise RuntimeError("No sub*.csv files found")

        subject_dfs = {}
        for f in files:
            subject_id = os.path.splitext(os.path.basename(f))[0]
            df = pd.read_csv(f, index_col=0)

            # keep numeric only
            df = df.apply(pd.to_numeric, errors="coerce")
            subject_dfs[subject_id] = df

        # --------------------------------------------------
        # Aggregate per seed × metric
        # --------------------------------------------------
        for seed in seeds_list:
            seed = str(seed)

            for metric in METRICS_TO_AGG:
                col_name = f"{seed}_{metric}"

                per_subject_cols = []

                for sub_id, df in subject_dfs.items():
                    if col_name not in df.columns:
                        raise RuntimeError(f"{col_name} missing in {sub_id}")

                    per_subject_cols.append(df[col_name].rename(sub_id))

                # shape: (train_percent × subjects)
                metric_df = pd.concat(per_subject_cols, axis=1)

                # mean across subjects
                metric_df["avg"] = metric_df.mean(axis=1)
                metric_df["sd"] = metric_df.std(axis=1)

                out_path = os.path.join(
                    folder_path, f"all_subjects_seed{seed}_{metric}.csv"
                )
                metric_df.to_csv(out_path)

        # --------------------------------------------------
        # Build avg_all_seeds.csv (example: test_acc only)
        # --------------------------------------------------
        # seed_avg_test_acc = pd.DataFrame(
        #     index=train_percent_list,
        #     columns=[str(s) for s in seeds_list],
        #     dtype=float,
        # )
        #
        # for seed in seeds_list:
        #     seed = str(seed)
        #     col_name = f"{seed}_test_acc"
        #
        #     per_subject_cols = []
        #     for df in subject_dfs.values():
        #         per_subject_cols.append(df[col_name])
        #
        #     tmp = pd.concat(per_subject_cols, axis=1)
        #     seed_avg_test_acc[seed] = tmp.mean(axis=1)
        #
        # seed_avg_test_acc["avg"] = seed_avg_test_acc.mean(axis=1)
        # seed_avg_test_acc["sd"] = seed_avg_test_acc.std(axis=1)
        #
        # seed_avg_test_acc.to_csv(
        #     os.path.join(folder_path, "avg_all_seeds.csv"), index=True
        # )
        aggregate_test_acc = self.get_aggregate_results('test_acc', train_percent_list, seeds_list, subject_dfs, folder_path)
        aggregate_test_f1 = self.get_aggregate_results('test_f1', train_percent_list, seeds_list, subject_dfs, folder_path)

        return {
            "test_acc": aggregate_test_acc,
            "test_f1": aggregate_test_f1
        }

    def combine_all_csv_files_for_sub_indep(self, folder_path):
        """
        Combine subject-independent results.
        Each subX.csv contains ONE fold / ONE result per subject.
        """

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"{folder_path} does not exist")

        # ---- collect subject files ----
        all_files = glob.glob(os.path.join(folder_path, "sub*.csv"))
        all_files = sorted(all_files, key=sort_key)

        if len(all_files) == 0:
            raise RuntimeError("No sub*.csv files found")

        all_subject_data = []

        for file in all_files:
            print(f"Reading {file}")
            df = pd.read_csv(file, index_col=0)

            # Expect exactly ONE column
            if df.shape[1] != 1:
                raise ValueError(f"{file} has {df.shape[1]} columns, expected 1")

            all_subject_data.append(df.iloc[:, 0])

        # ---- concatenate subjects ----
        all_subjects_df = pd.concat(all_subject_data, axis=1)
        all_subjects_df.columns = [f"sub{i}" for i in range(len(all_subject_data))]

        all_subjects_path = os.path.join(folder_path, "all_subjects.csv")
        all_subjects_df.to_csv(all_subjects_path)
        print(f"Saved: {all_subjects_path}")

        # ---- numeric-only for statistics ----
        numeric_df = all_subjects_df.apply(pd.to_numeric, errors="coerce")

        # ---- compute statistics across subjects ----
        combined_all = numeric_df.mean(axis=1).round(4)

        combined_all["mean_sd"] = numeric_df.loc["test_acc"].std()
        combined_all["f1_sd"] = numeric_df.loc["test_f1"].std()
        combined_all["median"] = numeric_df.loc["test_acc"].median()
        combined_all["25th Percentile"] = numeric_df.loc["test_acc"].quantile(0.25)
        combined_all["75th Percentile"] = numeric_df.loc["test_acc"].quantile(0.75)
        combined_all["Interquartile Range"] = (
                combined_all["75th Percentile"] - combined_all["25th Percentile"]
        )

        combined_all["max"] = numeric_df.loc["test_acc"].max()
        combined_all["min"] = numeric_df.loc["test_acc"].min()
        combined_all["Range (Max–Min)"] = (
                combined_all["max"] - combined_all["min"]
        )

        # ---- averaged confusion matrices (STRING SAFE) ----
        train_cm_avg = average_confusion_matrices(all_subjects_df.loc["train_cm"])
        val_cm_avg = average_confusion_matrices(all_subjects_df.loc["val_cm"])
        test_cm_avg = average_confusion_matrices(all_subjects_df.loc["test_cm"])

        combined_all.loc["train_cm"] = str(train_cm_avg.tolist())
        combined_all.loc["val_cm"] = str(val_cm_avg.tolist())
        combined_all.loc["test_cm"] = str(test_cm_avg.tolist())

        # ---- scale numeric rows only ----
        numeric_rows = combined_all.index.difference(["train_cm", "val_cm", "test_cm"])
        combined_all.loc[numeric_rows] *= 100

        return all_subjects_df, combined_all

    def replacing_non_numeric_data_with_numeric_data(self, dataframe_row_means, original_dataframe, train_cm_avg, val_cm_avg, test_cm_avg):
        dataframe_row_means.loc['model_name'] = original_dataframe.loc['model_name'].values[0]
        dataframe_row_means.loc['train_cm'] = train_cm_avg
        dataframe_row_means.loc['val_cm'] = val_cm_avg
        dataframe_row_means.loc['test_cm'] = test_cm_avg
        return dataframe_row_means

    def get_average_results_over_10_folds_pipeline(self, combined_results, train_cm_avg, val_cm_avg, test_cm_avg):
        combined_results_row_means = self.get_average_results_over_10folds(combined_results)

        # Convert all columns to numeric where possible
        combined_results_row_means = self.replacing_non_numeric_data_with_numeric_data(combined_results_row_means, combined_results, train_cm_avg, val_cm_avg, test_cm_avg)
        #save to csv

        combined_results['average_over_all_folds'] = combined_results_row_means
        return combined_results

    def get_specific_fold_results(self, combined_results, train_cm_avg, val_cm_avg, test_cm_avg, fold_no=0):
        specific_fold_results = combined_results.iloc[:, fold_no]
        specific_fold_results = self.replacing_non_numeric_data_with_numeric_data(specific_fold_results, combined_results, train_cm_avg, val_cm_avg, test_cm_avg)
        specific_fold_results.columns = [f'fold_{fold_no}']
        return specific_fold_results

    def save_one_subject_data_for_sub_dependent_10fold_cv(self, subject_id, subject_training_results,
                                                          subject_test_results, exp_data, folder_name, session_no=None, func_to_get_results=get_average_results_over_10_folds_pipeline):
        # train_cm_avg = get_average_confusion_matrix('train_cm', subject_training_results)
        # val_cm_avg = get_average_confusion_matrix('val_cm', subject_training_results)
        # test_cm_avg = get_average_confusion_matrix('test_cm', subject_test_results)

        combined_results = pd.concat([subject_training_results, subject_test_results])

        # combined_results_row_means = self.get_average_results_over_10folds(combined_results)
        #
        # # Convert all columns to numeric where possible
        # combined_results_row_means = self.replacing_non_numeric_data_with_numeric_data(combined_results_row_means, combined_results, train_cm_avg, val_cm_avg, test_cm_avg)
        # #save to csv
        #
        # combined_results['average_over_all_folds'] = combined_results_row_means
        # combined_results = func_to_get_results(combined_results, train_cm_avg, val_cm_avg, test_cm_avg)

        sub_save_name = f'sub{subject_id}'
        sub_save_path = os.path.join(exp_data, sub_save_name)
        combined_results.to_csv(f'{sub_save_path}.csv')

        return combined_results

    # def save_one_subject_data_for_sub_dependent_seeded_finetune(self, subject_id, subject_training_results,
    #                                                       subject_test_results, exp_data):
    #
    #     # combined_results = pd.concat([subject_training_results, subject_test_results])
    #     combined_results = pd.concat(
    #         [subject_training_results, subject_test_results],
    #         axis=1
    #     )
    #
    #     sub_save_name = f'sub{subject_id}'
    #     sub_save_path = os.path.join(exp_data, sub_save_name)
    #     combined_results.to_csv(f'{sub_save_path}.csv')
    #
    #     return combined_results
    #
    def save_one_subject_data_for_sub_dependent_seeded_finetune(
            self,
            subject_id,
            subject_training_results,
            subject_test_results,
            exp_data,
    ):
        combined_results = pd.concat(
            [subject_training_results, subject_test_results],
            axis=1
        )

        # ---- column ordering logic ----
        seeds = ["1", "2", "3"]

        acc_order = ["test_acc", "train_acc", "val_acc"]
        f1_order = ["test_f1", "train_f1", "val_f1"]

        ordered_cols = []

        # accuracy first
        for metric in acc_order:
            for seed in seeds:
                col = f"{seed}_{metric}"
                if col in combined_results.columns:
                    ordered_cols.append(col)

        # f1 next
        for metric in f1_order:
            for seed in seeds:
                col = f"{seed}_{metric}"
                if col in combined_results.columns:
                    ordered_cols.append(col)

        # remaining columns (anything else)
        remaining_cols = [
            c for c in combined_results.columns if c not in ordered_cols
        ]

        combined_results = combined_results[ordered_cols + remaining_cols]
        # --------------------------------

        sub_save_name = f"sub{subject_id}"
        sub_save_path = os.path.join(exp_data, sub_save_name)
        combined_results.to_csv(f"{sub_save_path}.csv")

        return combined_results
    #
    def save_one_subject_data_for_sub_indep(self, subject_id, subject_training_results,
                                                          subject_test_results, exp_data):

        combined_results = pd.concat([subject_training_results, subject_test_results])


        sub_save_name = f'sub{subject_id}'
        sub_save_path = os.path.join(exp_data, sub_save_name)
        combined_results.to_csv(f'{sub_save_path}.csv')

        return combined_results

def combine_various_methods_results(folder_path_list, save_path, name_of_the_type_of_results):
    # name_of_the_type_of_results = 'latex_table'
    for folder_path in folder_path_list:
        latex_table_path = os.path.join('../results/', folder_path, 'exp_data', f'{name_of_the_type_of_results}.csv')
        df = pd.read_csv(latex_table_path)
        if folder_path == folder_path_list[0]:
            combined_df = df
        else:
            combined_df = pd.concat([combined_df, df])
    save_path = os.path.join(save_path, f'combined_{name_of_the_type_of_results}.csv')
    combined_df.to_csv(save_path, index=False, header=True)
    return combined_df

def combine_interquantile_range_results(folder_path_list, save_path):
    name_of_the_type_of_results = 'interquantile_range'
    combine_various_methods_results(folder_path_list, save_path, name_of_the_type_of_results)

def combine_test_acc_results(folder_path_list, save_path):
    name_of_the_type_of_results = 'test_acc'
    combine_various_methods_results(folder_path_list, save_path, name_of_the_type_of_results)

def combine_wilcoxon_results(folder_path_list, save_path):
    name_of_the_type_of_results = 'wilcoxon'
    combine_various_methods_results(folder_path_list, save_path, name_of_the_type_of_results)



def choose_subjects_w_highest_acc(all_subjects_csv_file, top_k_subjects=2):
    '''

    :param all_subjects_csv_file:
    :param top_k_subjects:
    :return:
    '''
    all_subjects_csv_file = pd.read_csv(all_subjects_csv_file)

    #make the first column the index and transpose for easier sorting
    all_subjects_csv_file.index = all_subjects_csv_file.iloc[:, 0]
    all_subjects_csv_file = all_subjects_csv_file.iloc[:, 1:]
    all_subjects_csv_file_transpose = all_subjects_csv_file.T

    #sort the dataframe by test_acc
    sorted_file = all_subjects_csv_file_transpose.sort_values(by='test_acc', ascending=False)
    #get the top k subjects
    top_subjects = sorted_file.index.tolist()[:top_k_subjects]
    top_subjects_index_list = [int(i) -1 for i in top_subjects]
    return top_subjects_index_list

def sort_key(s):
    match = re.search(r'sub(\d+)', s)
    if match:
        return int(match.group(1))
    return 0

def get_list_of_folder_names():
    """Reads the experiments file and returns a list of folder names."""
    date_str = datetime.datetime.now().strftime('%Y%m%d')
    file_path = f'../jobs/experiments_{date_str}.txt'

    with open(file_path, 'r') as file:
        folder_names = file.readlines()

    return [name.strip() for name in folder_names]


def add_folder_name_to_file(folder_name):
    """Appends the folder_name to the experiments file if it's not already present."""
    date_str = datetime.datetime.now().strftime('%d%m%y')  # Get current date in the format YYYYMMDD
    file_path = f'../job/experiments_{date_str}.txt'

    # Read existing folder names from the file
    with open(file_path, 'a+') as file:  # 'a+' mode lets us read and append
        file.seek(0)  # Move to the start of the file to read from the beginning
        existing_folders = file.readlines()
        existing_folders = [name.strip() for name in existing_folders]  # Remove newline characters

        # If folder_name is not in the file, append it
        if folder_name not in existing_folders:
            file.write(folder_name + '\n')


def get_dep_10fold_cv_results():
    results_path = '../results'

    dep_cv_folders_list = [
        # 'ATCNet_subject_dependent_10_fold_cross_validation_open_close',
        # 'ViT_new_subject_dependent_10_fold_cross_validation_open_close',
        # 'PatchTST_new_subject_dependent_10_fold_cross_validation_open_close',


    # 'DCN_new_epochs_100_subject_dependent_10_fold_cross_validation_open_close_muscle',
    #  'PatchTST_new_epochs_100_subject_dependent_10_fold_cross_validation_open_close_muscle',
    #  'ATCNet_epochs_100_subject_dependent_10_fold_cross_validation_open_close_muscle',
    #  'Eegnet_new_epochs_100_subject_dependent_10_fold_cross_validation_open_close_muscle',
    #  'ViT_new_epochs_100_subject_dependent_10_fold_cross_validation_open_close_muscle'
    # 'CASTNet_epochs_100_subject_dependent_10_fold_cross_validation_open_close'
        'FBCNet_new_batch_size_48_subject_dependent_10_fold_cross_validation_open_close_fb'
    ]

    for folder_name in dep_cv_folders_list:
        print(folder_name)
        exp_data = os.path.join(results_path, folder_name, 'exp_data')
        foldermananager = FolderManager(folder_name, results_path
                                        )
        foldermananager.combine_all_csv_files_for_sub_dependent_cv(exp_data)


def get_dep_5fold_cv_results():
    results_path = '../results'
    dep_cv_folders_list = [

        'DCN_new_batch_size_48_dep_5fold_cv_open_close',
        'Eegnet_new_batch_size_48_dep_5fold_cv_open_close',
        'FBCNet_new_batch_size_48_dep_5fold_cv_open_close_fb',
        'ATCNet_batch_size_48_dep_5fold_cv_open_close',
        'ViT_new_batch_size_48_dep_5fold_cv_open_close',
        'PatchTST_new_batch_size_48_dep_5fold_cv_open_close',
        # 'CASTNet_batch_size_48_dep_5fold_cv_open_close',
        'FAST_Tower_batch_size_48_dep_5fold_cv_open_close',

    ]


    for folder_name in dep_cv_folders_list:
        print(folder_name)
        exp_data = os.path.join(results_path, folder_name, 'exp_data')
        foldermananager = FolderManager(folder_name, results_path
                                        )
        foldermananager.combine_all_csv_files_for_sub_dependent_cv(exp_data, nfolds=5)

def get_finetune_cv_results():
    results_path = '../results'

    cv_folders_list = [
        # 'Eegnet_new_finetune_epochs_100_subject_dependent_10fold_cv_for_one_subject_load_from_subject_independent_open_close',
        'DCN_new_finetune_epochs_100_subject_dependent_10fold_cv_for_one_subject_load_from_subject_independent_open_close',
    ]

    for folder_name in cv_folders_list:
        print(folder_name)
        exp_data = os.path.join(results_path, folder_name, 'exp_data')
        foldermananager = FolderManager(folder_name, results_path
                                        )
        foldermananager.combine_all_csv_files_for_sub_dependent_cv(exp_data)
# def get_indep_cv_results():
#     results_path = '../results'
#
#     indep_cv_folders_list = [
#
#
#         # "Eegnet_new_subset_F_subject_independent_cv_load_and_test_pretrained_weights_open_close", # cv0
#         # "DCN_new_subset_F_subject_independent_cv_load_and_test_pretrained_weights_open_close", #cv9
#         # 'FBCNet_new_subset_F_subject_independent_cv_load_and_test_pretrained_weights_open_close_fb', #cv9
#         # 'ATCNet_subset_F_subject_independent_cv_open_close', #cv0
#         # 'ViT_new_subject_independent_cv_open_close', #cv6
#         # 'PatchTST_new_subject_independent_cv_open_close', #cv2
#         # 'CASTNet_subject_independent_cv_open_close', #cv8
#         'FAST_Tower_subject_independent_cv_open_close' #cv7
#     ]
#
#     for folder_name in indep_cv_folders_list:
#         exp_data = os.path.join(results_path, folder_name, 'exp_data')
#         # foldermananager = FolderManager(folder_name, results_path
#         #                                 )
#         combine_sub_indep_cv_results(folder_name, 10, results_path)

def get_each_fold_result_dep_cv(folder, n_subjects, results_folder='../results', fold=10):


    columns = [f"sub{sub}" for sub in range(n_subjects)]
    columns.append('mean')
    fold_df = pd.DataFrame(index=[f"fold{f}" for f in range(fold)], columns=columns)
    for sub_id in range(n_subjects):
        csv_file = os.path.join(results_folder, folder, 'exp_data', f'sub{sub_id}.csv')
        df = pd.read_csv(csv_file)
        df = df.drop(columns=['10'])
        df.index = df['Unnamed: 0']
#
        # print(f'{"-"*10}Subject {sub_id}{"-"*10}')
        for i in range(fold):
            fold_df.loc[f'fold{i}', f'sub{sub_id}'] = df[str(i)]['test_acc']
            # print(df[str(i)]['test_acc'])

    fold_df = fold_df.apply(pd.to_numeric, errors='coerce')
    for i in range(fold):
        fold_df.loc[f'fold{i}', 'mean'] = fold_df.loc[f'fold{i}', columns[:-1]].mean()

    print(f"{'_'.join(folder.split('_')[:2])} Fold {fold_df['mean'].argmax()} has the best mean accuracy {fold_df['mean'].max()}")
    print('Mean Accuracy')
    print(fold_df['mean'])
    # print(f'Fold {} has the best mean accuracy'.format(fold_df['mean'].argmax(), fold_df['mean'].max()))
    fold_df.to_csv(os.path.join(results_folder, folder, 'exp_data', 'each_fold_results.csv'))
    return fold_df['mean']

    # print(fold_df)



    # def combine_all_csv_files_for_sub_dependent_cv(folder_path, more_than_1_session=False, session_no=None):
    #
    #     # List all files in the folder with the pattern 'sub*_*.csv'
    #     #
    #     if os.input_path.exists(folder_path):
    #         print('Path exist')
    #     else:
    #         print('Path does not exist')
    #     all_files = glob.glob(os.input_path.join(folder_path, f'sub*.csv'))
    #     all_files = sorted(all_files, key=sort_key)
    #
    #     if more_than_1_session:
    #         assert session_no is not None
    #         all_files = [file for file in all_files if f'session{session_no}' in file]
    #
    #     # for each file, get the column named 'average_over_all_folds' and concatenate them together
    #     all_data = []
    #     for file in all_files:
    #         print(file)
    #         df = pd.read_csv(file, index_col=0)  # Assuming first column is the index column
    #
    #         if 'average_over_all_folds' not in df.columns:
    #             # Convert the DataFrame to numeric, turning non-numeric values into NaN
    #             df_numeric = df.apply(pd.to_numeric, errors='coerce')
    #
    #             # Drop the column '10'
    #             df_numeric = df_numeric.drop(columns=['10'])
    #
    #             # Calculate the mean
    #             df_numeric['average_over_all_folds'] = df_numeric.mean(axis=1)
    #
    #             df['average_over_all_folds'] = df_numeric['average_over_all_folds']
    #             df.to_csv(file)
    #
    #         all_data.append(df['average_over_all_folds'])
    #
    #
    #
    #
    #
    #
    #     concatenated_data = pd.concat(all_data, axis=1)
    #     concatenated_data.columns = list(range(1, len(all_files) + 1))
    #     concatenated_data.to_csv(os.input_path.join(folder_path, f'all_session_{session_no}.csv'))
    #
    #     # combined_results = self.get_average_results_over_10folds(concatenated_data)
    #     # combined_results.to_csv(os.input_path.join(folder_path, f'combined_all_session_{session_no}.csv'))
    #
    #     return concatenated_data, combined_results

def get_all_models_all_fold_results(fold_mean_list, results_folder='../results'):
    index = [f"fold{f}" for f in range(10)]
    index.append('max')
    index.append('min')
    index.append('max-min')

    columns = ['Eegnet', 'DCN', 'FBCNet', 'ATCNet', 'ViT', 'PatchTST']
    all_folds_df = pd.DataFrame(index=index, columns=columns)

    # get the max and min
    max_acc = [fold.max() for fold in fold_mean_list]
    min_acc = [fold.min() for fold in fold_mean_list]
    max_min = [max_ - min_ for max_, min_ in zip(max_acc, min_acc)]

    for i in range(len(fold_mean_list)):
        fold_data = fold_mean_list[i].append(pd.Series([max_acc[i], min_acc[i], max_min[i]], index=['max', 'min', 'max-min']))
        all_folds_df[columns[i]] = fold_data

    # all_folds_df['Eegnet'] = fold_mean_list[0]
    # all_folds_df['DCN'] = fold_mean_list[1]
    # all_folds_df['FBCNet'] = fold_mean_list[2]
    # all_folds_df['ATCNet'] = fold_mean_list[3]
    # all_folds_df['ViT'] = fold_mean_list[4]
    # all_folds_df['PatchTST'] = fold_mean_list[5]
    #


    # for idx_of_rearranged_chan, fold_mean in enumerate(fold_mean_list):
    #     all_folds_df[columns[idx_of_rearranged_chan]] = fold_mean


    # all_folds_df.to_csv(os.input_path.join(results_folder, 'all_models_all_folds.csv')
    save_folder = os.path.join(results_folder, 'all_models_all_folds')
    if not os.path.exists(save_folder):

        os.makedirs(save_folder)
    all_folds_df.to_csv(os.path.join(save_folder, 'all_models_all_folds.csv'))
        # all_folds_df[] = fold_mean[0]



if __name__ == '__main__':
    folder_path = 'SHINE_benchmark_epochs_1_indep_nfolds_w_epoch_5_patients_rest_close_ica'
    results_folder = '../results'
    folder_manager = FolderManager(folder_path, results_folder)
    folder_manager.combine_all_csv_files_for_sub_indep(folder_manager.EXP_DATA)
    # csv_file = "/mnt/data/stulinxh/results/Eegnet_new_epochs_100_subject_dependent_10_fold_cross_validation_open_close/exp_data/sub0_Eegnet_new_epochs_100_subject_dependent_10_fold_cross_validation_open_close_session1.csv"

    # folders_list = [
    #     # 'Eegnet_new_epochs_100_subject_dependent_10_fold_cross_validation_open_close',
    #     # 'DCN_new_epochs_100_subject_dependent_10_fold_cross_validation_open_close',
    #
    #     # 'FBCNet_new_epochs_100_subject_dependent_10_fold_cross_validation_open_close_fb',
    #     # 'ATCNet_subject_dependent_10_fold_cross_validation_open_close',
    #     # 'ViT_new_subject_dependent_10_fold_cross_validation_open_close',
    #     # 'PatchTST_new_subject_dependent_10_fold_cross_validation_open_close',
    #
    #
    #     ]
    #
    # n_subjects = 50
    #
    # fold_mean_list = []
    # for folder in folders_list:
    #     fold_df = get_each_fold_result_dep_cv(folder, n_subjects, results_folder='../results', fold=10)
    #     fold_mean_list.append(fold_df)
    #
    # get_all_models_all_fold_results(fold_mean_list, results_folder='../results')
    # dash = '-' * 30
    # results_of_choice = [
    #     'Eegnet_new_epochs_100_batch_size_16_learning_rate_0.001_device_6_subject_dependent_holdout_open_close',
    #     'DCN_new_epochs_100_batch_size_16_learning_rate_0.001_device_5_subject_dependent_holdout_open_close',
    #
    # ]
    # # folder_path_list = ['DCN_split_spatial_kernel_3_layers_cross_after_temp_conv_learned_top_k_share_temp_conv_n_blk2_3_thresholds_distance_metric_euclidean_distance_top_k_channels_1_deterministic_T_subject_independent_all_folds_ku54_62chan_left_right_hemi', 'DCN_split_spatial_kernel_3_layers_cross_after_temp_conv_learned_top_k_share_temp_conv_n_blk2_3_thresholds_distance_metric_euclidean_distance_top_k_channels_2_deterministic_T_subject_independent_all_folds_ku54_62chan_left_right_hemi', 'DCN_split_spatial_kernel_3_layers_cross_after_temp_conv_learned_top_k_share_temp_conv_n_blk2_3_thresholds_distance_metric_euclidean_distance_top_k_channels_3_deterministic_T_subject_independent_all_folds_ku54_62chan_left_right_hemi', 'DCN_split_spatial_kernel_3_layers_cross_after_temp_conv_learned_top_k_share_temp_conv_n_blk2_3_thresholds_distance_metric_euclidean_distance_top_k_channels_4_deterministic_T_subject_independent_all_folds_ku54_62chan_left_right_hemi', 'DCN_split_spatial_kernel_3_layers_cross_after_temp_conv_learned_top_k_share_temp_conv_n_blk2_3_thresholds_distance_metric_euclidean_distance_top_k_channels_5_deterministic_T_subject_independent_all_folds_ku54_62chan_left_right_hemi', 'DCN_split_spatial_kernel_3_layers_cross_after_temp_conv_learned_top_k_share_temp_conv_n_blk2_3_thresholds_distance_metric_euclidean_distance_top_k_channels_0_deterministic_F_subject_independent_all_folds_ku54_62chan_left_right_hemi']
    # for folder_name in results_of_choice:
    #     folder_manager = FolderManager(folder_name, '../results')
    #     print(f'{dash}{folder_name}{dash}')
    #     folder_manager.combine_all_csv_files(folder_manager.EXP_DATA, folder_manager.folder_name)

        # top_subjects_index_list = choose_subjects_w_highest_acc(all_file_path, top_k_subjects=2)
        # print(top_subjects_index_list)
        # folder_manager.combine_latex_tables_from_various_methods(results_of_choice, save_path)


'''
If I want to separate the subject indepedent and dependent trainers, is there a way for me to keep common functions like the one below in a separate file?


'''
import argparse
import json

from utils.dynamic_import_script import dynamic_import_script


# def get_parse_args():
#
#     # parser.add_argument('--args', type=class, help='args to change')
#     parser = get_parser_only_and_not_args()
#     parser_args = parser.parse_args()
#     return parser_args
#
#
# def get_parser_only_and_not_args():
#     parser = argparse.ArgumentParser(description="Select the method to run.")
#     parser.add_argument('--n_gpus', type=int, default=1, help='Number of gpus to use')
#     parser.add_argument('--dataset_name', type=str)
#     parser.add_argument('--model_name', type=str)
#     parser.add_argument('--subject_training_method', type=str)
#     parser.add_argument('--start', type=int, default=0)
#     parser.add_argument('--end', type=int, default=54)
#     parser.add_argument('--nscc', action='store_true', default=False, help='Enable NSCC')
#     parser.add_argument('--fold_to_run', type=int, help='fold to run')
#     parser.add_argument('--folder_name', type=str, help='folder file_name to save the results')
#     parser.add_argument('--params_to_change_json', type=str, default=None, help='Serialized dictionary in JSON format')
#     parser.add_argument('--nfolds', type=int, help='Number of folds')
#     # parser = add_parser_args_specific_to_dcn(parser)
#
#     return parser


def get_training_args(model_name, args_to_change=None):
    args = dynamic_import_script(module_name='model_configs', script_name=model_name, class_name='Configs')
    if args_to_change is not None:
        for key, value in args_to_change.items():
            setattr(args, key, value)
    return args

import os

def count_gpus():
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    return len(cuda_devices.split(',')) if cuda_devices else 0

#s
def get_args_to_change(subject_training_method, model_name):
    # ngpus = get_ngpus_from_os_environ()
    ngpus = count_gpus()
    print(f'ngpus is {ngpus}')

    if 'subject_dependent' in subject_training_method and 'DCN_new_every_layer_w_25filters' or 'Eegnet_new' in model_name:
        args_to_change = {'learning_rate':1e-3, 'batch_size':128 * ngpus}
    else:
        args_to_change = {'batch_size':128 * ngpus}
    return args_to_change

def get_ngpus_from_os_environ():
    gpus = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if gpus:
        num_gpus = len(gpus.split(','))
    else:
        num_gpus = 0

    print(f'num_gpus is {num_gpus}')
    return num_gpus


def _fmt_param_value(v):
    """Shorten float values so folder names stay under the filesystem limit."""
    try:
        f = float(v)
        if f != int(f):
            return f"{f:.4g}"
    except (ValueError, TypeError):
        pass
    return str(v)


def get_folder_name(model_name, subject_training_method, dataset_name, params_to_change_json, nfolds=None):
    subject_training_methods_that_require_combining_folders = ['subject_independent_first_half_folds', 'subject_independent_second_half_folds']

    if subject_training_method in subject_training_methods_that_require_combining_folders:
        subject_training_method = 'subject_independent_all_folds'

    if params_to_change_json == 'None':
        folder_name = f'{model_name}_{subject_training_method}_{dataset_name}' if nfolds is None else f'{model_name}_{subject_training_method}_{nfolds}_{dataset_name}'
    else:
        params_to_change = json.loads(params_to_change_json)
        params_to_change = '_'.join([f'{k}_{_fmt_param_value(v)}' for k, v in params_to_change.items()])
        folder_name = f'{model_name}_{params_to_change}_{subject_training_method}_{dataset_name}' if nfolds is None else f'{model_name}_{params_to_change}_{subject_training_method}_{nfolds}_{dataset_name}'

    return folder_name

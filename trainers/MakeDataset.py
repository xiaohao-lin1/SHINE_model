import os

import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle

from trainers.EegDataset import EegDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

class MakeDataset:
    '''
    This class will be used to create a dataset object.
    :param dataset_name: file_name of the dataset
    :return: dataset object, which has a  load_data , and data_information methods.
    '''

    def __init__(self, dataset_name, nscc=False, augmentation=None, x_file_name='x_reshaped.npy', torch_dataset_object=EegDataset):
        '''

        :param nscc:
        :param dataset_name:
        :param x_file_name:
        :param y_file_name:
        '''
        self.dataset_name = dataset_name #
        self.augmentation = augmentation
        self.torch_dataset_object = torch_dataset_object
        self.x_file_name = x_file_name

        self.n_test_trials = self.get_data_information()['n_test_trials']
        self.data_path = self.get_data_information()['data_path']
        self.n_channels = self.get_data_information()['n_channels']
        self.n_timesteps = self.get_data_information()['n_timesteps']
        self.n_classes = self.get_data_information()['n_classes']
        self.sampling_rate = self.get_data_information()['sampling_rate']
        self.n_subjects = self.get_data_information()['n_subjects']
        self.in_channels = self.get_data_information()['in_channels']
        self.n_trials_per_session = self.get_data_information()['n_trials_per_session']
        self.n_sessions = self.get_data_information()['n_sessions']
        self.channel_names = self.get_data_information().get('channel_names', None)
        self.overlapped_channels = self.get_data_information().get('overlapped_channels', None)
        self.nscc = nscc

    def load_all_data_and_labels(self):
        #todo: load the data here
        if self.nscc:
            self.data_path = os.path.join('/scratch/users/ntu/xiaohao0/', self.data_path)
            self.data = self.load_all_subjects_data(self.data_path)
        else:

            self.data = self.load_all_subjects_data(self.data_path)

        self.labels = np.load(os.path.join(self.data_path, 'y_reshaped.npy'))
        # return
    def load_one_subject_data_and_label(self, subject):
        filename = f's{subject:03d}.npy'
        self.data = np.load(os.path.join(self.data_path, filename))
        #todo: change line 61 of MakeDataset.py to load the labels
        self.labels = np.load(os.path.join(self.data_path, 'y_reshaped.npy'))[subject]

    def load_all_subjects_data(self, data_path):
        data = np.empty((self.n_subjects, self.n_trials_per_session * self.n_sessions, self.in_channels, self.n_channels, self.n_timesteps))
        for subject in range(self.n_subjects):
            filename = f's{subject:03d}.npy'
            print(f'Loading {filename}')
            subject_data = np.load(os.path.join(data_path, filename))
            data[subject] = subject_data
        return data

    def get_data_information(self):
        data_info_dict = {
        # 'ku54_62chan':{'data_path':'../ku54_62chan', 'n_channels':62, 'n_timesteps':1000, 'n_classes':2, 'sampling_rate':250, 'n_subjects':54, 'in_channels': 1, 'n_test_trials':100, 'n_trials_per_session': 200, 'n_sessions': 2},
        'ku54_62chan_left_right_hemi_each_subject_npy':{'data_path':'ku54_62chan_left_right_hemi_each_subject_npy', 'n_channels':56, 'n_timesteps':1000, 'n_classes':2, 'sampling_rate':250, 'n_subjects':54, 'in_channels': 1, 'n_test_trials':100, 'n_trials_per_session': 200, 'n_sessions': 2, 'channel_names':[
            #left hemisphere
            'F9', 'FT9', 'TP9', 'PO9',
            'FTT9h', 'TPP9h', #overlap: 2
            'Fp1', 'AF7', 'F7', 'T7','TP7','P7','O1',
            'AF3',
            'TPP7h', #overlap: 1
            'F3','FC5','C5', 'CP5', 'PO3',
            'FC3','C3','CP3','P3',
            'FC1','C1','CP1','P1',
            #right hemisphere
            'F10', 'FT10', 'TP10', 'PO10',
            'FTT10h', 'TPP10h', #overlap: 2
            'Fp2', 'AF8', 'F8', 'T8','TP8','P8','O2',
            'AF4',
            'TPP8h', #overlap: 1
            'F4','FC6','C6', 'CP6', 'PO4',
            'FC4','C4','CP4','P4',
            'FC2','C2','CP2','P2',

        ], 'overlapped_channels':[
                             'FTT9h',
                             'TPP7h',
                             'TPP9h',
                             'FTT10h',
                             'TPP8h',
                             'TPP10h',]},

        'Cho2017_liurui_downsample_left_right_hemi_each_subject_npy': {'data_path': 'Cho2017_liurui_downsample_left_right_hemi_each_subject_npy', 'n_channels': 54, 'n_timesteps': 768, 'n_classes': 2, 'sampling_rate': 256, 'n_subjects':52, 'in_channels': 1, 'n_test_trials':200, 'n_trials_per_session': 200, 'n_sessions': 1,
                                                      'channel_names':['FP1', 'AF7', 'AF3', 'F1', 'F3', 'F5',
                                                                                 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1',
                                                                                 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3',
                                                                                 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9',
                                                                                 'PO7', 'PO3', 'O1', 'FP2', 'AF8',
                                                                                 'AF4', 'F2', 'F4', 'F6', 'F8', 'FT8',
                                                                                 'FC6', 'FC4', 'FC2', 'C2', 'C4', 'C6',
                                                                                 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2',
                                                                                 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4',
                                                                                 'O2'],

         'overalpped_channels':[]},
        'ku54_20chan':{'data_path':'ku54_20chan_each_subject_npy', 'n_channels':20, 'n_timesteps':1000, 'n_classes':2, 'sampling_rate':250, 'n_subjects':54, 'in_channels': 1, 'n_test_trials':100, 'n_trials_per_session': 200, 'n_sessions': 2},
        'ku54_20chan_filterbank_each_subject_npy':{'data_path':'ku54_20chan_filterbank_each_subject_npy', 'n_channels':20, 'n_timesteps':1000, 'n_classes':2, 'sampling_rate':250, 'n_subjects':54, 'in_channels': 9, 'n_test_trials':100, 'n_trials_per_session': 200, 'n_sessions': 2},
        'Cho2017_liurui_downsample': {'data_path': '../Cho2017_liurui_downsample', 'n_channels': 64, 'n_timesteps': 768, 'n_classes': 2, 'sampling_rate': 256, 'n_subjects':52, 'in_channels': 1, 'n_test_trials':200, 'n_trials_per_session': 200, 'n_sessions': 1},
        'Cho2017_liurui_downsample_left_right_hemi': {'data_path': 'Cho2017_liurui_downsample_left_right_hemi', 'n_channels': 54, 'n_timesteps': 768, 'n_classes': 2, 'sampling_rate': 256, 'n_subjects':52, 'in_channels': 1, 'n_test_trials':200, 'n_trials_per_session': 200, 'n_sessions': 1},


        'open_close': {'data_path': '../datasets/CORRECTED_openclose', 'n_channels': 61, 'n_timesteps': 1000,
                       'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 1,
                       'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'openclose_cp_p_o_chan': {'data_path': '../datasets/openclose_cp_p_o_chan', 'n_channels': 24, 'n_timesteps': 1000,
                          'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 1,
                            'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1}, #
        'openclose_cp_p_o_chan_fb': {'data_path': '../datasets/openclose_cp_p_o_chan_fb', 'n_channels': 24, 'n_timesteps': 1000,
                            'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 9,
                            'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1}, #

        'openclose_cp_p_o_chan_start0.5_end1.5':{'data_path': '../datasets/openclose_cp_p_o_chan_start0.5_end1.5', 'n_channels': 24, 'n_timesteps': 250,
                                                 'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 1,
                                                 'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1}, #
#
         'openclose_cp_p_o_chan_fb_start0.5_end1.5': {'data_path': '../datasets/openclose_cp_p_o_chan_fb_start0.5_end1.5', 'n_channels': 24, 'n_timesteps': 250,
                            'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 9,
                            'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1}, #

            'openclose_unfiltered_250hz':{'data_path':'../datasets/openclose_unfiltered_250hz', 'n_channels':61, 'n_timesteps':1000,
                                          'n_classes':2, 'sampling_rate':250, 'n_subjects':50, 'in_channels': 1,
                                          'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'openclose_unfiltered_250hz_fb':{'data_path':'../datasets/openclose_unfiltered_250hz_fb', 'n_channels':61, 'n_timesteps':1000,
                                          'n_classes':2, 'sampling_rate':250, 'n_subjects':50, 'in_channels': 9,
                                          'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'open_close_samp_rate_1000': {'data_path': '../datasets/open_close_samp_rate_1000', 'n_channels': 61, 'n_timesteps': 4000,
                       'n_classes': 2, 'sampling_rate': 1000, 'n_subjects': 50, 'in_channels': 1,
                       'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'open_close_samp_rate_250_no_ica': {'data_path': '../datasets/open_close_samp_rate_250_no_ica', 'n_channels': 61, 'n_timesteps': 1000,
                          'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 1,
                            'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'open_close_samp_rate_250_ica': {'data_path': '../datasets/open_close_samp_rate_250_ica', 'n_channels': 61, 'n_timesteps': 1000,
                          'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 1,
                            'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},


        #todo: change to ../datasets/CORRECTED_openclose_fb when you have time
        'open_close_fb': {'data_path': './CORRECTED_openclose_fb', 'n_channels': 61, 'n_timesteps': 1000,
                          'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 9,
                          'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'open_close_muscle': {'data_path': './CORRECTED_openclose_muscle', 'n_channels': 12, 'n_timesteps': 1000,
                       'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 1,
                       'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'open_close_2s': {'data_path': './CORRECTED_openclose_2s', 'n_channels': 61, 'n_timesteps': 500,
                       'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 1,
                       'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'open_close_2s_fb': {'data_path': './CORRECTED_openclose_2s_fb', 'n_channels': 61, 'n_timesteps': 500,
                       'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 9,
                       'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'open_rest': {'data_path': '../datasets/open_rest', 'n_channels': 61, 'n_timesteps': 1000,
                       'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 1,
                       'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'close_rest': {'data_path': '../datasets/close_rest', 'n_channels': 61, 'n_timesteps': 1000,
                          'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 1,
                            'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'open_rest_fb': {'data_path': '../datasets/open_rest_fb', 'n_channels': 61, 'n_timesteps': 1000,
                       'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 9,
                       'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'close_rest_fb': {'data_path': '../datasets/close_rest_fb', 'n_channels': 61, 'n_timesteps': 1000,
                          'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 9,
                            'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},


        'open_rest_c': {'data_path': '../datasets/open_rest_c', 'n_channels': 61, 'n_timesteps': 1000,
                       'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 1,
                       'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'close_rest_c': {'data_path': '../datasets/close_rest_c', 'n_channels': 61, 'n_timesteps': 1000,
                          'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 1,
                            'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'open_rest_c_fb': {'data_path': '../datasets/open_rest_c_fb', 'n_channels': 61, 'n_timesteps': 1000,
                       'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 9,
                       'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'close_rest_c_fb': {'data_path': '../datasets/close_rest_c_fb', 'n_channels': 61, 'n_timesteps': 1000,
                          'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 9,
                            'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},


        'pro_rest_open': {'data_path': '../datasets/pro_rest_open', 'n_channels': 61, 'n_timesteps': 1000,
                       'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 1,
                       'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'pro_rest_close': {'data_path': '../datasets/pro_rest_close', 'n_channels': 61, 'n_timesteps': 1000,
                          'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 1,
                            'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'pro_rest_open_ind_ica': {'data_path': '../datasets/pro_rest_open_ind_ica', 'n_channels': 61, 'n_timesteps': 1000,
                          'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 1,
                            'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'pro_rest_close_ind_ica': {'data_path': '../datasets/pro_rest_close_ind_ica', 'n_channels': 61, 'n_timesteps': 1000,
                            'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 1,
                            'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'pro_rest_open_fb': {'data_path': '../datasets/pro_rest_open_fb', 'n_channels': 61, 'n_timesteps': 1000,
                       'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 1,
                       'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'pro_rest_close_fb': {'data_path': '../datasets/pro_rest_close_fb', 'n_channels': 61, 'n_timesteps': 1000,
                          'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 1,
                            'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},


        'ica_rest_open': {'data_path': '../datasets/ica_rest_open', 'n_channels': 61, 'n_timesteps': 1000,
                       'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 1,
                       'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'ica_rest_close': {'data_path': '../datasets/ica_rest_close', 'n_channels': 61, 'n_timesteps': 1000,
                          'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 1,
                            'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},



        f'ica_rest_open_threshold_0{6}': {'data_path': f'../datasets/ica_rest_open_threshold_0{6}', 'n_channels': 61, 'n_timesteps': 1000,
                       'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 1,
                       'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        f'ica_rest_open_threshold_0{7}': {'data_path': f'../datasets/ica_rest_open_threshold_0{7}', 'n_channels': 61, 'n_timesteps': 1000,
                       'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 1,
                       'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        f'ica_rest_open_threshold_0{8}': {'data_path': f'../datasets/ica_rest_open_threshold_0{8}', 'n_channels': 61, 'n_timesteps': 1000,
                       'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 1,
                       'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        f'ica_rest_open_threshold_0{9}': {'data_path': f'../datasets/ica_rest_open_threshold_0{9}', 'n_channels': 61, 'n_timesteps': 1000,
                       'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 1,
                       'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'ica_rest_open_combined_th_05': {'data_path': f'../datasets/ica_rest_open_combined_th_05', 'n_channels': 61, 'n_timesteps': 1000,
                   'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 1,
                   'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'ica_rest_close_combined_th_05': {'data_path': f'../datasets/ica_rest_close_combined_th_05', 'n_channels': 61, 'n_timesteps': 1000,
                   'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 1,
                   'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},
        #problematic ica, acc 98%
        'ica_open_close': {'data_path': f'../datasets/ica_open_close', 'n_channels': 61, 'n_timesteps': 1000,
                   'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 1,
                   'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},



        'patients_rest_open_ica': {'data_path': f'../datasets/patients_rest_open_ica', 'n_channels': 61, 'n_timesteps': 1000,
                   'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 19, 'in_channels': 1,
                   'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'patients_rest_close_ica': {'data_path': f'../datasets/patients_rest_close_ica', 'n_channels': 61, 'n_timesteps': 1000,
                     'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 19, 'in_channels': 1,
                        'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'patients_rest_open_fb_ica_20chan': {'data_path': f'../datasets/patients_rest_open_fb_ica_20chan', 'n_channels': 61,
                                   'n_timesteps': 1000,
                                   'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 19, 'in_channels': 1,
                                   'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'patients_rest_close_fb_ica_20chan': {'data_path': f'../datasets/patients_rest_close_fb_ica_20chan', 'n_channels': 61,
                                    'n_timesteps': 1000,
                                    'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 19, 'in_channels': 1,
                                    'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'patients_rest_open_ind_ica': {'data_path': f'../datasets/patients_rest_open_ind_ica', 'n_channels': 61,
                                    'n_timesteps': 1000,
                                    'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 19, 'in_channels': 1,
                                    'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'patients_rest_close_ind_ica': {'data_path': f'../datasets/patients_rest_close_ind_ica', 'n_channels': 61,
                                    'n_timesteps': 1000,
                                    'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 19, 'in_channels': 1,
                                    'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'patients_rest_open': {'data_path': f'../datasets/patients_rest_open', 'n_channels': 61,
                                    'n_timesteps': 1000,
                                    'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 19, 'in_channels': 1,
                                    'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

        'patients_rest_close': {'data_path': f'../datasets/patients_rest_close', 'n_channels': 61,
                                    'n_timesteps': 1000,
                                    'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 19, 'in_channels': 1,
                                    'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},

            # 'open_close': {}
        # 'Cho2017_liurui_downsample_left_right_hemi': {'data_path': '../Cho2017_liurui_downsample_left_right_hemi', 'n_channels': 54, 'n_timesteps': 768, 'n_classes': 2, 'sampling_rate': 256, 'n_subjects':52, 'in_channels': 1, 'n_test_trials':200, 'n_trials_per_session': 200, 'n_sessions': 1},
        # 'ku54_62chan_fb':{'data_path':'../ku54_62chan_fb', 'n_channels':62, 'n_timesteps':1000, 'n_classes':2, 'sampling_rate':250, 'n_subjects':54, 'in_channels': 1, 'n_test_trials':100, 'n_trials_per_session': 200, 'n_sessions': 2},
        # 'bandpass_ku54_62chan_4to40hz': {'data_path': '../bandpass_ku54_62chan_4to40hz', 'n_channels': 62, 'n_timesteps': 1000, 'n_classes': 2, 'sampling_rate': 250, 'n_subjects':54, 'in_channels': 1, 'n_test_trials':100, 'n_trials_per_session': 200, 'n_sessions': 2},
        # 'bandpass_ku54_62chan_8to40hz': {'data_path': '../bandpass_ku54_62chan_8to40hz', 'n_channels': 62, 'n_timesteps': 1000, 'n_classes': 2, 'sampling_rate': 250, 'n_subjects':54, 'in_channels': 1, 'n_test_trials':100, 'n_trials_per_session': 200, 'n_sessions': 2},
        #
        # 'ku54_20channels': {'data_path': '../ku54_20channels', 'n_channels': 20, 'n_timesteps': 1000, 'n_classes': 2, 'sampling_rate': 250, 'n_subjects':54, 'in_channels': 1, 'n_test_trials':100, 'n_trials_per_session': 200, 'n_sessions': 2},
        # 'ku54_fbcnet': {'data_path': '../ku54_fbcnet', 'n_channels': 20, 'n_timesteps': 1000, 'n_classes': 2, 'sampling_rate': 250, 'n_subjects':54, 'in_channels': 9, 'n_test_trials':100, 'n_trials_per_session': 200, 'n_sessions': 2},
        #
        # 'cho_etal':{'data_path': '../cho_etal', 'n_channels': 64, 'n_timesteps': 769, 'n_classes': 2, 'sampling_rate': 256, 'n_subjects':52, 'in_channels': 1, 'n_test_trials':200, 'n_trials_per_session': 200, 'n_sessions': 1},
        # 'cho_etal_no_downsample':{'data_path': '../cho_etal_no_downsample', 'n_channels': 64, 'n_timesteps': 1537, 'n_classes': 2, 'sampling_rate': 256*2, 'n_subjects':52, 'in_channels': 1, 'n_test_trials':200, 'n_trials_per_session': 200, 'n_sessions': 1},
        # 'cho_etal_filterbank':{'data_path': '../cho_etal_filterbank', 'n_channels': 64, 'n_timesteps': 769, 'n_classes': 2, 'sampling_rate': 256, 'n_subjects':52, 'in_channels': 9, 'n_test_trials':200, 'n_trials_per_session': 200, 'n_sessions': 1},
        # 'cho_etal_filterbank_axis1':{'data_path': '../cho_etal_filterbank_axis1', 'n_channels': 64, 'n_timesteps': 769, 'n_classes': 2, 'sampling_rate': 256, 'n_subjects':52, 'in_channels': 9, 'n_test_trials':200, 'n_trials_per_session': 200, 'n_sessions': 1},
        # 'cho_etal_filterbank_filt_filt_axis_minus1': {'data_path': '../cho_etal_filterbank_filt_filt_axis_minus1', 'n_channels': 64, 'n_timesteps': 769, 'n_classes': 2, 'sampling_rate': 256, 'n_subjects':52, 'in_channels': 9, 'n_test_trials':200, 'n_trials_per_session': 200, 'n_sessions': 1},
        # 'cho_etal_juce_filterbank': {'data_path': '../cho_etal_juce_filterbank', }
        # 'cho_etal_juce_filterbank': {'data_path': '../cho_etal_juce_filterbank', 'n_channels': 64, 'n_timesteps': 768, 'n_classes': 2, 'sampling_rate': 256, 'n_subjects':52, 'in_channels': 9, 'n_test_trials':200, 'n_trials_per_session': 200, 'n_sessions': 1},
        # 'cho_etal_juce_filterbank_0_to_40hz': {'data_path': '../cho_etal_juce_filterbank_0_to_40hz', 'n_channels': 64, 'n_timesteps': 769, 'n_classes': 2, 'sampling_rate': 256, 'n_subjects':52, 'in_channels': 9, 'n_test_trials':200, 'n_trials_per_session': 200, 'n_sessions': 1},
        #
        # 'Cho2017_liurui': {'data_path': '../Cho2017_liurui', 'n_channels': 64, 'n_timesteps': 1536, 'n_classes': 2, 'sampling_rate': 256*2, 'n_subjects':52, 'in_channels': 1, 'n_test_trials':200, 'n_trials_per_session': 200, 'n_sessions': 1},

        }
        return data_info_dict[self.dataset_name]

    def get_trials(self, index, slice_trials=None, specific_trials=None):
        '''

        :param index:
        :param slice_trials:
        :param specific_trials:
        :return: data: shape (ntrials, 1, n_channels, n_timesteps)
                label: shape (ntrials, )
        '''
        in_channels = self.data.shape[2]

        if specific_trials is not None:
            data = self.data[index, specific_trials, :, :, :]
            label = self.labels[index, specific_trials]
        elif slice_trials is not None:
            data = self.data[index, :slice_trials, :, :, :]
            label = self.labels[index, :slice_trials]
        else:
            data = self.data[index, :, :, :, :]
            label = self.labels[index, :]

        data = data.reshape(-1, in_channels, self.n_channels, self.n_timesteps)
        label = label.reshape(-1)
        return data, label

    def get_train_data_subject_independent(self, train_val_indices):
        train_data, train_label = self.get_trials(train_val_indices, slice_trials=None, specific_trials=None)
        return train_data, train_label


    def get_train_data_subject_dependent_holdout(self, train_index):
        train_data, train_label = self.get_trials(train_index, slice_trials=self.n_trials_per_session,
                                                  specific_trials=None)
        return train_data, train_label



    def get_val_data_from_train_subject_independent(self, train_data, train_label):
        train_data, val_data, train_label, val_label = train_test_split(
            train_data, train_label, test_size=0.15, random_state=0, stratify=train_label)

        return train_data, val_data, train_label, val_label

    def get_val_data_from_train_subject_dependent_holdout(self, train_data, train_label):
        train_data, val_data, train_label, val_label = train_test_split(
            train_data, train_label, test_size=0.15, random_state=0, stratify=train_label)

        return train_data, val_data, train_label, val_label

    def get_val_data_from_train_subject_dependent_10fold_cv(self, train_data, train_label):
        train_data, val_data, train_label, val_label = train_test_split(
            train_data, train_label, test_size=0.15, random_state=0, stratify=train_label)

        return train_data, val_data, train_label, val_label
        #generate

    def get_test_data_subject_independent(self, test_indices):
        in_channels = self.data.shape[2]
        test_data = self.data[test_indices, -self.n_test_trials:, :, :, :]
        test_label = self.labels[test_indices, -self.n_test_trials:]
        test_data = test_data.reshape(-1, in_channels, self.n_channels, self.n_timesteps)
        test_label = test_label.reshape(-1)
        return test_data, test_label

    def get_test_data_subject_dependent_holdout(self, test_indices):
        return self.get_test_data_subject_independent(test_indices)
    def get_test_data_subject_dependent_10fold_cv(self, test_indices):
        pass
    def normalise(self, train_data, val_data, test_data):
        '''
        This function get the mean and std from train data, and normalise train, val and test data.
        :param train_data:
        :param val_data:
        :param test_data:
        :return: normalised train, val and test data.
        '''
        print('----------Normalising data------------------')
        mean, std = self.get_channelwise_mean_std_from_train_data(train_data)
        self.channelwise_mean = mean
        self.channelwise_std = std

        train_data = self.minus_mean_divide_std(train_data, mean, std)
        test_data = self.minus_mean_divide_std(test_data, mean, std)

        if val_data is not None:
            val_data = self.minus_mean_divide_std(val_data, mean, std)
            return train_data, val_data, test_data
        else:
            return train_data, test_data

    def get_channelwise_mean_std_from_train_data(self, train_data):
        '''
        :param train_data: shape (ntrials, 1, n_channels, n_timesteps)
        :return: channel_wise_mean shape (n_channels, 1)
        '''
        channel_wise_mean = train_data.mean(axis=(0, 1, 3), keepdims=False)
        channel_wise_mean = channel_wise_mean[:, np.newaxis].astype(np.float32)

        channel_wise_std = train_data.std(axis=(0, 1, 3), keepdims=False)
        channel_wise_std = channel_wise_std[:, np.newaxis] + 1e-10
        channel_wise_std = channel_wise_std.astype(np.float32)


        return channel_wise_mean, channel_wise_std

    def minus_mean_divide_std(self, data: object, mean: object, std: object) -> object:
        data -= mean
        data /= std
        return data

    def subject_independent_get_unnormalised_train_val_test_data_pipeline(self, train_val_indices: list, test_indices: list):
        '''
        This function is the pipeline to get the unnormalised train, val and test data.
        :param train_val_indices:
        :param test_indices:
        :return:train_data shape (ntrials, 1, n_channels, n_timesteps)
        '''
        train_data, train_labels = self.get_train_data_subject_independent(train_val_indices)
        train_data, val_data, train_label, val_label = self.get_val_data_from_train_subject_independent(train_data, train_labels)

        test_data, test_label = self.get_test_data_subject_independent(test_indices)
        return train_data, val_data, test_data, train_label, val_label, test_label

    def subject_independent_get_normalised_train_val_test_data_pipeline(self, train_val_indices: list, test_indices:list):
        '''
        This function is the pipeline to get the normalised train, val and test data.
        :param train_val_indices:
        :param test_indices:
        :return:
        '''

        train_data, val_data, test_data, train_label, val_label, test_label = self.subject_independent_get_unnormalised_train_val_test_data_pipeline(train_val_indices, test_indices)

        train_data, val_data, test_data = self.normalise(train_data, val_data, test_data)
        return train_data, val_data, test_data, train_label, val_label, test_label

    def subject_independent_get_normalised_train_and_test_data_only_pipeline(self, train_val_indices: list, test_indices:list):
        '''
        This function is the pipeline to get the normalised train, val and test data.
        :param train_val_indices:
        :param test_indices:
        :return:
        '''
        train_data, train_label = self.get_train_data_subject_independent(train_val_indices)
        print(f'len of train_data, {len(train_data)}')
        print(f'len of train_labels, {len(train_label)}')
        test_data, test_label = self.get_test_data_subject_independent(test_indices)
        print(f'len of test_data, {len(test_data)}')
        print(f'len of test_label, {len(test_label)}')
        mean, std = self.get_channelwise_mean_std_from_train_data(train_data)
        train_data = self.minus_mean_divide_std(train_data, mean, std)
        test_data = self.minus_mean_divide_std(test_data, mean, std)
        return train_data, test_data, train_label, test_label
    def subject_dependent_holdout_get_normalised_train_val_test_data_pipeline(self):
        '''
        This function is the pipeline to get the normalised train, val and test data.
        train_data shape (nsubjects, ntrials, 1, n_channels, n_timesteps)
        train_label shape (nsubjects, ntrials)

        :return:
        '''
        train_data = self.data[:self.n_trials_per_session, :, :, :]
        train_label = self.labels[:self.n_trials_per_session]

        if 'ku' in self.dataset_name:
            val_data = self.data[self.n_trials_per_session:int(1.5 * self.n_trials_per_session), :, :, :]
            val_label = self.labels[self.n_trials_per_session:int(1.5 * self.n_trials_per_session)]

            test_data = self.data[int(1.5 * self.n_trials_per_session):, :, :, :]
            test_label = self.labels[int(1.5 * self.n_trials_per_session):]
        else:
            val_data = None
            val_label = None

            test_data = self.data[self.n_trials_per_session:, :, :, :]
            test_label = self.labels[self.n_trials_per_session:]

        train_data, val_data, test_data = self.normalise(train_data, val_data, test_data)
        return train_data, val_data, test_data, train_label, val_label, test_label


    def subject_dependent_10fold_cv_get_normalised_train_val_test_data_pipeline(self, data, labels, train_val_indices, test_indices):
        '''
        This function is the pipeline to get the normalised train, val and test data.
        :param train_index:
        :return:
        '''
        train_data = data[train_val_indices]
        train_labels = labels[train_val_indices]

        test_data = data[test_indices]
        test_labels = labels[test_indices]

        # train_data, val_data, train_labels, val_labels = self.get_val_data_from_train_subject_dependent_10fold_cv(train_data, train_labels)

        train_data, val_data, test_data = self.normalise(train_data, None, test_data)
        val_data, val_labels = None, None
        return train_data, val_data, test_data, train_labels, val_labels, test_labels


    def get_normalised_train_val_test_data_pipeline(self, data, labels, train_indices, val_indices, test_indices):
        '''
        This function is the pipeline to get the normalised train, val and test data.
        :param train_index:
        :return:
        '''
        train_data = data[train_indices]
        train_labels = labels[train_indices]

        val_data, val_labels = data[val_indices], labels[val_indices]

        if test_indices is not None:
            test_data = data[test_indices]
            test_labels = labels[test_indices]
            train_data, val_data, test_data = self.normalise(train_data, val_data, test_data)
            return train_data, val_data, test_data, train_labels, val_labels, test_labels
        else:
            train_data, val_data = self.normalise(train_data, val_data=None, test_data=val_data)
            return train_data, val_data, None, train_labels, val_labels, None
        # todo: i changed to EegDataset_return_augmented_tgt_with_original, need to change back

    def convert_numpy_to_torch_dataloaders(self, batchsize, data, labels):

        data = self.torch_dataset_object(data, labels, self.sampling_rate, self.augmentation)
        dataloader = DataLoader(
            data,
            batch_size=batchsize,
            pin_memory=True,
            shuffle=True,
            num_workers=0
        )
        return dataloader

    def make_train_val_test_dataloaders(self, batch_size, train_data, val_data, test_data, train_label, val_label, test_label):
        train_dataloader = self.convert_numpy_to_torch_dataloaders(batch_size, train_data, train_label)

        if val_data is None:
            val_dataloader = None
        else:
            val_dataloader = self.convert_numpy_to_torch_dataloaders(batch_size, val_data, val_label)

        # test data is always unaugmented
        test_data = EegDataset(test_data, test_label, self.sampling_rate, augmentation=None)

        test_dataloader = DataLoader(
            test_data,
            batch_size=batch_size,
            # pin_memory=True,
            shuffle=False,
        )
        return train_dataloader, val_dataloader, test_dataloader

    def __repr__(self):
        return self.dataset_name


    def __call__(self, *args, **kwargs):
        data_information = self.get_data_information()
        return data_information['data_path'], data_information['n_channels'], data_information['n_timesteps'], data_information['n_classes'], data_information['sampling_rate']


class MakeDataset_OpenClose(MakeDataset):


    def __init__(self, dataset_name, nscc=False, augmentation=None, x_file_name='x_reshaped.npy', torch_dataset_object=EegDataset):
        super().__init__(dataset_name, nscc, augmentation, x_file_name, torch_dataset_object)
    # def get_data_information(self):
    #     data_info_dict = {
    #         'open_close': {'data_path': '../datasets/CORRECTED_openclose', 'n_channels': 61, 'n_timesteps': 1000,
    #                        'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 1,
    #                        'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},
    #
    #         'open_close_fb': {'data_path': './CORRECTED_openclose_fb', 'n_channels': 61, 'n_timesteps': 1000,
    #                        'n_classes': 2, 'sampling_rate': 250, 'n_subjects': 50, 'in_channels': 9,
    #                        'n_test_trials': 20, 'n_trials_per_session': 200, 'n_sessions': 1},
    #     }
    #
    #     return data_info_dict[self.dataset_name]

        # return
    def load_one_subject_data_and_label(self, subject):

        x_filename = f's{subject:03d}.npy'
        y_filename = f'y{subject:03d}.npy'
        self.data = np.load(os.path.join(self.data_path, x_filename))
        self.labels = np.load(os.path.join(self.data_path, y_filename))

        # return data, labels
    def load_one_subject_data_and_label_no_setting_self_data_label(self, subject):

        x_filename = f's{subject:03d}.npy'
        y_filename = f'y{subject:03d}.npy'
        data = np.load(os.path.join(self.data_path, x_filename))
        labels = np.load(os.path.join(self.data_path, y_filename))
        return data, labels

    def load_seeded_one_subject_data_and_label_no_setting_self_data_label(self, subject, seed_no, train_val_test):

        x_filename = f's{subject:03d}_seed_{seed_no}_{train_val_test}.npy'
        y_filename = f'y{subject:03d}_seed_{seed_no}_{train_val_test}.npy'
        data = np.load(os.path.join(self.data_path, x_filename))
        labels = np.load(os.path.join(self.data_path, y_filename))
        return data, labels



    # def generate_seeded_one_subject_data_and_label_no_setting_self_data_label(
    #         self, subject, seeds=(1, 2, 3)
    # ):
    #     """
    #     Generate 6-2-2 (train/val/test) stratified splits per seed
    #     and save them as .npy files.
    #
    #     Train: 60%
    #     Val:   20%
    #     Test:  20%
    #     """
    #
    #     # ------------------------
    #     # Load data
    #     # ------------------------
    #     x_filename = f's{subject:03d}.npy'
    #     y_filename = f'y{subject:03d}.npy'
    #
    #     data = np.load(os.path.join(self.data_path, x_filename))
    #     labels = np.load(os.path.join(self.data_path, y_filename))
    #
    #     assert len(data) == len(labels), "Data and label size mismatch"
    #
    #     # ------------------------
    #     # Loop over seeds
    #     # ------------------------
    #     for seed in seeds:
    #         # -------- Step 1: Train+Val vs Test (80 / 20) --------
    #         sss_test = StratifiedShuffleSplit(
    #             n_splits=1, test_size=0.2, random_state=seed
    #         )
    #
    #         train_val_idx, test_idx = next(sss_test.split(data, labels))
    #
    #         X_train_val = data[train_val_idx]
    #         y_train_val = labels[train_val_idx]
    #
    #         X_test = data[test_idx]
    #         y_test = labels[test_idx]
    #
    #         # -------- Step 2: Train vs Val (75 / 25 of remaining) --------
    #         sss_val = StratifiedShuffleSplit(
    #             n_splits=1, test_size=0.25, random_state=seed
    #         )
    #
    #         train_idx, val_idx = next(sss_val.split(X_train_val, y_train_val))
    #
    #         X_train = X_train_val[train_idx]
    #         y_train = y_train_val[train_idx]
    #
    #         X_val = X_train_val[val_idx]
    #         y_val = y_train_val[val_idx]
    #
    #         # ------------------------
    #         # Save to disk
    #         # ------------------------
    #         np.save(
    #             os.path.join(self.data_path, f's{subject:03d}_seed_{seed}_train.npy'),
    #             X_train,
    #         )
    #         np.save(
    #             os.path.join(self.data_path, f'y{subject:03d}_seed_{seed}_train.npy'),
    #             y_train,
    #         )
    #
    #         np.save(
    #             os.path.join(self.data_path, f's{subject:03d}_seed_{seed}_val.npy'),
    #             X_val,
    #         )
    #         np.save(
    #             os.path.join(self.data_path, f'y{subject:03d}_seed_{seed}_val.npy'),
    #             y_val,
    #         )
    #
    #         np.save(
    #             os.path.join(self.data_path, f's{subject:03d}_seed_{seed}_test.npy'),
    #             X_test,
    #         )
    #         np.save(
    #             os.path.join(self.data_path, f'y{subject:03d}_seed_{seed}_test.npy'),
    #             y_test,
    #         )
    #
    #         print(
    #             f"[OK] Subject {subject:03d} | Seed {seed} "
    #             f"Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}"
    #         )
    #
    #     return data, labels

    import os
    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit

    def generate_seeded_one_subject_data_and_label_no_setting_self_data_label(
            self, subject, seeds=(1, 2, 3), balance_tol=1e-6
    ):
        """
        Generate 6-2-2 (train/val/test) stratified splits per seed,
        save them, check class balance, and print indices.

        balance check: mean(y) ≈ 0.5
        """

        # ------------------------
        # Load data
        # ------------------------
        x_filename = f's{subject:03d}.npy'
        y_filename = f'y{subject:03d}.npy'

        data = np.load(os.path.join(self.data_path, x_filename))
        labels = np.load(os.path.join(self.data_path, y_filename))

        assert len(data) == len(labels), "Data and label size mismatch"

        # ------------------------
        # Loop over seeds
        # ------------------------
        for seed in seeds:
            print(f"\n===== Subject {subject:03d} | Seed {seed} =====")

            # -------- Step 1: Train+Val vs Test (80 / 20) --------
            sss_test = StratifiedShuffleSplit(
                n_splits=1, test_size=0.2, random_state=seed
            )

            train_val_idx, test_idx = next(sss_test.split(data, labels))

            X_train_val = data[train_val_idx]
            y_train_val = labels[train_val_idx]

            X_test = data[test_idx]
            y_test = labels[test_idx]

            # -------- Step 2: Train vs Val (75 / 25 of remaining) --------
            sss_val = StratifiedShuffleSplit(
                n_splits=1, test_size=0.25, random_state=seed
            )

            train_idx_local, val_idx_local = next(
                sss_val.split(X_train_val, y_train_val)
            )

            # map back to ORIGINAL indices
            train_idx = train_val_idx[train_idx_local]
            val_idx = train_val_idx[val_idx_local]

            X_train = data[train_idx]
            y_train = labels[train_idx]

            X_val = data[val_idx]
            y_val = labels[val_idx]

            # ------------------------
            # Balance checks
            # ------------------------
            def check_balance(y, split_name):
                mean_y = np.mean(y)
                print(f"{split_name} mean(y): {mean_y:.4f}")
                if abs(mean_y - 0.5) > balance_tol:
                    print(
                        f"[WARNING] {split_name} not balanced "
                        f"(mean={mean_y:.4f})"
                    )

            check_balance(y_train, "Train")
            check_balance(y_val, "Val")
            check_balance(y_test, "Test")

            # ------------------------
            # Print indices
            # ------------------------
            print("Train indices:", train_idx.tolist())
            print("Val indices:  ", val_idx.tolist())
            print("Test indices: ", test_idx.tolist())

            # ------------------------
            # Save to disk
            # ------------------------
            np.save(
                os.path.join(self.data_path, f's{subject:03d}_seed_{seed}_train.npy'),
                X_train,
            )
            np.save(
                os.path.join(self.data_path, f'y{subject:03d}_seed_{seed}_train.npy'),
                y_train,
            )

            np.save(
                os.path.join(self.data_path, f's{subject:03d}_seed_{seed}_val.npy'),
                X_val,
            )
            np.save(
                os.path.join(self.data_path, f'y{subject:03d}_seed_{seed}_val.npy'),
                y_val,
            )

            np.save(
                os.path.join(self.data_path, f's{subject:03d}_seed_{seed}_test.npy'),
                X_test,
            )
            np.save(
                os.path.join(self.data_path, f'y{subject:03d}_seed_{seed}_test.npy'),
                y_test,
            )

            print(
                f"[SAVED] Train {len(X_train)} | "
                f"Val {len(X_val)} | Test {len(X_test)}"
            )

        return data, labels

    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit

    import numpy as np

    def get_percent_of_data(self, percent, data, labels, seed=0):
        """
        Return a nested subset of the data with the given percentage.
        Guarantees:
            20% ⊂ 40% ⊂ 60% ⊂ 80% ⊂ 100%

        Args:
            percent (int or float): one of {0, 20, 40, 60, 80, 100}
            data (np.ndarray)
            labels (np.ndarray)
            seed (int): deterministic seed

        Returns:
            subset_data
            subset_labels
            subset_indices  (indices w.r.t. input data)
        """

        assert 0 <= percent <= 100, f"Invalid percent: {percent}"
        assert len(data) == len(labels), "Data / label length mismatch"

        n = len(labels)

        # -------------------------
        # Edge cases
        # -------------------------
        if percent == 0:
            return np.empty((0, *data.shape[1:])),np.empty((0,), dtype=labels.dtype),
                # np.array([], dtype=int),

        if percent == 100:
            return data, labels

        # -------------------------
        # Step 1: stratified ordering
        # -------------------------
        rng = np.random.RandomState(seed)

        # ordered_indices_label_0 = []
        # ordered_indices_label_1 = []

        cls_0_indices = np.where(labels == 0)[0]
        cls_1_indices = np.where(labels == 1)[0]


        # for cls in np.unique(labels):
        #     cls_indices = np.where(labels == cls)[0]
        #     # rng.shuffle(cls_indices)
        #     ordered_indices.append(cls_indices)

        # interleave to preserve balance
        # ordered_indices = np.concatenate(ordered_indices)

        # -------------------------
        # Step 2: prefix slice
        # -------------------------
        n_keep_half = int(round(n * percent / 100 /2))
        subset_cls_0_indices = cls_0_indices[:n_keep_half]
        subset_cls_1_indices = cls_1_indices[:n_keep_half]
        subset_indices = np.concatenate([subset_cls_0_indices, subset_cls_1_indices])
        rng = np.random.RandomState(seed)
        rng.shuffle(subset_indices)

        subset_data = data[subset_indices]
        subset_labels = labels[subset_indices]
        # print('subset labels')
        print(subset_labels)

        return subset_data, subset_labels

    def load_all_subjects_data(self, subjects_list):
        data = np.vstack([np.load(os.path.join(self.data_path, f's{subject:03d}.npy')) for subject in subjects_list])
        return data
    #
    # def load_all_subjects_labels(self, subjects_list):
    #     labels = np.vstack([np.load(os.input_path.join(self.data_path, f'y{subject:03d}.npy')) for subject in subjects_list])
    #     return labels
    # def load_all_subjects_data(self, subjects_list):
    #     data_list = []
    #     for subject in subjects_list:
    #         file_path = os.input_path.join(self.data_path, f's{subject:03d}.npy')
    #         if os.input_path.exists(file_path):
    #             subject_data = np.load(file_path)
    #             data_list.append(subject_data)
    #         else:
    #             print(f"File {file_path} does not exist.")
    #     data = np.vstack(data_list)
    #     return data

    def load_all_subjects_labels(self, subjects_list):
        labels_list = []
        for subject in subjects_list:
            file_path = os.path.join(self.data_path, f'y{subject:03d}.npy')
            if os.path.exists(file_path):
                subject_data = np.expand_dims(np.load(file_path),1) #expand the dim by 1 to stack it
                labels_list.append(subject_data)
            else:
                print(f"File {file_path} does not exist.")
        labels = np.vstack(labels_list).squeeze(1)
        return labels

    def subject_dependent_holdout_get_normalised_train_val_test_data_pipeline(self):
        '''
        This function is the pipeline to get the normalised train, val and test data.
        train_data shape (ntrials, 1, n_channels, n_timesteps)
        train_label shape (ntrials)

        :return:
        '''

        n_val_trials = self.n_test_trials

        train_data, test_data, train_label, test_label = train_test_split(
            self.data, self.labels, test_size=self.n_test_trials, random_state=0, stratify=self.labels, shuffle=True)

        train_data, val_data, train_label, val_label = train_test_split(
            train_data, train_label, test_size=n_val_trials, random_state=0, stratify=train_label, shuffle=True
        )

        train_data, val_data, test_data = self.normalise(train_data, val_data, test_data)
        return train_data, val_data, test_data, train_label, val_label, test_label



    # def slice_train_val_test_trials(self, n_train_trials, n_val_trials, n_test_trials):
    #     train_data = self.data[:n_train_trials, :, :, :]
    #     train_label = self.labels[:n_train_trials]
    #
    #     val_data = self.data[n_train_trials:n_train_trials + n_val_trials, :, :,:]
    #     val_label = self.labels[n_train_trials:n_train_trials + n_val_trials]
    #
    #     test_data = self.data[-n_test_trials:, :, :, :]
    #     test_label = self.labels[-n_test_trials:]
    #
    #     return train_data, val_data, test_data, train_label, val_label, test_label

    def subject_dependent_test_data_only_normalised_train_val_test_data_pipeline(self):
        '''
        This function is the pipeline to get the normalised train, val and test data.
        train_data shape (ntrials, 1, n_channels, n_timesteps)
        train_label shape (ntrials)

        :return:
        '''
        _, _, test_data, _, _, test_label = self.subject_dependent_holdout_get_normalised_train_val_test_data_pipeline()
        train_data, val_data = test_data, test_data
        train_label, val_label = test_label, test_label
        return train_data, val_data, test_data, train_label, val_label, test_label


    def subject_dependent_test_all_trials_normalised_train_val_test_data_pipeline(self):
        train_data = np.copy(self.data)
        val_data = np.copy(self.data)
        test_data = np.copy(self.data)

        train_label = np.copy(self.labels)
        val_label = np.copy(self.labels)
        test_label = np.copy(self.labels)


        train_data, val_data, test_data = self.normalise(train_data, val_data, test_data)
        return train_data, val_data, test_data, train_label, val_label, test_label

    def subject_independent_get_normalised_train_val_test_data_pipeline(self, train_indices: list, val_indices, test_indices: list):
        '''
        This function is the pipeline to get the normalised train, val and test data.
        :param train_val_indices:
        :param test_indices:
        :return:
        '''

        train_data, val_data, test_data, train_label, val_label, test_label = self.subject_independent_get_unnormalised_train_val_test_data_pipeline(train_indices, val_indices, test_indices)

        train_data, val_data, test_data = self.normalise(train_data, val_data, test_data)

        return train_data, val_data, test_data, train_label, val_label, test_label

    def subject_independent_get_unnormalised_train_val_test_data_pipeline(self, train_indices: list, val_indices, test_indices: list):
        '''
        This function is the pipeline to get the unnormalised train, val and test data.
        ONLY the last 20 trials are used for testing
        :param train_val_indices:
        :param test_indices:
        :return:train_data shape (ntrials, 1, n_channels, n_timesteps)
        '''
        train_data = self.load_all_subjects_data(train_indices)
        train_label = self.load_all_subjects_labels(train_indices)

        val_data = self.load_all_subjects_data(val_indices)
        val_label = self.load_all_subjects_labels(val_indices)

        test_data = self.load_all_subjects_data(test_indices)
        test_label = self.load_all_subjects_labels(test_indices)

        _, test_data, _, test_label = train_test_split(
            test_data, test_label, test_size=self.n_test_trials, random_state=0, stratify=test_label, shuffle=True)


        return train_data, val_data, test_data, train_label, val_label, test_label

    def subject_independent_load_pretrained_get_unnormalised_train_val_test_data_pipeline(self, train_indices: list, val_indices,
                                                                          test_indices: list):
        '''
        This function is the pipeline to get the unnormalised train, val and test data.
        ONLY the last 20 trials are used for testing
        :param train_val_indices:
        :param test_indices:
        :return:train_data shape (ntrials, 1, n_channels, n_timesteps)
        '''
        train_data = self.load_all_subjects_data(train_indices)
        train_label = self.load_all_subjects_labels(train_indices)

        val_data = self.load_all_subjects_data(val_indices)
        val_label = self.load_all_subjects_labels(val_indices)

        test_data = self.load_all_subjects_data(test_indices)
        test_label = self.load_all_subjects_labels(test_indices)

        # print(f'test label shape {test_label.shape}')

        return train_data, val_data, test_data, train_label, val_label, test_label

    def get_trials(self, index, slice_trials=None, specific_trials=None):
        '''

        :param index:
        :param slice_trials:
        :param specific_trials:
        :return: data: shape (ntrials, 1, n_channels, n_timesteps)
                label: shape (ntrials, )
        '''
        in_channels = self.data.shape[2]

        if specific_trials is not None:
            data = self.data[index, specific_trials, :, :, :]
            label = self.labels[index, specific_trials]
        elif slice_trials is not None:
            data = self.data[index, :slice_trials, :, :, :]
            label = self.labels[index, :slice_trials]
        else:
            data = self.data[index, :, :, :, :]
            label = self.labels[index, :]

        data = data.reshape(-1, in_channels, self.n_channels, self.n_timesteps)
        label = label.reshape(-1)
        return data, label


    def subject_independent_cv_selected_get_normalised_train_val_test_data_pipeline(self, train_indices, val_indices,
                                                                                    test_indices):
        '''
        This function is the pipeline to get the unnormalised train, val and test data.
        ONLY the last 20 trials are used for testing
        :param train_val_indices:
        :param test_indices:
        :return:train_data shape (ntrials, 1, n_channels, n_timesteps)
        '''

        n_val_trials = self.n_test_trials

        train_data_ls = []
        train_label_ls = []
        val_data_ls = []
        val_label_ls = []

        for subject in train_indices:

            x_filename = f's{subject:03d}.npy'
            y_filename = f'y{subject:03d}.npy'
            data = np.load(os.path.join(self.data_path, x_filename))
            labels = np.load(os.path.join(self.data_path, y_filename))

            train_data, _, train_label, _ = train_test_split(
                data, labels, test_size=self.n_test_trials, random_state=0, stratify=labels,
                shuffle=True)

            train_data, val_data, train_label, val_label = train_test_split(
                train_data, train_label, test_size=n_val_trials, random_state=0, stratify=train_label, shuffle=True
            )
            train_data_ls.append(train_data)
            train_label_ls.append(train_label)
            val_data_ls.append(val_data)
            val_label_ls.append(val_label)

        train_data = np.concatenate(train_data_ls, axis=0)
        train_label = np.concatenate(train_label_ls, axis=0)
        val_data = np.concatenate(val_data_ls, axis=0)
        val_label = np.concatenate(val_label_ls, axis=0)

        test_data = self.load_all_subjects_data(test_indices)
        test_label = self.load_all_subjects_labels(test_indices)

        _, test_data, _, test_label = train_test_split(
            test_data, test_label, test_size=self.n_test_trials, random_state=0, stratify=test_label, shuffle=True)


        return train_data, val_data, test_data, train_label, val_label, test_label


import os
import numpy as np

# -----------------------------
# IMPORT THE FUNCTION UNDER TEST
# -----------------------------
# from your_module import get_percent_of_data   # adjust import path


def load_saved_train_data(data_path, subject, seed):
    """
    Load previously saved train split from earlier pipeline
    """
    x_path = os.path.join(
        data_path, f"s{subject:03d}_seed_{seed}_train.npy"
    )
    y_path = os.path.join(
        data_path, f"y{subject:03d}_seed_{seed}_train.npy"
    )

    X = np.load(x_path)
    y = np.load(y_path)

    return X, y


def test_nested_percent_splits():
    # -----------------------------
    # CONFIG
    # -----------------------------
    dataset = MakeDataset_OpenClose('patients_rest_open_ica', nscc=False)

    data_path = r'../../datasets/patients_rest_open_ica'
    subject = 1
    seed = 1
    percents = [0, 20, 40, 60, 80, 100]

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    X, y = load_saved_train_data(data_path, subject, seed)

    print(f"Loaded train data: X={X.shape}, y={y.shape}")
    print(f"Original mean(y): {np.mean(y):.4f}\n")

    prev_indices = None

    # -----------------------------
    # RUN TESTS
    # -----------------------------
    for p in percents:
        X_p, y_p, idx_p = dataset.get_percent_of_data(
            percent=p,
            data=X,
            labels=y,
            seed=0
        )

        print(f"--- {p}% DATA ---")
        print(f"Samples: {len(y_p)}")
        print(f"Mean(y): {np.mean(y_p) if len(y_p) > 0 else 'N/A'}")
        print(f"Indices: {idx_p.tolist()}")

        # -----------------------------
        # TEST 1: SIZE CHECK
        # -----------------------------
        expected_n = int(round(len(y) * p / 100))
        assert len(y_p) == expected_n, (
            f"Size mismatch at {p}%: "
            f"got {len(y_p)}, expected {expected_n}"
        )

        # -----------------------------
        # TEST 2: NESTING PROPERTY
        # -----------------------------
        if prev_indices is not None:
            assert set(prev_indices).issubset(set(idx_p)), (
                f"Nesting violated: {p}% does not include previous subset"
            )

        prev_indices = idx_p

        # -----------------------------
        # TEST 3: BALANCE CHECK (skip 0%)
        # -----------------------------
        if p > 0:
            assert abs(np.mean(y_p) - 0.5) < 0.05, (
                f"Class imbalance too large at {p}%"
            )

        print("✓ PASSED\n")

    print("✅ ALL TESTS PASSED")


# if __name__ == "__main__":
#     test_nested_percent_splits()


if __name__ == '__main__':
    test_nested_percent_splits()
    # dataset = MakeDataset_OpenClose('patients_rest_open_ica', nscc=False)
    # subject = 0
    # for subject in range(19):
    #     dataset.generate_seeded_one_subject_data_and_label_no_setting_self_data_label(subject)
    # train_data, train_label = dataset.get_train_data_subject_independent([0])
    # subject_0_train_data = dataset.data[0, :dataset.n_trials_per_session, :, :, :]
    # subject_0_train_label = dataset.labels[0, :dataset.n_trials_per_session]
    # subject_0_test_label = dataset.labels[0, dataset.n_trials_per_session:]
    #
    # # data, labels = dataset.load_data('x_reshaped.npy', 'y_reshaped.npy')
    # train_data, val_data, test_data, train_label, val_label, test_label = dataset.subject_dependent_holdout_get_normalised_train_val_test_data_pipeline(train_index=[0])
    # train_data, val_data, test_data = dataset.make_train_val_test_dataloaders(32, train_data, val_data, test_data,
    #                                                                           train_label, val_label, test_label)
    # print()
    # print(next(iter(train_data)))
    # print(f'train_data.shape, {train_data.shape}')
    # print(f'train_label.shape, {train_label.shape}')
    # print(f'val_data.shape, {val_data.shape}')
    # print(f'val_label.shape, {val_label.shape}')
    # print(f'test_data.shape, {test_data.shape}')
    # print(f'test_label.shape, {test_label.shape}')
    # print(val_data.shape)
    # print(test_data.shape)




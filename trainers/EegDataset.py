import torch
from torch.utils.data import Dataset, DataLoader
class EegDataset(Dataset):
    # x_tensor: (sample.py, channel, datapoint(feature)) type = tnp
    # y_tensor: (sample.py,) type = np

    def __init__(self, x, y, sampling_rate, augmentation=None):
        # x_tensor = torch.tensor(x, dtype=torch.float)
        # y_tensor = torch.tensor(y, dtype=torch.int64)
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.int64)

        assert self.x.size(0) == self.y.size(0)

        self.sampling_rate = sampling_rate
        self.augmentation = augmentation

    def __getitem__(self, index):
        x = self.x[index]
        # print('x.mean before augmentation', x.mean())
        if self.augmentation is not None:
            x = self.data_augmentation(x)
        return x, self.y[index]

    def __len__(self):
        return len(self.y)

    def data_augmentation(self, one_trial_x, probability=1):
        '''
        Apply a data augmentation method to one trial, with a probability
        :param one_trial_x:
        :param probability:
        :return:
        '''

        if torch.rand(1) < probability:
            method_to_run = getattr(self, self.augmentation)
            one_trial_x = method_to_run(one_trial_x)
            return one_trial_x
        # pass

    import torch

    def make_disturbed_patch(self, one_trial_x):
        """
        Sets a random patch of the input EEG data to zeros.

        Parameters:
        one_trial_x (torch tensor): The input data, shape (1, n_channels, n_timesteps)

        Returns:
        torch tensor:
        """
        _, n_channels, n_timesteps = one_trial_x.size()

        # Randomly select duration of the time factors_of_sample_rate to be distorted
        duration = torch.randint(0, int(0.25 * self.sampling_rate), (1,)).item()
        # duration = 0.5 * self.sampling_rate
        # Randomly select the position where to place this time factors_of_sample_rate
        start_pos = torch.randint(0, n_timesteps - duration, (1,)).item()

        # Randomly select the number of channels to be distorted
        # n_channels: This is the high value.
        # This means the maximum random number generated will be n_channels - 1.
        n_channels_distorted = torch.randint(low=1, high=n_channels//2 + 1, size=(1,)).item()
        channels_distorted = torch.randint(0, n_channels, (n_channels_distorted,))

        # Create a copy of the input data
        # augmented_one_trial_x = one_trial_x.clone()

        # Set the selected patch to zeros
        end_index = start_pos + duration if start_pos + duration < n_timesteps else n_timesteps
        return channels_distorted, start_pos, end_index

    def patch_to_zeros(self, one_trial_x):
        channels_distorted, start_pos, end_index = self.make_disturbed_patch(one_trial_x)
        one_trial_x[0, channels_distorted, start_pos:end_index] = 0
        return one_trial_x

    def patch_add_gaussian_noise(self, one_trial_x):
        channels_distorted, start_pos, end_index = self.make_disturbed_patch(one_trial_x)

        noise_std = torch.rand(1).item() * 2 * one_trial_x.std()
        noise = torch.randn_like(one_trial_x[0, channels_distorted, start_pos:end_index]) * noise_std
        one_trial_x[0, channels_distorted, start_pos:end_index] += noise
        return one_trial_x



    # Example usage:
    # one_trial_x is a single EEG trial with shape (1, n_channels, n_timesteps)
    # augmented_one_trial_x = patch_to_zeros(one_trial_x)

    def hemisphere_perturbation(self, one_trial_x):
        pass

    def random_shift(self, one_trial_x):
        pass

class EegDataset_return_augmented_tgt_with_original(EegDataset):
    def __getitem__(self, item):
        x = self.x[item]
        x_clone = x.clone().detach()
        x = self.data_augmentation(x)

        x = torch.cat((x_clone.unsqueeze(0), x.unsqueeze(0)), dim=0)
        y = torch.cat((self.y[item].unsqueeze(0), self.y[item].unsqueeze(0)), dim=0)


        return x, y

if __name__ == '__main__':
    torch_tensor_to_numpy = lambda x: x.detach().cpu().numpy()

    #generate a test case for EegDataset
    one_trial_x = torch.rand(400, 1, 62, 1000)
    clone_one_trial_x = one_trial_x.clone().detach()
    one_trial_y = torch.ones(400)
    dataset = EegDataset_return_augmented_tgt_with_original(one_trial_x, one_trial_y, 256, augmentation='patch_to_zeros')


    #generate a test case for patch_to_zeros
    # one_trial_x = torch.rand(1, 62, 1000)
    #  = dataset.make_disturbed_patch(one_trial_x)
    dataset[0].shape

    # np_augmented = torch_tensor_to_numpy(augmented_one_trial_x)
    # one_trial_np = torch_tensor_to_numpy(clone_one_trial_x)
    # print(np_augmented[0, channels_distorted, start_pos:end_index] == one_trial_np[0, channels_distorted, start_pos:end_index])
    #
    # print(augmented_one_trial_x)
    # print(augmented_one_trial_x.size())

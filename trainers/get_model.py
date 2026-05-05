
# class Get_Model():
#     '''
#     This class will be used to create a model object.
#     '''
#     def __init__(self, model_name):
#         self.model_name = model_name
#
#     def import_model_from_file(self, *args, **kwargs):
#         '''
#         return the model
#         :param args:
#         :param classes:
#         :param train_data: 4D shaped (nSamples, 1, nChan, nTime)
#         :return:
#         '''
#         model_module = importlib.import_module(f"models.{self.model_name}")
#         model_class = getattr(model_module, self.model_name)
#         model = model_class(*args, **kwargs)
#         return model
#     # ...

import importlib
def get_model(model_name, *args, **kwargs):
    '''
    :param model_name:
    :param folder:
    :param args:
    :param kwargs:
    :return:
    '''
    model_module = importlib.import_module(f"models.{model_name}")
    model_class = getattr(model_module, model_name)
    model = model_class(*args, **kwargs)
    return model

# Example usage:
# model = get_model('dcn', n_channels, n_timesteps, n_classes, args)

if __name__ == '__main__':
    model_name = 'DCN_split_2hemisphere_concat_spatial_dimension'
    from model_configs.DCN_new import Configs
    args = Configs()

    from MakeDataset import MakeDataset
    dataset = MakeDataset('ku54_62chan')

    model = get_model(model_name, args=args, dataset=dataset)
    print(model)

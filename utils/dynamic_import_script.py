
import importlib

def dynamic_import_script(module_name, script_name, class_name, *args, **kwargs):
    '''
    Return the initialised class,
    note: the script_name must be the same as the class file_name

    :param module_name:
    :param script_name:
    :param args:
    :param kwargs:
    :return:
    '''

    script = importlib.import_module(f"{module_name}.{script_name}")
    _class = getattr(script, class_name)
    initialised_class = _class(*args, **kwargs)
    return initialised_class

# Example usage:
# model = get_model('dcn', n_channels, n_timesteps, n_classes, args)

if __name__ == '__main__':
    module_name = 'model_configs'
    script_name = 'DCN_split_2hemisphere_concat_spatial_dimension'
    class_name = 'Configs'
    initialised_class = dynamic_import_script(module_name, script_name, class_name)
    print(initialised_class)


def set_args_to_specific_param_value(parameter_name, parameter_value, args):
    setattr(args, parameter_name, parameter_value)
    return args
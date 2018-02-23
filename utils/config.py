import json
from utils.cl_options import ARGS


class Config(dict):
    def __init__(self, init_dict=None, ARGS=ARGS, config_file_name=None):
        """
        Takes in command-line arguments as well as the configuration file and parses them.
        Parameters
        -----------------
        :param init_dict: <dict>: a dictionary of configuration settings to use
        :param ARGS: <key-val>: the command-line arguments to take, structured like a dictionary
        :param config_file_name: <str>: relative filepath to config file
        """
        super().__init__()
        if init_dict:
            self.load_config(init_dict)
        config_file_name = './configs/' + (config_file_name or (ARGS and ARGS.config) or '')
        config_file_name = config_file_name.replace('configs/configs', 'configs')
        if config_file_name != './configs/':
            self.load_helper(config_file_name, ARGS)
        if ARGS is not None:
            with open('./configs/constants.json') as f:
                self['constants'] = json.load(f)
            with open('./configs/tasks.json') as f:
                self['tasks'] = json.load(f)

    def load_helper(self, path=None, ARGS=None):
        # If no path is specified, load the default
        if path is None:
            with open('./configs/default.json', 'r') as f:
                base_config = json.load(f)
            self.load_config(base_config)
        else:
            with open(path, 'r') as f:
                config_dict = json.load(f)

            # If a parent configuration exists, use the parent configuration
            if config_dict['parentConfig']:
                self.load_helper(('./configs/' + config_dict['parentConfig'])
                                 .replace('./configs/configs/', './configs/'))
            self.load_config(config_dict, ARGS)

    def load_config(self, json_dict: dict, ARGS=None):
        """
        Loads in a configuration from a dictionary
        :param json_dict: <key-val>: a key-val data structure
        :return: None
        """
        for key, value in json_dict.items():
            if key == 'gpu' and ARGS and ARGS.gpu != -1:
                self['gpu'] = ARGS.gpu
            if type(value) is dict:
                if key in self:
                    self[key].load_config(value)
                else:
                    self[key] = Config(init_dict=value, ARGS=None, config_file_name=None)
            else:
                self[key] = value
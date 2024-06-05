from utils.path_utils import PathUtils

import yaml
import os


class ConfigUtils(object):

    def __init__(self):
        pass

    @staticmethod
    def get_config_dict(config_file_name):
        config_file_full_path = os.path.join(PathUtils.CONFIG_HOME_PATH, config_file_name)
        return yaml.load(open(config_file_full_path, "r"), Loader=yaml.Loader)

    @staticmethod
    def get_basic_config():
        configs = ["process_control_config.yaml"]
        config_dict = {}
        for config_file in configs:
            config_dict.update(ConfigUtils.get_config_dict(config_file))
        return config_dict

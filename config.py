import logging
import yaml
import os
logger = logging.getLogger('crgan')


class Config:
    """Loads parameters from config.yaml into global object
    """
    def __init__(self, path_to_config):
        self.path_to_config = path_to_config
        if os.path.isfile(path_to_config):
            pass
        else:
            self.path_to_config = '{}'.format(self.path_to_config)
        with open(self.path_to_config, "r") as f:
            self.dictionary = yaml.load(f.read(), Loader=yaml.FullLoader)
        for k, v in self.dictionary.items():
            setattr(self, k, v)

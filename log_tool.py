import logging
import os
current_path = os.path.abspath(__file__)
if not os.path.exists(os.path.dirname(current_path) + "/results"):
    os.makedirs(os.path.dirname(current_path) + "/results")


def setup_logging(id):
    id_file = os.path.dirname(current_path) + "/results/" + id
    file_list = [id_file, id_file+"/logs", id_file+"/models", id_file+"/y_hat", id_file+"/figures"]
    if not os.path.exists(id_file):
        for i in range(len(file_list)):
            os.makedirs(file_list[i])
    logger = logging.getLogger('crgan')
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(id_file + "/logs/" + id + ".txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


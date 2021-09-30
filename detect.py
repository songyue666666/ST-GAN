import log_tool
import config
import preprocess
import model
import score
import helpers
import test
import time
import os
import logging
logger = logging.getLogger('crgnn')
current_path = os.path.abspath(__file__)


def detect():
    id = time.strftime("%Y-%m-%d_%H.%M.%S")
    log_tool.setup_logging(id) # create folder “result/…” and save logs
    # load config parameters
    paras = config.Config("config.yaml")
    # preprocess the original data, return data and label
    data = preprocess.Dataprocess(id, paras)
    data.load_data()
    if paras.test:   # only calcaulate the indicator
        cur_test = test.Test(data, paras)
        cur_test.test()
    else:
        start = time.time() 
        cur_model = model.crnn_model(paras, id, data)
        if paras.predict:
            anom_score = score.Score(cur_model, paras, data)
            anom_score.batch_predict()
        end = time.time()
        print('time is', end-start, 's')


if __name__ == "__main__":
    detect()

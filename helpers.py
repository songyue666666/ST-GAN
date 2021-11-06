from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import time
import heapq
logger = logging.getLogger('crgan')


def cos_similarity(gen_samples, samples):
    # caculate the cosine similarity of generated samples and original samples
    res_list = []
    for i in range(len(gen_samples)):
        gen_sample = np.reshape(np.array(gen_samples[i]), (len(gen_samples[i]), -1))
        sample = np.reshape(np.array(samples[i]), (len(samples[i]), -1))
        res_para = []
        for j in range(gen_sample.shape[1]):
            pairs = [gen_sample[:][j], sample[:][j]]
            res = cosine_similarity(pairs)[0][1]
            res_para.append(np.abs(res))
        # top5 = heapq.nsmallest(5, range(len(res_para)), res_para.__getitem__)
        # print(top5)
        res_list.append(res_para)
    return res_list


def load_para(self, path):
    if os.path.exists(path + '.npy'):
        info = np.load(path + '.npy', allow_pickle=True).item()
        for k, v in info.items():
            setattr(self, k, v)


def indicator(pred, label):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    acc = 0
    pre = 0
    recall = 0
    F1 = 0
    for i in range(len(pred)):
        if pred[i] == label[i]:
            acc += 1
        if label[i] == 1:
            if pred[i] == 1:
                TP += 1
            else:
                FN += 1
        else:
            if pred[i] == 1:
                FP += 1
            else:
                TN += 1
    acc = acc / len(pred)
    logger.info("accuracy: %.4f" % acc)
    if TP+FP != 0:
        pre = TP / (TP + FP)
        logger.info("precision: %.4f" % pre)
    if TP+FN != 0:
        recall = TP / (TP + FN)
        logger.info("recall: %.4f" % recall)
    if pre + recall != 0:
        F1 = 2 * pre * recall / (pre + recall)
        logger.info("F1: %.4f" % F1)
    return pre, recall, F1
    # print("accuracy: ", acc, "precision: ", pre, "recall: ", recall, "F1: ", F1)


def plotting1(id, name, score, threshold, label=[]):
    plt.figure(figsize=(12, 6))
    normal_x = []
    normal_y = []
    outlier_x = []
    outlier_y = []
    for i in range(len(score)):
        if score[i] > threshold:
            outlier_x.append(i)
            outlier_y.append(score[i])
        else:
            normal_x.append(i)
            normal_y.append(score[i])
    plt.scatter(outlier_x, outlier_y, s=5, c="#ff1212", label='Outlier', zorder=2)
    plt.scatter(normal_x, normal_y, s=5, label='Normal', zorder=2)
    if label:
        for i in range(len(label)):
            if label[i] == 1:
                plt.vlines(i, 0.35, 0.65, colors="seashell", alpha=0.05, zorder=1)
    # plt.plot(score, label=name, linewidth=1)
    plt.plot(threshold*np.ones(len(score)), label='Threshold', linewidth=3, color='#FFA500')
    font = {"family": "Times New Roman", "size": 18}
    plt.xlabel("Timestamps", font)
    plt.ylabel("Anomaly score", font)
    plt.tick_params(labelsize=15)
    plt.legend(prop=font)
    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()
    plt.savefig("results/" + id + "/figures/" + name + ".png")
    plt.show()


def plotting2(id, name, score, threshold):
    plt.figure(figsize=(12, 6))
    plt.plot(score, label=name, linewidth=1)
    plt.plot(threshold*np.ones(len(score)), label='threshold', linewidth=3, color='#FFA500')
    font = {"family": "Times New Roman", "size": 18}
    plt.xlabel("Time steps", font)
    plt.ylabel("Value", font)
    plt.tick_params(labelsize=15)
    plt.legend(prop=font)
    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()
    plt.savefig("results/" + id + "/figures/" + name + ".png")
    plt.show()




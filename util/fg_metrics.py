import numpy as np
import scipy.misc
from scipy import linalg as linAlg

from IPython import embed

# def accuracy(output, target, topk=(1,)):
#     """ From The PyTorch ImageNet example """
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res


"""
    Checks label class location in predictions
        -- assuming each class predictions has a unique probability
"""
def kPosition(pred, obj_class):
    preds = np.argsort(pred)[::-1] # sort predictions by descending order
    return np.where(preds == obj_class)[0][0]


class fg_metrics(object):

    def __init__(self, num_classes = 500, data_split = 'test'):
        self.results_dict   = dict()
        self.num_classes    = num_classes
        self.data_split     = data_split

    """
        Updates the keypoint dictionary
        params:     unique_id       unique id of each instance (NAME_objc#_kpc#)
                    predictions     the predictions for each vector
    """
    def update_dict(self, unique_id, predictions):
        """Log a scalar variable."""
        if type(predictions) == int:
            predictions = [predictions]
            labels      = [labels]

        for i in range(0, len(unique_id)):
            image       = unique_id[i].split('_objc')[0]
            obj_class   = int(unique_id[i].split('_objc')[1].split('_kpc')[0])
            kp_class    = int(unique_id[i].split('_objc')[1].split('_kpc')[1])

            if image in self.results_dict.keys():
                self.results_dict[image]['preds'][kp_class] = predictions[i]
            else:
                self.results_dict[image] = {'class' : obj_class, 'preds' : {kp_class : predictions[i]} }


    def metrics(self, unique = False):

        type_correct_1  = np.zeros(self.num_classes, dtype=np.float32)
        type_correct_5  = np.zeros(self.num_classes, dtype=np.float32)
        type_correct_10 = np.zeros(self.num_classes, dtype=np.float32)
        type_total      = np.zeros(self.num_classes, dtype=np.float32)

        for image in self.results_dict.keys():
            curr_total  = 0.
            curr_1      = 0.
            curr_5      = 0.
            curr_10     = 0.

            for kp in self.results_dict[image]['preds'].keys():

                obj_class = self.results_dict[image]['class']
                pred      = self.results_dict[image]['preds'][kp]

                pred_pos = kPosition(  pred, obj_class)

                curr_total += 1.

                if pred_pos <= 1:
                    curr_1   += 1.
                    curr_5   += 1.
                    curr_10  += 1.
                elif pred_pos <= 5:
                    curr_5   += 1.
                    curr_10  += 1.
                elif pred_pos <= 10:
                    curr_10  += 1.

            if unique:
                curr_total  = curr_total    / curr_total
                curr_1      = curr_1        / curr_total
                curr_5      = curr_5        / curr_total
                curr_10     = curr_10       / curr_total


            type_total[obj_class]       += curr_total
            type_correct_1[obj_class]   += curr_1
            type_correct_5[obj_class]   += curr_5
            type_correct_10[obj_class]  += curr_10


        type_accuracy   = np.zeros(self.num_classes, dtype=np.float16)
        for i in range(0, self.num_classes):
            if type_total[i] > 0:
                type_correct_1[i]   = type_correct_1[i]   / type_total[i]
                type_correct_5[i]   = type_correct_5[i]   / type_total[i]
                type_correct_10[i]  = type_correct_10[i]  / type_total[i]

        # self.calculate_performance_baselines()
        return type_correct_1,type_correct_5,type_correct_10, type_total

    def save_dict(self, dict_name):
        np.save(dict_name + "_" + self.data_split + '_kp_dict.npy', self.results_dict)

    def calculate_performance_baselines(self, mode = 'real'):
        assert False, "To be done! "
        pass

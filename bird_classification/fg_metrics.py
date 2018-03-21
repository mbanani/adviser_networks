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

        # max, median, mean, min
        correct_1   = [0., 0., 0., 0.]
        correct_5   = [0., 0., 0., 0.]
        correct_10  = [0., 0., 0., 0.]
        total_uni   = 0.

        kp_counter = []

        for image in self.results_dict.keys():
            curr_total  = 0.
            curr_1      = 0.
            curr_5      = 0.
            curr_10     = 0.


            self.results_dict[image]['correct_1']  = []
            self.results_dict[image]['correct_5']  = []
            self.results_dict[image]['correct_10'] = []
            self.results_dict[image]['pred_order'] = {}

            kp_counter.append(len(self.results_dict[image]['preds'].keys()))

            for kp in self.results_dict[image]['preds'].keys():

                obj_class = self.results_dict[image]['class']
                pred      = self.results_dict[image]['preds'][kp]

                pred_pos = kPosition(  pred, obj_class)
                self.results_dict[image]['pred_order'][kp] = pred_pos

                curr_total += 1.

                if pred_pos <= 1:
                    curr_1   += 1.
                    curr_5   += 1.
                    curr_10  += 1.
                    self.results_dict[image]['correct_1'].append(1)
                    self.results_dict[image]['correct_5'].append(1)
                    self.results_dict[image]['correct_10'].append(1)
                elif pred_pos <= 5:
                    curr_5   += 1.
                    curr_10  += 1.
                    self.results_dict[image]['correct_1'].append(0)
                    self.results_dict[image]['correct_5'].append(1)
                    self.results_dict[image]['correct_10'].append(1)
                elif pred_pos <= 10:
                    curr_10  += 1.
                    self.results_dict[image]['correct_1'].append(0)
                    self.results_dict[image]['correct_5'].append(0)
                    self.results_dict[image]['correct_10'].append(1)
                else:
                    self.results_dict[image]['correct_1'].append(0)
                    self.results_dict[image]['correct_5'].append(0)
                    self.results_dict[image]['correct_10'].append(0)



            if unique:
                curr_total  = curr_total    / curr_total
                curr_1      = curr_1        / curr_total
                curr_5      = curr_5        / curr_total
                curr_10     = curr_10       / curr_total


            # Stuff for baselines
            total_uni       += 1.
            correct_1[0]    +=  np.max(np.asarray(self.results_dict[image]['correct_1']))
            correct_1[1]    +=  np.median(np.asarray(self.results_dict[image]['correct_1']))
            correct_1[2]    +=  np.mean(np.asarray(self.results_dict[image]['correct_1']))
            correct_1[3]    +=  np.min(np.asarray(self.results_dict[image]['correct_1']))


            correct_5[0]    +=  np.max(np.asarray(self.results_dict[image]['correct_5']))
            correct_5[1]    +=  np.median(np.asarray(self.results_dict[image]['correct_5']))
            correct_5[2]    +=  np.mean(np.asarray(self.results_dict[image]['correct_5']))
            correct_5[3]    +=  np.min(np.asarray(self.results_dict[image]['correct_5']))


            correct_10[0]    +=  np.max(np.asarray(self.results_dict[image]['correct_10']))
            correct_10[1]    +=  np.median(np.asarray(self.results_dict[image]['correct_10']))
            correct_10[2]    +=  np.mean(np.asarray(self.results_dict[image]['correct_10']))
            correct_10[3]    +=  np.min(np.asarray(self.results_dict[image]['correct_10']))




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

        # print baselines
        correct_1   = [correct_1[i]  * 100. / total_uni for i in range(0, len(correct_1)) ]
        correct_5   = [correct_5[i]  * 100. / total_uni for i in range(0, len(correct_5)) ]
        correct_10  = [correct_10[i] * 100. / total_uni for i in range(0, len(correct_10))]

        print "Metric \t\t Max \t Median\t Mean \t Min "
        print "Acc@1  \t\t %.2f \t %.2f \t %.2f \t %.2f " % (correct_1[0], correct_1[1], correct_1[2], correct_1[3])
        print "Acc@5  \t\t %.2f \t %.2f \t %.2f \t %.2f " % (correct_5[0], correct_5[1], correct_5[2], correct_5[3])
        print "Acc@10 \t\t %.2f \t %.2f \t %.2f \t %.2f " % (correct_10[0], correct_10[1], correct_10[2], correct_10[3])

        # print keypoint stat
        kp_counter = np.asarray(kp_counter)
        print ""
        print "Keypoint Statistics : Mean: %.2f, Median: %.2f, Max: %.2f, Min: %.2f, Std. Dev: %.2f" % (np.mean(kp_counter),
                                                                                                        np.median(kp_counter),
                                                                                                        np.max(kp_counter),
                                                                                                        np.min(kp_counter),
                                                                                                        np.std(kp_counter),
                                                                                                        )

        # self.calculate_performance_baselines()
        return type_correct_1,type_correct_5,type_correct_10, type_total, [correct_1, correct_5, correct_10]

    def save_dict(self, dict_name):
        np.save(dict_name + "_" + self.data_split + '_kp_dict.npy', self.results_dict)

    def calculate_performance_baselines(self, mode = 'real'):
        assert False, "To be done! "
        pass

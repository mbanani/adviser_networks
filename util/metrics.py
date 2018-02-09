import numpy as np
import scipy.misc
from scipy import linalg as linAlg

from IPython import embed


class adviser_metrics(object):

    def __init__(self, performance_dict, regression=False):
        self.results_dict  = { 4: dict(), 5: dict(), 8: dict() }
        self.performance_dict = performance_dict
        self.class_ranges  = [0, 0,  0,  0,  0, 12, 24, 24, 24, 34, 34,  34,  34]
        self.threshold = np.pi / 6.0
        self.first_time = True
        self.regression = False
    """
        Updates the keypoint dictionary
        params:     unique_id       unique id of each instance (NAME_objc#_kpc#)
                    predictions     the predictions for each vector
    """
    def update_dict(self, predictions, labels, obj_classes, keys):
        """Log a scalar variable."""
        for i in range(0, len(keys)):
            key         = keys[i]
            obj_class   = obj_classes[i]

            start_index = self.class_ranges[obj_class]
            end_index   = self.class_ranges[obj_class + 1]

            pred_probs   = predictions[i][start_index:end_index]
            label_probs  = labels[i][start_index:end_index]

            self.results_dict[obj_class][key] = {'pred' : pred_probs, 'label' : label_probs }

    def reset(self):
        self.results_dict  = { 4: dict(), 5: dict(), 8: dict() }


    def metrics(self, unique = False):

        type_geo_dist   = [ [] for x in range(0, 3)]
        type_correct    = np.zeros(3, dtype=np.float32)
        type_total      = np.zeros(3, dtype=np.float32)

        obj_classes = [4,5,8]

        qualitative_dict = dict()

        for index in range(0,3):
            for key in self.results_dict[obj_classes[index]].keys():
                if self.regression:
                    rank_pred   = np.argsort(self.results_dict[obj_classes[index]][key]['pred'])
                    rank_label  = np.argsort(self.results_dict[obj_classes[index]][key]['label'])
                else:
                    rank_pred   = np.argsort(self.results_dict[obj_classes[index]][key]['pred'])[::-1]
                    rank_label  = np.argsort(self.results_dict[obj_classes[index]][key]['label'])[::-1]

                kp_ind      = self.performance_dict[key]['geo_dist'].keys()

                rank_pred  += self.class_ranges[obj_classes[index]]
                rank_label += self.class_ranges[obj_classes[index]]
                # kp_ind     += self.class_ranges[obj_classes[index]]

                top_real    = [k for k in rank_pred if k in set(kp_ind)][0]
                top_all     = rank_pred[0]

                type_correct[index] += self.performance_dict[key]['correct'][top_real]
                type_geo_dist[index].append(self.performance_dict[key]['geo_dist'][top_real])
                type_total[index]   += 1.0

                qualitative_dict[key] = { 'geo_dist' : self.performance_dict[key]['geo_dist'], 'top_real' : top_real}


        type_accuracy  = np.around([ 100.             * np.mean([ num < self.threshold for num in type_geo_dist[i] ]) for i in range(0, 3) ], decimals = 2)
        type_medError  = np.around([ (180. / np.pi ) * np.median(type_geo_dist[i]  ) for i in range(0, 3) ], decimals = 2)

        # type_accuracy = [type_correct[index] / type_total[index] if type_correct[index] > 0 else -1 for index in range(0,3)]
        # type_geo_dist = [ np.median(type_geo_dist[index]) for index in range(0,3)]


        print "==========================================================================="
        print "Accuracy  : ", type_accuracy , " -- mean : ", np.round(np.mean(type_accuracy ), decimals = 2)
        print "Geo Dist  : ", type_medError , " -- mean : ", np.round(np.mean(type_medError ), decimals = 2)
        print "Latex     : ", type_accuracy[0],  ' &', type_accuracy[1],  ' &', type_accuracy[2],  ' &', np.round(np.mean(type_accuracy), decimals = 2),
        print           ' &', type_medError[0],  ' &', type_medError[1],  ' &', type_medError[2],  ' &', np.round(np.mean(type_medError), decimals = 2)
        print " --------------------- "
        all_keys = [self.results_dict[obj_classes[index]].keys() for index in range(0,3)]
        self.calculate_performance_baselines(keys = all_keys[0] + all_keys[1] + all_keys[2])
        return type_accuracy, type_total, type_geo_dist, qualitative_dict



    def calculate_performance_baselines(self, mode = 'real', keys = None):


        if self.first_time:
            worst_baseline  = [ [] for x in range(0, 3)]
            best_baseline   = [ [] for x in range(0, 3)]
            mean_baseline   = [ [] for x in range(0, 3)]
            median_baseline = [ [] for x in range(0, 3)]
            freq_prior_baseline = [ [] for x in range(0, 3)]
            perf_prior_baseline = [ [] for x in range(0, 3)]

            freq_prior  = [0.] * 34
            perf_prior  = [0.] * 34

            #iterate over batch
            for image in self.performance_dict.keys():
                if image in keys:
                    obj_cls = self.performance_dict[image]['class']
                    if obj_cls == 4:
                        obj_cls = 0
                    elif obj_cls == 5:
                        obj_cls = 1
                    elif obj_cls == 8:
                        obj_cls = 2

                    perf = [self.performance_dict[image]['geo_dist'][kp] for kp in self.performance_dict[image]['geo_dist'].keys()]

                    best_baseline[obj_cls  ].append(np.min(perf))
                    worst_baseline[obj_cls ].append(np.max(perf))
                    mean_baseline[obj_cls  ].append(np.mean(perf))
                    median_baseline[obj_cls].append(np.median(perf))

                    best_perf = np.min(perf)
                    for kp in self.performance_dict[image]['geo_dist'].keys():
                        freq_prior[kp] += 1.
                        if self.performance_dict[image]['geo_dist'][kp] == best_perf:
                            perf_prior[kp] += 1.

            bus = ['body_back_left_lower', 'body_back_left_upper', 'body_back_right_lower',
                            'body_back_right_upper', 'body_front_left_upper', 'body_front_right_upper',
                            'body_front_left_lower', 'body_front_right_lower', 'left_back_wheel',
                            'left_front_wheel', 'right_back_wheel', 'right_front_wheel']

            car = ['left_front_wheel', 'left_back_wheel', 'right_front_wheel',
                            'right_back_wheel', 'upper_left_windshield', 'upper_right_windshield',
                            'upper_left_rearwindow', 'upper_right_rearwindow', 'left_front_light',
                            'right_front_light', 'left_back_trunk', 'right_back_trunk']

            motorbike = ['back_seat', 'front_seat', 'head_center', 'headlight_center',
                            'left_back_wheel', 'left_front_wheel', 'left_handle_center',
                            'right_back_wheel', 'right_front_wheel', 'right_handle_center']

            # calculate priors
            # perf_rank = np.argsort(perf_prior)
            # freq_rank = np.argsort(freq_prior)
            perf_rank = np.argsort(perf_prior)[::-1]
            freq_rank = np.argsort(freq_prior)[::-1]
            print "Bus"
            class_range = range(0, 12)
            print "Perf Prior: ", [bus[x] for x in perf_rank if x in class_range]
            print "Freq Prior: ", [bus[x] for x in freq_rank if x in class_range]
            print "Car"
            class_range = range(12, 24)
            print "Perf Prior: ", [car[x - 12] for x in perf_rank if x in class_range]
            print "Freq Prior: ", [car[x - 12] for x in freq_rank if x in class_range]
            print "Motorbike"
            class_range = range(24, 34)
            print "Perf Prior: ", [motorbike[x - 24] for x in perf_rank if x in class_range]
            print "Freq Prior: ", [motorbike[x - 24] for x in freq_rank if x in class_range]

            for image in self.performance_dict.keys():
                if image in keys:
                    kp_ind = self.performance_dict[image]['geo_dist'].keys()
                    top_freq_kp    = [k for k in freq_rank if k in set(kp_ind)][0]
                    top_perf_kp    = [k for k in perf_rank if k in set(kp_ind)][0]

                    obj_cls = self.performance_dict[image]['class']
                    if obj_cls == 4:
                        obj_cls = 0
                    elif obj_cls == 5:
                        obj_cls = 1
                    elif obj_cls == 8:
                        obj_cls = 2

                    freq_prior_baseline[obj_cls].append(self.performance_dict[image]['geo_dist'][top_freq_kp])
                    perf_prior_baseline[obj_cls].append(self.performance_dict[image]['geo_dist'][top_perf_kp])

            # calculate baselines
            self.accuracy_best    = np.around([ 100. * np.mean([ num < self.threshold for num in best_baseline[i]   ]) for i in range(0, 3) ], decimals = 2)
            self.accuracy_worst   = np.around([ 100. * np.mean([ num < self.threshold for num in worst_baseline[i]  ]) for i in range(0, 3) ], decimals = 2)
            self.accuracy_mean    = np.around([ 100. * np.mean([ num < self.threshold for num in mean_baseline[i]   ]) for i in range(0, 3) ], decimals = 2)
            self.accuracy_median  = np.around([ 100. * np.mean([ num < self.threshold for num in median_baseline[i] ]) for i in range(0, 3) ], decimals = 2)
            self.accuracy_freq_p  = np.around([ 100. * np.mean([ num < self.threshold for num in freq_prior_baseline[i] ]) for i in range(0, 3) ], decimals = 2)
            self.accuracy_perf_p  = np.around([ 100. * np.mean([ num < self.threshold for num in perf_prior_baseline[i] ]) for i in range(0, 3) ], decimals = 2)

            self.medError_best    = np.around([ (180. / np.pi ) * np.median(best_baseline[i]  ) for i in range(0, 3) ], decimals = 2)
            self.medError_worst   = np.around([ (180. / np.pi ) * np.median(worst_baseline[i] ) for i in range(0, 3) ], decimals = 2)
            self.medError_mean    = np.around([ (180. / np.pi ) * np.median(mean_baseline[i]  ) for i in range(0, 3) ], decimals = 2)
            self.medError_median  = np.around([ (180. / np.pi ) * np.median(median_baseline[i]) for i in range(0, 3) ], decimals = 2)
            self.medError_perf_p  = np.around([ (180. / np.pi ) * np.median(perf_prior_baseline[i]  ) for i in range(0, 3) ], decimals = 2)
            self.medError_freq_p  = np.around([ (180. / np.pi ) * np.median(freq_prior_baseline[i]) for i in range(0, 3) ], decimals = 2)

            len_best_error      = [len(best_baseline[i])  for i in range(0, 3) ]
            # print "Length of each vector: ", len_best_error, "Number of elements: ", sum(len_best_error)
            self.first_time = False

        print "Accuracy "
        print "best      : ", self.accuracy_best   , " -- mean : ", np.round(np.mean(self.accuracy_best   ), decimals = 2)
        print "worst     : ", self.accuracy_worst  , " -- mean : ", np.round(np.mean(self.accuracy_worst  ), decimals = 2)
        print "mean      : ", self.accuracy_mean   , " -- mean : ", np.round(np.mean(self.accuracy_mean   ), decimals = 2)
        print "median    : ", self.accuracy_median , " -- mean : ", np.round(np.mean(self.accuracy_median ), decimals = 2)
        print "freq prior: ", self.accuracy_freq_p , " -- mean : ", np.round(np.mean(self.accuracy_freq_p ), decimals = 2)
        print "perf prior: ", self.accuracy_perf_p , " -- mean : ", np.round(np.mean(self.accuracy_perf_p ), decimals = 2)
        print "Median Error "
        print "best      : ", self.medError_best   , " -- mean : ",  np.round(np.mean(self.medError_best   ), decimals = 2)
        print "worst     : ", self.medError_worst  , " -- mean : ",  np.round(np.mean(self.medError_worst  ), decimals = 2)
        print "mean      : ", self.medError_mean   , " -- mean : ",  np.round(np.mean(self.medError_mean   ), decimals = 2)
        print "median    : ", self.medError_median , " -- mean : ",  np.round(np.mean(self.medError_median ), decimals = 2)
        print "freq prior: ", self.medError_freq_p , " -- mean : ", np.round(np.mean(self.medError_freq_p ), decimals = 2)
        print "perf prior: ", self.medError_perf_p , " -- mean : ", np.round(np.mean(self.medError_perf_p ), decimals = 2)

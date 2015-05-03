# This Python file uses the following encoding: utf-8
# !/usr/local/bin/python3.4
# ###################################################
# <Copyright (C) 2012, 2013, 2014, 2015 Yeray Alvarez Romero>
# This file is part of MULLPY.
####################################################
import copy
import re
import sys

import numpy as np

from mullpy.auxiliar import AutoVivification

####################################################


class Statistics:
    """
    The class where are defined all statistics functions as goodness, standard deviation or mean square error.
    All the information relative to the classifiers is saved on the class structure indexable by name
    """

    def __init__(self):
        """
        Initialize the internal structure as AutoVivification class
        """
        self.measures = AutoVivification()

    #####################################################
    @staticmethod
    def change_ranges(value, **kwargs):
        """
        Project a given value, from old ranges to new ranges
        """
        if len(kwargs.keys()) != 4:
            raise ValueError("Change ranges need 4 parameters")

        old_min = kwargs["oldMin"]
        old_max = kwargs["oldMax"]
        new_max = kwargs["newMax"]
        new_min = kwargs["newMin"]

        old_range = old_max - old_min
        new_range = new_max - new_min
        old_value = value

        return (((old_value - old_min) * new_range) / old_range) + new_min

    #############################################
    def rms(self, classifier_name, context, information, pattern_kind):
        """
        Calculate all rms to different patterns kind relative to the classifier.
        """
        list_outputs_classifier = information.info[classifier_name]["continuous_outputs"][pattern_kind]
        self.measures[classifier_name]["rms"][pattern_kind] = 0.0
        pattern = copy.deepcopy(context["patterns"].patterns[classifier_name][pattern_kind])
        #Difference between desired outputs(patterns) and the real outputs
        classes_texts = context["classifiers"][classifier_name]["classes_names"]
        len_inputs = len(pattern[0]) - len(classes_texts)
        for outputs, desired in zip(list_outputs_classifier, pattern):
            if context["classifiers"][classifier_name]["patterns"]["range"] is not [0, 1]:
                for i, desire in enumerate(desired[len_inputs:]):
                    desired[len_inputs:][i] = \
                        self.change_ranges(
                            desire,
                            oldMin=context["classifiers"][classifier_name]["patterns"]["range"][0],
                            oldMax=context["classifiers"][classifier_name]["patterns"]["range"][1],
                            newMin=0,
                            newMax=1)

            self.measures[classifier_name]["rms"][pattern_kind] += sum(0.5 * (desired[len_inputs:] - outputs) ** 2)
        self.measures[classifier_name]["rms"][pattern_kind] /= float(len(pattern))

    #############################################

    @staticmethod
    def discretize_outputs(value):
        """
        Used like a lambda function
        """
        if value == -1:
            return 0.
        return value

    #############################################

    def initialize_goodness(self, context, classifier_name, instances_number, classes_names):
        #Initialize the structure of goodness values.
        for values_kind in ['fp', 'fn', 'tp', 'tn']:
            self.measures[classifier_name]["matrix"][values_kind] = \
                np.zeros([instances_number, len(classes_names)], dtype=np.float16)

            self.measures[classifier_name][values_kind] = 0.0
            for class_text in classes_names:
                self.measures[classifier_name][class_text][values_kind] = 0.0

    #############################################

    def build_list_oracle_outputs(self, classifier_name):
        self.measures[classifier_name]["matrix"]["oracle_outputs"] = \
            self.measures[classifier_name]["matrix"]["tp"] + self.measures[classifier_name]["matrix"]["tn"]

    #############################################

    def goodness(self, context, classifier_name, list_outputs_classifier, pattern_outputs):
        """
        Calculate the goodness of the classifier. It contain an error formula to penalize more the instances
        with one class, and less with more classes presents in the same instances.
        It is a generalization of the multiclass problem.
        Calculate the goodness in terms of FP, FN, TP, TN and different kinds of error as global error,
        false positive error, false negative error.
        """
        #TODO:Change the input parameters from list outputs and patterns to Information
        if not len(pattern_outputs):
            raise NameError('Statistics doesnt get the patterns of the classifier %s correctly at dir %s' %
                            (classifier_name, context["classifiers"][classifier_name]["paths"]["patterns"]))
        if not len(list_outputs_classifier):
            raise NameError('Statistics doesnt get the outputs of the classifier %s correctly' % classifier_name)
        if len(list_outputs_classifier) != len(pattern_outputs):
            raise NameError('Different lengths in patterns and outputs on classifier %s' % classifier_name)

        #############################################
        #To improve code readability
        classes_names = context["classifiers"][classifier_name]["classes_names"]
        instances_number = float(len(pattern_outputs))
        len_inputs = len(pattern_outputs[0]) - len(classes_names)

        self.initialize_goodness(context, classifier_name, int(instances_number), classes_names)

        #############################################
        #Measure the error by instance
        for instance in range(int(instances_number)):
            #Number of classes present in an instance. For multilabel problems
            for output_index, class_text in enumerate(classes_names):
                output_wanted = pattern_outputs[instance][len_inputs:][output_index]
                output = list_outputs_classifier[instance][output_index]

                if output == (-1.):
                    output = 0.
                if output_wanted == (-1.):
                    output_wanted = 0.

                #If there is an error
                if output_wanted != output:
                    #If output wanted was activated means a FN
                    if output_wanted == 1.0:
                        #FN
                        self.measures[classifier_name]["matrix"]['fn'][instance][output_index] = 1.
                    else:
                        # If not output wanted was activated means a FP
                        self.measures[classifier_name]["matrix"]['fp'][instance][output_index] = 1.
                #No error
                else:
                    #TP
                    if output_wanted == 1.0:
                        self.measures[classifier_name]["matrix"]['tp'][instance][output_index] = 1.
                    #TN
                    else:
                        self.measures[classifier_name]["matrix"]['tn'][instance][output_index] = 1.
        #############################################
        #The goodness values in terms of sum of the instances
        for good in self.measures[classifier_name]["matrix"].keys():
            self.measures[classifier_name][good] = np.sum(self.measures[classifier_name]["matrix"][good])
            for output_index, class_text in enumerate(classes_names):
                self.measures[classifier_name][class_text][good] = \
                    np.sum(self.measures[classifier_name]["matrix"][good], 0)[output_index]

    #########################################################################################

    def error_fn(self, classifier_name, context, information, pattern_kind):
        pattern_outputs = context["patterns"].patterns[classifier_name][pattern_kind]
        classes_names = context["classifiers"][classifier_name]["classes_names"]

        self.measures[classifier_name]["error_fn"] = 0.0
        for class_text in classes_names:
            self.measures[classifier_name][class_text]["error_fn"] = 0.0

        for output_index, class_text in enumerate(classes_names):
            num_instances_of_the_class = np.sum([self.measures[classifier_name]["matrix"]['tp'][i][output_index] +
                                                 self.measures[classifier_name]["matrix"]['fn'][i][output_index]
                                                 for i in range(len(pattern_outputs))])

            #The error depends on the number of instances of it class and on the total number of classes
            if len(classes_names) == 1:
                self.measures[classifier_name][class_text]["error_fn"] = \
                    0.5 * np.sum(self.measures[classifier_name]["matrix"]["fn"], 0)[output_index] / \
                    num_instances_of_the_class
                self.measures[classifier_name][class_text]["error_fn"] = \
                    0.5 * np.sum(self.measures[classifier_name]["matrix"]["fn"], 0)[output_index] / \
                    (float(len(pattern_outputs)) - num_instances_of_the_class)
            else:
                self.measures[classifier_name][class_text]["error_fn"] = \
                    (np.sum(self.measures[classifier_name]["matrix"]["fn"], 0)[output_index] /
                     num_instances_of_the_class) / len(classes_names)

        for class_text in classes_names:
            self.measures[classifier_name]["error_fn"] += self.measures[classifier_name][class_text]["error_fn"]

    #########################################################################################

    def error_fp(self, classifier_name, context, information, pattern_kind):
        pattern_outputs = context["patterns"].patterns[classifier_name][pattern_kind]
        classes_names = context["classifiers"][classifier_name]["classes_names"]

        self.measures[classifier_name]["error_fp"] = 0.0
        for class_text in classes_names:
            self.measures[classifier_name][class_text]["error_fp"] = 0.0

        for output_index, class_text in enumerate(classes_names):
            num_instances_of_the_class = np.sum([self.measures[classifier_name]["matrix"]['tp'][i][output_index] +
                                                 self.measures[classifier_name]["matrix"]['fn'][i][output_index]
                                                 for i in range(len(pattern_outputs))])

            #The error depends on the number of instances of it class and on the total number of classes
            if len(classes_names) == 1:
                self.measures[classifier_name][class_text]["error_fp"] = \
                    0.5 * np.sum(self.measures[classifier_name]["matrix"]["FP"], 0)[output_index] / \
                    num_instances_of_the_class
                self.measures[classifier_name][class_text]["error_fp"] = \
                    0.5 * np.sum(self.measures[classifier_name]["matrix"]["FP"], 0)[output_index] / \
                    (float(len(pattern_outputs)) - num_instances_of_the_class)
            else:
                self.measures[classifier_name][class_text]["error_fp"] = \
                    (np.sum(self.measures[classifier_name]["matrix"]["FP"], 0)[output_index] /
                     num_instances_of_the_class) / len(classes_names)

        for class_text in classes_names:
            self.measures[classifier_name]["error_fp"] += self.measures[classifier_name][class_text]["error_fp"]

    #########################################################################################

    def error(self, classifier_name, context, information, pattern_kind):
        """
        Calculate the errors of the classifier given by name.
        This error compensates the minority class by dividing each error class by the number of instances
        of that class, and finally divided by the number of classes.
        """
        self.error_fp(classifier_name, context, information, pattern_kind)
        self.error_fn(classifier_name, context, information, pattern_kind)

        for class_text in context["classifiers"][classifier_name]["classes_names"]:
            self.measures[classifier_name][class_text]["error"] = \
                self.measures[classifier_name][class_text]["error_fp"] + \
                self.measures[classifier_name][class_text]["error_fn"]

        self.measures[classifier_name]["error"] = \
            self.measures[classifier_name]["error_fp"] + self.measures[classifier_name]["error_fn"]

    #########################################################################################

    def balanced_accuracy(self, classifier_name, context, *args):
        self.tnr(classifier_name, context)
        self.tpr(classifier_name, context)

        for class_text in context["classifiers"][classifier_name]["classes_names"]:
            self.measures[classifier_name][class_text]["balanced_accuracy"] = \
                (self.measures[classifier_name][class_text]["tpr"] +
                 self.measures[classifier_name][class_text]["tnr"]) / 2.

        self.measures[classifier_name]["balanced_accuracy"] = \
            np.mean([self.measures[classifier_name][x]["balanced_accuracy"] for x in
                     context["classifiers"][classifier_name]["classes_names"]])

    #########################################################################################

    def g_means(self, classifier_name, context, *args):
        """
        Geometric mean as the sqrt of the sensibility*specificity
        """
        self.tnr(classifier_name, context)
        self.tpr(classifier_name, context)

        self.measures[classifier_name]["g_means"] = np.sqrt(np.dot(self.measures[classifier_name]["tnr"],
                                                                   self.measures[classifier_name]["tpr"]))

        for class_text in context["classifiers"][classifier_name]["classes_names"]:
            self.measures[classifier_name][class_text]["g_means"] = \
                np.sqrt(np.dot(self.measures[classifier_name][class_text]["tnr"],
                               self.measures[classifier_name][class_text]["tpr"]))

    #########################################################################################

    def tnr(self, classifier_name, context, *args):
        """
        True Negative Rate
        """
        fp = self.measures[classifier_name]["fp"]
        tn = self.measures[classifier_name]["tn"]

        if tn + fp > 0:
            self.measures[classifier_name]["tnr"] = np.divide(tn, tn + fp)
        else:
            self.measures[classifier_name]["tnr"] = 0.0

        for class_text in context["classifiers"][classifier_name]["classes_names"]:
            fp = self.measures[classifier_name][class_text]["fp"]
            tn = self.measures[classifier_name][class_text]["tn"]

            if tn + fp > 0:
                self.measures[classifier_name]["tnr"] = np.divide(tn, tn + fp)
            else:
                self.measures[classifier_name]["tnr"] = 0.0

    #########################################################################################

    def tpr(self, classifier_name, context, *args):
        """
        True Positive Rate
        """
        tp = self.measures[classifier_name]["tp"]
        fn = self.measures[classifier_name]["fn"]

        if tp + fn > 0:
            self.measures[classifier_name]["tpr"] = np.divide(tp, tp + fn)
        else:
            self.measures[classifier_name]["tpr"] = 0.0

        for class_text in context["classifiers"][classifier_name]["classes_names"]:
            tp = self.measures[classifier_name][class_text]["tp"]
            fn = self.measures[classifier_name][class_text]["fn"]

            if tp + fn > 0:
                self.measures[classifier_name]["tpr"] = np.divide(tp, tp + fn)
            else:
                self.measures[classifier_name]["tpr"] = 0.0

    #########################################################################################
    @staticmethod
    def get_ytrue_ypred(context, information, classifier_name, pattern_kind):
        len_classes = len(context["classifiers"][context["classifier_list"][0]]["classes_names"])
        len_inputs = len(context["patterns"].patterns[classifier_name][pattern_kind][0]) - len_classes
        y_true = list(context["patterns"].patterns[classifier_name][pattern_kind][:, range(len_inputs,
                                                                                           len_inputs +
                                                                                           len_classes)])
        y_pred = information.info[classifier_name]["continuous_outputs"][pattern_kind]
        return y_true, y_pred

    #########################################################################################

    def explained_variance_score(self, classifier_name, context, information, pattern_kind):
        from sklearn.metrics import explained_variance_score

        y_true, y_pred = self.get_ytrue_ypred(context, information, classifier_name, pattern_kind)
        self.measures[classifier_name]["explained_variance_score"] = \
            explained_variance_score(y_true, y_pred)

    #########################################################################################

    def mean_absolute_error(self, classifier_name, context, information, pattern_kind):
        from sklearn.metrics import mean_absolute_error

        y_true, y_pred = self.get_ytrue_ypred(context, information, classifier_name, pattern_kind)
        self.measures[classifier_name]["explained_variance_score"] = \
            mean_absolute_error(y_true, y_pred)

    #########################################################################################

    def mean_squared_error(self, classifier_name, context, information, pattern_kind):
        from sklearn.metrics import mean_squared_error

        y_true, y_pred = self.get_ytrue_ypred(context, information, classifier_name, pattern_kind)
        self.measures[classifier_name]["mean_squared_error"] = \
            mean_squared_error(y_true, y_pred)

    #########################################################################################

    def r2_score(self, classifier_name, context, information, pattern_kind):
        from sklearn.metrics import r2_score

        y_true, y_pred = self.get_ytrue_ypred(context, information, classifier_name, pattern_kind)
        self.measures[classifier_name]["r2_score"] = \
            r2_score(y_true, y_pred)

    #########################################################################################
    @staticmethod
    def confusion_matrix(classifier_name, context, information, pattern_kind):
        from sklearn.metrics import confusion_matrix

        confusion_matrix(context["patterns"].patterns[classifier_name][pattern_kind],
                         information.info[classifier_name]["discretized_outputs"][pattern_kind],
                         context["classifiers"][classifier_name]["classes_names"])

    #########################################################################################
    @staticmethod
    def matthews_corrcoef(classifier_name, context, information, pattern_kind):
        from sklearn.metrics import matthews_corrcoef

        matthews_corrcoef(context["patterns"].patterns[classifier_name][pattern_kind],
                          information.info[classifier_name]["discretized_outputs"][pattern_kind])

    #########################################################################################

    def hamming_loss(self, classifier_name, context, information, pattern_kind):
        from sklearn.metrics import hamming_loss

        self.measures[classifier_name]["hamming_loss"] = \
            hamming_loss(
                context["patterns"].patterns[classifier_name][pattern_kind],
                information.info[classifier_name]["discretized_outputs"][pattern_kind])

    #########################################################################################

    def kappa(self, classifier_name, *args):
        self.measures[classifier_name]["kappa"] = \
            self.measures[classifier_name]["matrix"]['tp'] + self.measures[classifier_name]["matrix"]['tn']

    #########################################################################################

    def f_measure(self, classifier_name, *args):
        self.recall(classifier_name, *args)
        self.accuracy(classifier_name, *args)

        self.measures[classifier_name]["f_measure"] = \
            (2 * self.measures[classifier_name]["recall"] * self.measures[classifier_name]["accuracy"]) / \
            (self.measures[classifier_name]["recall"] + self.measures[classifier_name]["accuracy"])

    #########################################################################################

    def accuracy(self, classifier_name, *args):
        self.measures[classifier_name]["accuracy"] = \
            self.measures[classifier_name]['tp'] / (
                self.measures[classifier_name]['tp'] + self.measures[classifier_name]['fp'])

    #########################################################################################

    def error_rate(self, classifier_name, *args):
        self.measures[classifier_name]["error_rate"] = \
            self.measures[classifier_name]["matrix"]['tp'] + self.measures[classifier_name]["matrix"]['tn'] / \
                                                             (np.sum([self.measures[classifier_name]["matrix"][goodness]
                                                                      for goodness in
                                                                      self.measures[classifier_name]["matrix"].keys()]))

    #########################################################################################

    def accuracy_rate(self, classifier_name, *args):
        self.measures[classifier_name]["accuracy_rate"] = \
            self.measures[classifier_name]["matrix"]['fp'] + self.measures[classifier_name]["matrix"]['fn'] / \
                                                             (np.sum(
                                                                 [self.measures[classifier_name]["matrix"][goodness] for
                                                                  goodness in
                                                                  self.measures[classifier_name]["matrix"].keys()]))

    #########################################################################################

    def recall(self, classifier_name, *args):
        self.tpr(classifier_name, *args)
        self.measures[classifier_name]["recall"] = self.measures[classifier_name]["tpr"]

    #########################################################################################

    def fn_rate(self, classifier_name, *args):
        self.measures[classifier_name]["fn_rate"] = self.measures[classifier_name]["matrix"]['fn'] / (
            self.measures[classifier_name]["matrix"]['tp'] + self.measures[classifier_name]["matrix"]['fn'])

    #########################################################################################

    def fp_rate(self, classifier_name, *args):
        self.measures[classifier_name]["fp_rate"] = self.measures[classifier_name]["matrix"]['fp'] / (
            self.measures[classifier_name]["matrix"]['tn'] + self.measures[classifier_name]["matrix"]['fp'])

    #########################################################################################

    def auc(self, classifier_name, context, information, pattern_kind):
        from sklearn.metrics import roc_auc_score

        classes_names = context["classifiers"][classifier_name]["classes_names"]
        inputs = len(context["patterns"].patterns[classifier_name][pattern_kind][0]) - len(classes_names)
        self.measures[classifier_name]["auc"] = 0.0
        for i, class_name in enumerate(classes_names):
            self.measures[classifier_name][class_name]["auc"] = \
                roc_auc_score(context["patterns"].patterns[classifier_name][pattern_kind][:, inputs + i],
                              information.info[classifier_name]["continuous_outputs"][pattern_kind][:, i])
            self.measures[classifier_name]["auc"] += self.measures[classifier_name][class_name]["auc"]

        self.measures[classifier_name]["auc"] = np.divide(np.mean(self.measures[classifier_name]["auc"]),
                                                          len(classes_names))

    #########################################################################################

    def std(self, classifier_name, context, *args):
        """
        Calculate the standard deviation of the classifier passed as args, for each kind of error.
        Thus, there is a std for false positive error, another to false positive error, etc.
        """

        self.measures[classifier_name]['dt_efp'] = np.std(self.measures[classifier_name]["matrix"]['efp'])
        self.measures[classifier_name]['dt_efn'] = np.std(self.measures[classifier_name]["matrix"]['efn'])
        self.measures[classifier_name]['dt_e'] = np.std(self.measures[classifier_name]["matrix"]['efp'] +
                                                        self.measures[classifier_name]["matrix"]['efn'])

        for output_index, class_text in enumerate(context["classifiers"][classifier_name]["classes_names"]):
            self.measures[classifier_name][class_text]['dt_efp'] = \
                np.std(self.measures[classifier_name]["matrix"]["efp"], 0)[output_index]
            self.measures[classifier_name][class_text]['dt_efn'] = \
                np.std(self.measures[classifier_name]["matrix"]["efn"], 0)[output_index]
            self.measures[classifier_name][class_text]['dt_e'] = \
                np.std(self.measures[classifier_name]["matrix"]["e"], 0)[output_index]

    #############################################
    @staticmethod
    def __build_multiple_name(sub_list):
        name = ""
        if type(sub_list) != list:
            for x_tuple in sub_list:
                name = "+".join([x for x in x_tuple])
        else:
            for i, name_i in enumerate(sub_list):
                if i == len(sub_list) - 1:
                    name += name_i
                else:
                    name += name_i + "+"
        return name

    #############################################

    def correctly_classified(self, sub_list):
        correctly_classified = np.zeros(len(self.measures[sub_list[0]]["matrix"]["oracle_outputs"]))
        for i in range(len(self.measures[sub_list[0]]["matrix"]["oracle_outputs"])):
            for j, classifier_name in enumerate(sub_list):
                if (np.array(self.measures[classifier_name]["matrix"]["oracle_outputs"][i]) == np.ones(
                        len(self.measures[classifier_name]["matrix"]["oracle_outputs"][i]))).all():
                    correctly_classified[i] += 1

        return correctly_classified

    #############################################

    def interrater_agreement_k_non_pairwise(self, context, sub_list):
        error = 0.0
        correctly_classified = self.correctly_classified(sub_list)
        p = np.sum([self.measures[x]['E'] for x in self.measures if 'E' in self.measures[x]]) / \
            (len(sub_list) * len(self.measures[sub_list[0]]["matrix"]["oracle_outputs"]))

        for i in range(len(correctly_classified)):
            error += correctly_classified[i] * (len(sub_list) - correctly_classified[i])

        if p == 0.0:
            p = np.exp(100)

        error /= len(self.measures[sub_list[0]]["matrix"]["oracle_outputs"]) * (len(sub_list) - 1) * p * (1 - p)
        return 1 - error

    #############################################

    def difficulty(self, context, sub_list):
        error = 0.0
        correctly_classified = self.correctly_classified(sub_list)
        mean_errors = np.mean(correctly_classified)
        for i in range(len(correctly_classified)):
            error += np.power((correctly_classified[i] - (correctly_classified[i] / mean_errors)), 2)
        error /= (len(self.measures[sub_list[0]]["matrix"]["oracle_outputs"]) * np.power(len(sub_list), 2))
        return 1. - error

    #############################################

    def kohavi_wolpert(self, context, sub_list):
        error = 0.0
        correctly_classified = self.correctly_classified(sub_list)
        for i in range(len(correctly_classified)):
            error += correctly_classified[i] * (len(sub_list) - correctly_classified[i])
        error /= len(sub_list)

        error /= (len(self.measures[sub_list[0]]["matrix"]["oracle_outputs"]) * np.power(len(sub_list), 2))
        return error

    #############################################

    def entropy(self, context, sub_list):
        Error = 0.0
        correctly_classified = self.correctly_classified(sub_list)
        for i in range(len(correctly_classified)):
            Error += (min(correctly_classified[i], len(sub_list) - correctly_classified[i])
                      /
                      (len(sub_list) - np.ceil(len(sub_list) / 2.)))

        Error /= len(self.measures[sub_list[0]]["matrix"]["oracle_outputs"])
        return Error

    #############################################

    def diversity_non_pairwise_structure(self, context, function, classifier_list):
        for i, classifier_name in enumerate(classifier_list):
            if context["interactive"]["activate"]:
                sys.stdout.write("\r{0}>".format("Completed:%f%%" % ((float(i) / len(classifier_list)) * 100)))
                sys.stdout.flush()

            # name = self.__build_multiple_name(sub_list)

            self.measures[classifier_name][function] = \
                getattr(self, function)(context, context["classifiers"][classifier_name]["classifiers"])

    #############################################

    def diversity_pairwise_structure(self, context, function, classifier_list):
        for i, classifier_1 in enumerate(classifier_list):
            if context["interactive"]["activate"]:
                sys.stdout.write("\r{0}>".format("Completed:%f%%" % ((float(i) / len(classifier_list)) * 100)))
                sys.stdout.flush()
            for classifier_2 in context["classifiers"].keys():

                if "pairwise_diversity" in self.measures[classifier_2].keys() and function in \
                        self.measures[classifier_2][
                            "pairwise_diversity"].keys() and classifier_1 in \
                        self.measures[classifier_2]["pairwise_diversity"][
                            function].keys():

                    self.measures[classifier_1]["pairwise_diversity"][function][classifier_2] = \
                        self.measures[classifier_2]["pairwise_diversity"][function][classifier_1]

                else:
                    self.measures[classifier_1]["pairwise_diversity"][function][classifier_2] = \
                        getattr(self, function)(classifier_1, classifier_2, context)

            vector = [self.measures[classifier_1]["pairwise_diversity"][function][x] for x in
                      self.measures[classifier_1]["pairwise_diversity"][function].keys() if x != classifier_1]
            self.measures[classifier_1]["pairwise_diversity"][function]["mean"] = np.mean(vector)
            self.measures[classifier_1]["pairwise_diversity"][function]["median"] = np.median(vector)
            self.measures[classifier_1]["pairwise_diversity"][function]["std"] = np.std(vector)
            self.measures[classifier_1]["pairwise_diversity"][function]["variance"] = np.var(vector)

    #############################################

    def error_correlation(self, classifier_1, classifier_2, context):
        return np.corrcoef(self.measures[classifier_1]["matrix"]["e"], self.measures[classifier_2]["matrix"]["e"])[0][1]

    #############################################

    def n01(self, classifier_1, classifier_2):
        counter = 0
        for a, b in zip(self.measures[classifier_1]["matrix"]["oracle_outputs"],
                        self.measures[classifier_2]["matrix"]["oracle_outputs"]):
            if np.sum(a) < len(a) and np.sum(b) == len(b):
                counter += 1
        return counter

    #############################################

    def n10(self, classifier_1, classifier_2):
        counter = 0
        for a, b in zip(self.measures[classifier_1]["matrix"]["oracle_outputs"],
                        self.measures[classifier_2]["matrix"]["oracle_outputs"]):
            if np.sum(a) == len(a) and np.sum(b) < len(b):
                counter += 1
        return counter

    #############################################

    def n11(self, classifier_1, classifier_2):
        counter = 0
        for a, b in zip(self.measures[classifier_1]["matrix"]["oracle_outputs"],
                        self.measures[classifier_2]["matrix"]["oracle_outputs"]):
            if (a == b).all() and np.sum(a) == len(a):
                counter += 1
        return counter

    #############################################

    def n00(self, classifier_1, classifier_2):
        counter = 0
        for a, b in zip(self.measures[classifier_1]["matrix"]["oracle_outputs"],
                        self.measures[classifier_2]["matrix"]["oracle_outputs"]):
            if np.sum(b) < len(b) and np.sum(a) < len(a):
                counter += 1
        return counter

    #############################################

    def _n_values(self, classifier_1, classifier_2, context):
        #this results may be divided
        n11 = None
        n00 = None
        n10 = None
        n01 = None

        if context["results"]["to_file"]["diversity_study"]["exact_match"]:
            n11 = self.n11(classifier_1, classifier_2)
            n00 = self.n00(classifier_1, classifier_2)
            n10 = self.n10(classifier_1, classifier_2)
            n01 = self.n01(classifier_1, classifier_2)

        elif context["results"]["to_file"]["diversity_study"]["by_class"]:
            # TODO: change this part
            for i in range(len(self.measures[classifier_1]["matrix"]["oracle_outputs"][0])):
                n11 = sum([1 if x == y and x == 1 else 0 for x, y in
                           zip(self.measures[classifier_1]["matrix"]["oracle_outputs"],
                               self.measures[classifier_2]["matrix"]["oracle_outputs"])])
                n00 = sum(
                    [1 if x == y and x == 0 else 0 for x, y in
                     zip(self.measures[classifier_1]["matrix"]["oracle_outputs"],
                         self.measures[classifier_2]["matrix"]["oracle_outputs"])])
                n01 = sum([1 if x != y and x == 0 and y == 1 else 0 for x, y in
                           zip(self.measures[classifier_1]["matrix"]["oracle_outputs"],
                               self.measures[classifier_2]["matrix"]["oracle_outputs"])])
                n10 = sum([1 if x != y and x == 1 and y == 0 else 0 for x, y in
                           zip(self.measures[classifier_1]["matrix"]["oracle_outputs"],
                               self.measures[classifier_2]["matrix"]["oracle_outputs"])])
        else:
            raise ValueError("No option selected in diversity study: by class or by exact match")

        return {"n11": n11, "N00": n00, "N01": n01, "N10": n10}

    #############################################

    def interrater_agreement_k(self, classifier_1, classifier_2, context):
        values = self._n_values(classifier_1, classifier_2, context)
        denominator = ((values["N11"] + values["N10"]) * (values["N01"] + values["N00"])) + \
                      ((values["N11"] + values["N01"]) * (values["N10"] + values["N00"]))
        numerator = 2 * ((values["N11"] * values["N00"]) - (values["N01"] * values["N10"]))
        return numerator / denominator

    #############################################

    def q_statistic(self, classifier_1, classifier_2, context):
        values = self._n_values(classifier_1, classifier_2, context)
        denominator = values["N11"] * values["N00"] + values["N01"] * values["N10"]
        if not denominator:
            denominator = 1
        return (values["N11"] * values["N00"] - values["N01"] * values["N10"]) / denominator

    #############################################

    def coefficient_p(self, classifier_1, classifier_2, context):
        values = self._n_values(classifier_1, classifier_2, context)
        denominator = np.sqrt((values["N11"] + values["N10"]) * (values["N01"] + values["N00"]) * (
            values["N11"] + values["N01"]) * (values["N10"] + values["N00"]))
        if not denominator:
            denominator = 1
        return (values["N11"] * values["N00"] - values["N01"] * values["N10"]) / denominator

    #############################################

    def disagreement(self, classifier_1, classifier_2, context):
        values = self._n_values(classifier_1, classifier_2, context)
        denominator = values["N11"] * values["N00"] + values["N01"] + values["N10"]
        if not denominator:
            denominator = 1
        return (values["N01"] + values["N10"]) / denominator

    #############################################

    def double_fault(self, classifier_1, classifier_2, context):
        values = self._n_values(classifier_1, classifier_2, context)
        denominator = values["N11"] + values["N10"] + values["N01"] + values["N00"]
        if not denominator:
            denominator = 1
        return values["N00"] / denominator

    ################################################################

    def configuration_evaluation(self, context, classifier_name, information):
        """
        To be reconstructed into a abstraction model. Initialize the information of each classifier.
        """
        #information_class.automatic_threshold_determine(context,classifier_name)
        pattern_kind = "validation"
        self.rms(classifier_name, context, information, pattern_kind)

        name = classifier_name[:re.search(r'[A-Za-z]+[0-9]*', classifier_name).end()]
        neurons = context["classifiers"][classifier_name]["configuration"]["neurons"][0]

        if len(self.measures[name]["evaluation"][neurons].keys()):
            self.measures[name]["evaluation"][neurons]['rms'].append(
                self.measures[classifier_name]['rms'][pattern_kind])
            self.measures[name]["evaluation"][neurons]['names'].append(classifier_name)
        else:
            self.measures[name]["evaluation"][neurons]['rms'] = []
            self.measures[name]["evaluation"][neurons]['rms'].append(
                self.measures[classifier_name]['rms'][pattern_kind])
            self.measures[name]["evaluation"][neurons]['names'] = []
            self.measures[name]["evaluation"][neurons]['names'].append(classifier_name)

    ####################################################

    def best_choice(self):
        """
        Select the best configuration of a NN classifier with the class attributes information.
        """
        for name in sorted([x for x in self.measures.keys() if "evaluation" in self.measures[x].keys()]):
            self.measures[name]["selection"]["rms"] = [99999.0]
            self.measures[name]["selection"]["neurons"]["hidden"] = [0]
            self.measures[name]["selection"]["name"] = [""]

            for neuron in sorted(self.measures[name]["evaluation"].keys()):
                self.measures[name]["selection"]["neurons"][neuron]["amount"] = 0
                rms_list, names_list = (list(t) for t in zip(*sorted(zip(self.measures[
                                                                             name]["evaluation"][neuron]['rms'],
                                                                         self.measures[name]["evaluation"][neuron][
                                                                             'names']))))

                mean_rms = np.mean(self.measures[name]["evaluation"][neuron]['rms'])

                if mean_rms < self.measures[name]["selection"]["rms"][0]:
                    self.measures[name]["selection"]["rms"] = [mean_rms]
                    self.measures[name]["selection"]["neurons"]["hidden"] = [neuron]
                    self.measures[name]["selection"]["neurons"][neuron]["amount"] = 1
                    self.measures[name]["selection"]["names"] = \
                        [self.measures[name]["evaluation"][neuron]['names'][self.measures[name]["evaluation"][neuron][
                            'rms'].index(sorted(
                                self.measures[name]["evaluation"][neuron][
                                    'rms'])[0])]]

                elif mean_rms == self.measures[name]["selection"]["rms"][0]:
                    self.measures[name]["selection"]["rms"].append(mean_rms)
                    self.measures[name]["selection"]["neurons"]["hidden"].append(neuron)
                    for i in range(len(self.measures[name]["evaluation"][neuron]['rms'])):
                        if rms_list[i] == rms_list[0]:
                            self.measures[name]["selection"]["names"].append(names_list[i])
                            self.measures[name]["selection"]["neurons"][neuron]["amount"] += 1

    ################################################################
    @staticmethod
    def pre_forecasting_statistic(context, classifier_name, information, pattern_kind):
        len_classes = len(context["classifiers"][classifier_name]["classes_names"])
        len_inputs = len(context["patterns"].patterns[classifier_name][pattern_kind][0]) - len_classes
        classifier_outputs = information.info[classifier_name]["continuous_outputs"][pattern_kind]
        classifier_patterns = \
            context["patterns"].patterns[classifier_name][pattern_kind][:, (len_inputs - 1, len_inputs)]
        len_patterns = len(context["patterns"].patterns[classifier_name][pattern_kind])
        d_change_pred = np.zeros(len_patterns)
        d_change_true = np.zeros(len_patterns)

        for i, instance, outputs in zip(range(len_patterns), classifier_patterns, classifier_outputs):
            d_change_true[i] = instance[1] - instance[0]
            d_change_pred[i] = outputs[0] - instance[0]
        return d_change_pred, d_change_true

    ################################################################

    def tendency_accuracy(self, classifier_name, context, information, pattern_kind):
        """
        Calculates the number of tends hits on a regression problem.
        The regression tolerance is a parameter added to avoid the errors due to overflow
        :param classifier_name:
        :param context:
        :param information:
        :param pattern_kind:
        :return:
        """
        array_change_pred, array_change_true = Statistics().pre_forecasting_statistic(context,
                                                                                     classifier_name,
                                                                                     information,
                                                                                     pattern_kind)
        hits = np.zeros(len(array_change_pred))
        for i, d_change_pred, d_change_true in zip(range(len(array_change_pred)), array_change_pred, array_change_true):
            if d_change_pred * d_change_true > 0.0:
                hits[i] = 1.
            elif d_change_pred * d_change_true == 0.0:
                hits[i] = 1.
            else:
                if np.sqrt(np.abs(d_change_pred * d_change_true)) < context["regression_tolerance_tendency"]:
                    hits[i] = 1.
                else:
                    hits[i] = 0.

        self.measures[classifier_name]["tendency_accuracy"] = np.mean(hits)

    #########################################################################################

    def mase(self, classifier_name, context, information, pattern_kind):
        """
        Mean Absolute error. Returns the inverse of the mase with a denominator that sums 1 to the error.
        It is intended to give an error between 1 and 0, where the 1 is the lowest error and 0.0 the highest in order
        to be compatible ordering different measures in the presentations.
        :param classifier_name:
        :param context:
        :param information:
        :param pattern_kind:
        :return:
        """
        array_change_pred, array_change_true = self.pre_forecasting_statistic(context, classifier_name,
                                                                              information, pattern_kind)

        self.measures[classifier_name]["mase"] = np.divide(np.mean(np.absolute(array_change_pred)),
                                                           np.mean(np.absolute(array_change_true)))

        ################################################################

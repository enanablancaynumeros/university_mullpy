# This Python file uses the following encoding: utf-8
# !/usr/local/bin/python3.3
####################################################
# <Copyright (C) 2012, 2013, 2014, 2015 Yeray Alvarez Romero>
# This file is part of MULLPY.
####################################################
import copy
from itertools import permutations
from mullpy.statistics import Statistic

import numpy as np

from mullpy.auxiliar import AutoVivification


class ClassifiersInfo:
    """
    Abstraction model to combine the information of different kinds of classifiers.
    Constructed over the real_output function defined by each kind of classifier.
    Context parameters are modified across this functions.
    """

    def __init__(self):
        """
        Internal structure as AutoVivification class
        """
        self.info = AutoVivification()

    #####################################################
    def build_real_outputs(self, context, classifier_name, pattern_text):
        """
        Construct the array for the test and validation outputs in real values
        If the classifier works in a different range than zero-one the continuous outputs are projected linearly
        """
        values_list = context["classifiers"][classifier_name]["instance"].real_outputs(
            context, classifier_name, context["patterns"].patterns[classifier_name][pattern_text])
        transformed_list = []

        for i, classes_list in enumerate(values_list):
            temp = np.zeros(len(classes_list), dtype=np.float32)
            for j, value in enumerate(classes_list):
                if context["classifiers"][classifier_name]["patterns"]["range"] is not [0, 1]:
                    value = Statistic().change_ranges(
                        value,
                        oldMin=context["classifiers"][classifier_name]["patterns"]["range"][0],
                        oldMax=context["classifiers"][classifier_name]["patterns"]["range"][1],
                        newMin=0,
                        newMax=1)
                temp[j] = value
            transformed_list.append(temp)
        self.info[classifier_name]["continuous_outputs"][pattern_text] = np.asarray(transformed_list)
        #####################################################

    def discretize_outputs(self, context, classifier_name, pattern_kind):
        """
            Some classifiers needs to build discretized outputs. Calculate the discretized outputs for each kind of pattern.
            """
        #If more performance is needed, substitute pattern_texts by validation or the pattern wanted
        num_instances = len(self.info[classifier_name]["continuous_outputs"][pattern_kind])
        num_classes = len(context["classifiers"][classifier_name]["classes_names"])
        self.info[classifier_name]["discretized_outputs"][pattern_kind] = np.zeros(
            [num_instances, num_classes], dtype=np.float32)

        for i, instance in enumerate(self.info[classifier_name]["continuous_outputs"][pattern_kind]):
            for j, output in enumerate(instance):
                if output >= context["classifiers"][classifier_name]["thresholds"]["value"][j]:
                    self.info[classifier_name]["discretized_outputs"][pattern_kind][i][j] = 1.

                    #####################################################

    def discretize_outputs_for_roc(self, context, classifier_name, classifier_outputs):
        """
            Discretize many points as indicated in classifier definition to paint the curve roc.
            Its used to calculate the threshold optimum value too.
            """
        if not len(classifier_outputs):
            raise NameError(
                'discretize_outputs_for_roc is not getting the outputs of the classifier %s correctly' %
                classifier_name)

        thresholds = self.get_thresholds(context, classifier_name)
        #Generate the discretized outputs for the values generated

        #Generate the discretized outputs for the values generated
        for threshold in thresholds:
            self.info[classifier_name]["roc_outputs"][threshold] = np.zeros(
                [len(classifier_outputs), len(classifier_outputs[0])], dtype=np.float32)
            for i, instance in enumerate(classifier_outputs):
                for j, output in enumerate(instance):
                    if output >= threshold:
                        self.info[classifier_name]["roc_outputs"][threshold][i][j] = 1.

    #####################################################

    def get_thresholds(self, context, classifier_name):
        range_lower = context["classifiers"][classifier_name]["thresholds"]["range"][0]
        range_upper = context["classifiers"][classifier_name]["thresholds"]["range"][1]
        repetitions = context["classifiers"][classifier_name]["thresholds"]["repetitions"]

        if context["results"]["roc"]["activate"] == 1:
            #From the first point indicated in thresholds range to the second one, generating many points as
            #  indicated in repetitions
            thresholds = np.linspace(range_lower, range_upper, num=repetitions)
        else:
            interval = context["classifiers"][classifier_name]["thresholds"]["interval_from_limits"]
            #Calculating the threshold optimum value
            if interval != 0.0:
            #Indicating a interval from the limits to increment the performance if you have prior
            #  knowledge about the value
                thresholds_a = np.linspace(range_lower, range_lower + interval, num=repetitions)
                thresholds_b = np.linspace(range_upper, range_upper - interval, num=repetitions)
                thresholds = np.concatenate((thresholds_a, thresholds_b), axis=0)
            else:
                #Generate a wide range of elements from the limits without restrictions
                thresholds = np.linspace(range_lower, range_upper, num=repetitions)
        return thresholds
        #####################################################

    def threshold_determination(self, context, classifier_name, patterns_outputs):
        """
            With the discretized outputs for roc values, determine the best values for the threshold.
            """
        statistic_class = Statistic()
        #Aux structures
        threshold_list = AutoVivification()
        minimum_error = AutoVivification()

        for class_text in context["classifiers"][classifier_name]["classes_names"]:
        #Initialize the aux structures
            threshold_list[class_text] = []
            minimum_error[class_text] = float('inf')
            self.info[classifier_name][class_text]["threshold"]["medium"] = float('inf')
            self.info[classifier_name][class_text]["threshold"]["minimum"] = float('inf')
            self.info[classifier_name][class_text]["threshold"]["maximum"] = float('-inf')
            #For each value of threshold generated
        for threshold in self.info[classifier_name]["roc_outputs"]:
            #Calculate the goodness of the classifier
            statistic_class.goodness(context, classifier_name, self.info[classifier_name]["roc_outputs"][threshold],
                                     patterns_outputs)
            for class_text in context["classifiers"][classifier_name]["classes_names"]:
                error = 0.0
                for function in context["classifiers"][classifier_name]["thresholds"]["metric"]:
                    getattr(statistic_class, function)(classifier_name, context, self, "validation")
                    error += statistic_class.measures[classifier_name][class_text][function]
                #If we find a minimum error, we save it
                if error < minimum_error[class_text]:
                    minimum_error[class_text] = error
                    threshold_list[class_text] = [threshold]
                    #When we find a new global minimum we have to reset the list
                    #And save it again
                    #If there is a tie in terms of goodness, save all the range of values with the minimum error
                if error == minimum_error[class_text]:
                    threshold_list[class_text].append(threshold)
                    #Determine different kinds of thresholds

                if len(threshold_list[class_text]) == 0:
                    raise ValueError("There is no threshold selected")
        return threshold_list

    #####################################################

    def threshold_calculation(self, context, classifier_name, threshold_list):

        for class_text in context["classifiers"][classifier_name]["classes_names"]:
            threshold_list[class_text] = sorted(threshold_list[class_text])

            #Minimum
            for threshold in threshold_list[class_text]:
                if 0.0 < threshold < 1.0:
                    self.info[classifier_name][class_text]["threshold"]["minimum"] = threshold
                    break

            if self.info[classifier_name][class_text]["threshold"]["minimum"] == float('inf'):
                self.info[classifier_name][class_text]["threshold"]["minimum"] = 0.001

            #Maximum
            for threshold in reversed(threshold_list[class_text]):
                if 0.0 < threshold < 1.0:
                    self.info[classifier_name][class_text]["threshold"]["maximum"] = threshold
            if self.info[classifier_name][class_text]["threshold"]["maximum"] == float('-inf'):
                self.info[classifier_name][class_text]["threshold"]["maximum"] = 0.999

            #Medium
            medium = threshold_list[class_text][int(len(threshold_list[class_text]) / 2)]
            if 0.0 < medium < 1.0:
                self.info[classifier_name][class_text]["threshold"]["medium"] = medium
            else:
                self.info[classifier_name][class_text]["threshold"]["medium"] = 0.5

    #####################################################

    def automatic_threshold_determine(self, context, classifier_name):
        """
        Automatic threshold determination implies to substitute the value indicated in the user given
         configuration of the classifier.
        """
        output_kind = "continuous_outputs"
        #No sense to determine the threshold with discretized outputs
        classes_text = context["classifiers"][classifier_name]["classes_names"]

        self.discretize_outputs_for_roc(context, classifier_name, self.info[classifier_name][output_kind]["validation"])

        threshold_list = self.threshold_determination(context, classifier_name,
                                                      context["patterns"].patterns[classifier_name]["validation"])
        self.threshold_calculation(context, classifier_name, threshold_list)

        #Save a list of optimum thresholds, depending of the preference (minimum, middle or maximum value).
        # One per output
        context["classifiers"][classifier_name]["thresholds"]["value"] = \
            [self.info[classifier_name][component]["threshold"][context["classifiers"][classifier_name][
                "thresholds"]["threshold_kind"]] for component in classes_text]

        print("Thresholds of %s are %s" % (classifier_name, ["%.5f" % x for x in
                                                             context["classifiers"][classifier_name]["thresholds"][
                                                                 "value"]]))

    #####################################################

    def build_roc(self, context, classifier_name, pattern_outputs):
        """
            Build the tpr and tnr for the ROC curve of the classifier given.
            """
        len_outputs = len(self.info[classifier_name]["roc_outputs"].keys())
        self.info[classifier_name]['tpr'] = np.zeros(len_outputs, dtype=np.float32)
        self.info[classifier_name]['tnr'] = np.zeros(len_outputs, dtype=np.float32)
        for component in context["classifiers"][classifier_name]["classes_names"]:
            self.info[classifier_name][component]['tpr'] = np.zeros(len_outputs, dtype=np.float32)
            self.info[classifier_name][component]['tnr'] = np.zeros(len_outputs, dtype=np.float32)
        statistic_class = Statistic()

        for i, threshold in enumerate(sorted(self.info[classifier_name]["roc_outputs"])):
            statistic_class.goodness(context, classifier_name, self.info[classifier_name]["roc_outputs"][threshold],
                                     pattern_outputs)
            statistic_class.tpr(classifier_name, context)
            statistic_class.tnr(classifier_name, context)

    #######################################################################

    def classes_error(self, context, classifier_name):

        self.info[classifier_name]["selection_errors"] = []

        statistic_class = Statistic()
        values = AutoVivification()
        pattern_kind = context["pattern_kind"]
        outputs_kind = context["outputs_kind"]

        if classifier_name in context["classifier_list"]:
            temporal_patterns = copy.deepcopy(context["patterns"].patterns[classifier_name][pattern_kind])
        else:
            original = self.info[classifier_name][outputs_kind][pattern_kind]
            original_pattern_ref = context["patterns"].patterns[classifier_name][pattern_kind]

        for i in range(1, len(context["classifiers"][classifier_name]["classes_names"])):
            temp = [1] * i
            temp.extend([-1] * (len(context["classifiers"][classifier_name]["classes_names"]) - i))
            values[i] = [temp]
            for new in permutations(values[i][0]):
                if new not in values[i]:
                    values[i].append(new)

            if classifier_name in context["classifier_list"]:
                context["patterns"].modify_patterns_temporally(classifier_name, pattern_kind,
                                                               context["patterns"].filter_classes(classifier_name,
                                                                                                  pattern_kind,
                                                                                                  values[i]))
                self.build_real_outputs(context, classifier_name, pattern_kind)
                self.discretize_outputs(context, classifier_name, pattern_kind)
                ref_patterns = context["patterns"].patterns[classifier_name][pattern_kind]
            else:
                positions = [position for position, instance in enumerate(original_pattern_ref) if
                             instance[1] in values[i]]
                self.info[classifier_name][outputs_kind][pattern_kind] = \
                    [original[i] for i in range(len(original)) if i in positions]
                ref_patterns = [original_pattern_ref[i] for i in range(len(original_pattern_ref)) if i in positions]

            statistic_class.goodness(context, classifier_name, self.info[classifier_name][outputs_kind][
                pattern_kind], ref_patterns)
            self.info[classifier_name]["selection_errors"].append(statistic_class.measures[classifier_name]['E'])

            if classifier_name in context["classifier_list"]:
                #Recovery the original patterns
                context["patterns"].modify_patterns_temporally(classifier_name, pattern_kind, temporal_patterns)
                self.build_real_outputs(context, classifier_name, pattern_kind)
                self.discretize_outputs(context, classifier_name, pattern_kind)
            else:
                self.info[classifier_name][outputs_kind][pattern_kind] = original
                from mullpy.ensembles import Ensemble

                Ensemble(context, classifier_name, self, [pattern_kind])

    #######################################################################

    def instances_error(self, context, classifier_name):
        """
            Measure the error of the classifier giving a list of instances to check.
            """
        statistic_class = Statistic()
        self.info[classifier_name]["selection_errors"] = []
        pattern_kind = context["pattern_kind"]

        if classifier_name in context["classifier_list"]:
            outputs_kind = context["outputs_kind"]
            temporal_patterns = copy.deepcopy(context["patterns"].patterns[classifier_name][pattern_kind])
        else:
            outputs_kind = context["outputs_kind"]
            original = self.info[classifier_name][outputs_kind][pattern_kind]
            original_pattern_ref = context["patterns"].patterns[classifier_name][pattern_kind]

        for counter, filter_list in enumerate(context["filter_list"]):
            #We need to overwrite the context[patterns] variable because build real outputs and discretize use them
            if classifier_name in context["classifier_list"]:
                context["patterns"].modify_patterns_temporally(
                    classifier_name,
                    pattern_kind,
                    context["patterns"].filter_instances(classifier_name, pattern_kind, filter_list))

                self.build_real_outputs(context, classifier_name, pattern_kind)
                self.discretize_outputs(context, classifier_name, pattern_kind)
                ref_patterns = context["patterns"].patterns[classifier_name][pattern_kind]
            else:
                self.info[classifier_name][outputs_kind][pattern_kind] = \
                    [original[i] for i in range(len(original)) if i in filter_list]
                ref_patterns = [original_pattern_ref[i] for i in range(len(original_pattern_ref)) if i in filter_list]

            statistic_class.goodness(context, classifier_name, self.info[classifier_name][outputs_kind][
                pattern_kind], ref_patterns)

            if counter == 0:
                self.info[classifier_name]["selection_errors"].append(
                    [statistic_class.measures[classifier_name][x]['EFP'] for x in context["filter_component"]])
            else:
                self.info[classifier_name]["selection_errors"].append(
                    [statistic_class.measures[classifier_name][x]['EFN'] for x in context["filter_component"]])
                #Recovery the original patterns

            if classifier_name in context["classifier_list"]:
                #Recovery the original patterns
                context["patterns"].modify_patterns_temporally(classifier_name, pattern_kind, temporal_patterns)
                self.build_real_outputs(context, classifier_name, pattern_kind)
                self.discretize_outputs(context, classifier_name, pattern_kind)
            else:
                self.info[classifier_name][outputs_kind][pattern_kind] = original
                from mullpy.ensembles import Ensemble

                Ensemble(context, classifier_name, self, [pattern_kind])
                #######################################################################

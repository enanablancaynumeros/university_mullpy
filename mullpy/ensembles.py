# This Python file uses the following encoding: utf-8
# !/usr/local/bin/python3.4
####################################################
# <Copyright (C) 2012, 2013, 2014, 2015 Yeray Alvarez Romero>
# This file is part of MULLPY.
####################################################
import numpy as np

from mullpy.auxiliar import AutoVivification, nested_dict_access
from mullpy.statistics import Statistics
from mullpy.classifier_info import ClassifiersInfo
from functools import reduce
#######################################################################


class Ensemble:
    """
    Abstract model of Ensemble.
    In the same terms of classifiers_info, aims to build a data structure where all kind of classifiers and kinds of
     ensembles can be constructed.
    First, it needs to initialize the classifiers info abstraction model that was defined in its context parameters.
    """

    def __init__(self, context, ensemble_name, information, pattern_kind_list):
        """
        Complete the Information class with the ensembles decisions
        Self.info as a AutoVivification class might contain only ensemble internal information
        Build real and discretized outputs of the Ensemble, depending of the Ensemble kind.
        """
        self.info = AutoVivification()
        self.weights = None
        self.determine_ensemble_threshold(context, ensemble_name)

        for pattern_kind in pattern_kind_list:
            self._init_decision_matrix(context, ensemble_name, pattern_kind)
            self._build_decision_matrix(context, ensemble_name, information, pattern_kind)
            if nested_dict_access(["classifiers", ensemble_name, "meta_learner"], context):
                self.meta_learner(context, ensemble_name, information)
            else:
                self._schedule_decisions(context, ensemble_name, information, pattern_kind)

    ####################################################

    def _init_decision_matrix(self, context, ensemble_name, pattern_kind):
        """
        :param context:
        :param ensemble_name:
        :param instances_number:
        Construct a list of numpy 2D-arrays where:
            The index is guided by the classes of the ensemble (could be different from the ensemble members)
            x-dimension is the number of instances
            y-dimension is the number of classifiers that classify the present class (could be heterogeneous members)

        The matrix represents the decisions of each classifier for each instance in each class in [0,1]
        """
        classes_array = context["classifiers"][ensemble_name]["classes_names"]

        dim_x = len(context["patterns"].patterns[ensemble_name][pattern_kind])
        if not dim_x:
            msg = 'The ensemble {} is not getting correctly the outputs of the classifiers'
            raise ValueError(msg.format(ensemble_name))

        for class_text in classes_array:
            # TODO: Fix heterogeneous classes base classifiers
            dim_y = len(
                [classifier for classifier in context["classifiers"][ensemble_name]["classifiers"]
                 if class_text in context["classifiers"][classifier]["classes_names"]])
            if not dim_y:
                msg = 'The ensemble {} is not getting correctly the outputs of the classifiers'
                raise ValueError(msg.format(ensemble_name))
            self.info[ensemble_name]["decision_matrix"][pattern_kind][class_text] = \
                np.zeros([dim_x, dim_y], dtype=np.float16)

    ####################################################

    def _build_decision_matrix(self, context, ensemble_name, information, pattern_kind):
        """
        Assign the values of the classifiers to the ensemble matrix decision
        :param context:
        :param ensemble_name:
        :param information:
        """
        outputs_kind = context["classifiers"][ensemble_name]["outputs_kind"]
        classes_array = context["classifiers"][ensemble_name]["classes_names"]

        if context["classifiers"][ensemble_name]["combination_rule"] == "WMV":
            self._obtain_weights(context, ensemble_name, information)

        combination_rule = nested_dict_access(["classifiers", ensemble_name, "combination_rule"], context)
        if combination_rule in ("SMV", "WMV"):
            # Forcing the threshold to 0.5, calculate the mean, then apply the threshold
            context["classifiers"][ensemble_name]["thresholds"]["value"] = \
                [0.5] * len(context["classifiers"][ensemble_name]["classes_names"])

        instances_number = len(context["patterns"].patterns[ensemble_name][pattern_kind])
        for class_pos, class_text in enumerate(classes_array):
            classifier_list = [classifier for classifier in
                               context["classifiers"][ensemble_name]["classifiers"]
                               if class_text in context["classifiers"][classifier]["classes_names"]]
            for i in range(instances_number):
                for j, classifier_name in enumerate(classifier_list):
                    self.info[ensemble_name]["decision_matrix"][pattern_kind][class_text][i, j] = \
                        information.info[classifier_name][outputs_kind][pattern_kind][i][class_pos]

    #######################################################################

    def _schedule_decisions(self, context, ensemble_name, information, pattern_kind):
        """
            Schedule and define the ensembles information structure with continuous and discrete outputs
            :param information:
            :param context:
            :param ensemble_name:
            """
        outputs_number = len(context["classifiers"][ensemble_name]["classes_names"])
        outputs_kind = context["classifiers"][ensemble_name]["outputs_kind"]
        classes_names = context["classifiers"][ensemble_name]["classes_names"]

        instances_number = len(context["patterns"].patterns[ensemble_name][pattern_kind])
        # For optimization reasons in many combinations, generate exactly the arrays we need instead a clear code
        information.info[ensemble_name][outputs_kind][pattern_kind] = \
            np.zeros((instances_number, outputs_number), dtype=np.float16)
        for class_index in range(len(classes_names)):
            for instance in range(instances_number):
                value = self.info[ensemble_name]["decision_matrix"][pattern_kind][
                    classes_names[class_index]][instance]

                combined_output = self.combine_outputs(context, ensemble_name, value, class_index)

                information.info[ensemble_name]["continuous_outputs"][pattern_kind][instance][
                    class_index] = combined_output

        if context["ml_paradigm"] == "classification" and context["execution_kind"] != "NClearning":
            #if outputs_kind == "continuous_outputs":
            #if "learning" not in context["execution_kind"]:
            #if context["classifiers"][ensemble_name]["thresholds"]["determine_threshold"]:
            #    ClassifiersInfo().automatic_threshold_determine(context, ensemble_name)

            # Once the thresholds were calculated we must apply the criterion tie if we are in a classification context
            # Or apply the threshold if we took the continuous outputs from the classifiers
            instances_number = len(context["patterns"].patterns[ensemble_name][pattern_kind])
            for class_text in range(len(classes_names)):
                for instance in range(instances_number):
                    self._criterion_tie_threshold_application(context,
                                                              ensemble_name,
                                                              instance,
                                                              class_text,
                                                              pattern_kind,
                                                              information)

    ####################################################

    def meta_learner(self, context, ensemble_name, information):
        """
        The meta_learner will be used only in a deployment phase. The learning and validation phases will be
        developed as a normal classifier. The pattern kind will be always test.
        The meta learner has to be initialized with all the classifiers at the beginning.
        It is supposed to be learned before taking this point.
        :param context:
        :param ensemble_name:
        :return:
        """
        meta_learner_name = context["classifiers"][ensemble_name]["meta_learner"]
        amount_classifiers = len(context["classifiers"][ensemble_name]["classifiers"].keys())
        len_classes_names = len(context["classifiers"][ensemble_name]["classes_names"])
        classes_names = context["classifiers"][ensemble_name]["classes_names"]
        instances_number = 1
        pattern_kind = "test"

        # Create the patterns structure
        context["patterns"].patterns[meta_learner_name][pattern_kind] = \
            np.ndarray(shape=(instances_number, amount_classifiers * len_classes_names), dtype=np.float32, order='C')

        # Feed the pattern structure with the outputs of the ensemble members
        for class_index in range(len_classes_names):
            for instance in range(instances_number):
                context["patterns"].patterns[meta_learner_name][pattern_kind][instance][class_index] = \
                    self.info[ensemble_name]["decision_matrix"][pattern_kind][classes_names[class_index]][instance]

        # Transform the data if corresponds
        if nested_dict_access(["classifiers", meta_learner_name, "transformer"], context):
            len_inputs = len(context["patterns"].patterns[meta_learner_name][pattern_kind][0]) - len_classes_names
            context["patterns"].patterns[meta_learner_name][pattern_kind][0][:len_inputs] = \
                context["classifiers"][meta_learner_name]["transformer"].transform(
                    context["patterns"].patterns[meta_learner_name][pattern_kind][0][:len_inputs])

        # The output of the ensembles will be take from information.info structure
        information.build_real_outputs(context, meta_learner_name, pattern_kind)
        information.discretize_outputs(context, meta_learner_name, pattern_kind)

    ####################################################

    def _obtain_weights(self, context, ensemble_name, information):

        """
        Obtain weights from the test pattern set
        :param context:
        :param ensemble_name:
        :param information:
        :return:
        """
        statistic_class = Statistics()
        self.weights = np.zeros(len(context["classifiers"][ensemble_name]["classifiers"]))
        outputs_kind = context["classifiers"][ensemble_name]["outputs_kind"]
        for i, classifier_name in enumerate(context["classifiers"][ensemble_name]["classifiers"]):
            if context["outputs_kind"] != "validation":
                # Test information may not exist. Just the context[patter_kind] would be built
                information.build_real_outputs(context, classifier_name, "validation")
                information.discretize_outputs(context, classifier_name, "validation")

            statistic_class.goodness(context, classifier_name,
                                     information.info[classifier_name][outputs_kind]["validation"],
                                     context["patterns"].patterns[classifier_name]["validation"])

            self.weights[i] = statistic_class.measures[classifier_name]["E"]

        return statistic_class

    ####################################################

    def combine_outputs(self, context, ensemble_name, values, class_index):
        """

        :param context:
        :param ensemble_name:
        :param values:
        :param class_index:
        :return:
        """
        # if "meta_learner" in context["classifiers"][ensemble_name] and \
        # context["classifiers"][ensemble_name] is not None:
        #     return self.meta_learner(context, ensemble_name, class_index)

        combination_rule = context["classifiers"][ensemble_name]["combination_rule"]
        combination_rules = {
            "mean": np.mean(values),
            "product": np.divide(np.prod(values), float(len(values))),
            "max": np.max(values),
            "min": np.min(values),
            "median": np.median(values),
            "SVM": np.mean(values),
            "WMV": reduce(lambda w, v: (1. / np.exp(w)) * v, zip(self.weights, values))
        }

        if combination_rule not in combination_rules:
            msg = 'Combination rule not defined. Specify a combination rule for the ensemble from the list: {}'
            raise NameError(msg.format(",".join(combination_rules.keys())))
        else:
            return combination_rules.get(combination_rule)

    #######################################################################

    def _criterion_tie_threshold_application(self, context, ensemble_name, instance, output, pattern_kind, information):
        # Apply the criterion tie when there is a draw in the ensemble decision
        if context["classifiers"][ensemble_name]["criterion_tie"] == "MIN_FN":
            if information.info[ensemble_name]["continuous_outputs"][pattern_kind][instance][output] >= \
                    context["classifiers"][ensemble_name]["thresholds"]["value"][output]:
                information.info[ensemble_name]["discretized_outputs"][pattern_kind][instance][output] = 1.
            else:
                information.info[ensemble_name]["discretized_outputs"][pattern_kind][instance][output] = 0.
        else:
            if information.info[ensemble_name]["continuous_outputs"][pattern_kind][instance][output] > \
                    context["classifiers"][ensemble_name]["thresholds"]["value"][output]:
                information.info[ensemble_name]["discretized_outputs"][pattern_kind][instance][output] = 1.
            else:
                information.info[ensemble_name]["discretized_outputs"][pattern_kind][instance][output] = 0.

    #######################################################################

    def determine_ensemble_threshold(self, context, ensemble_name):
        if context["ml_paradigm"] == "classification" and context["execution_kind"] != "NClearning":
            if "determine_threshold" in context["classifiers"][ensemble_name]["thresholds"] and \
                    context["classifiers"][ensemble_name]["thresholds"]["determine_threshold"]:
                temp_information = ClassifiersInfo()
                pattern_kind = "validation"
                if context["classifiers"][ensemble_name]["outputs_kind"] == "continuous_outputs":
                    if context["outputs_kind"] != "validation":
                        for i, classifier_name in enumerate(context["classifiers"][ensemble_name]["classifiers"]):
                            # Test information may not exist. Just the context[patter_kind] would be built
                            temp_information.build_real_outputs(context, classifier_name, pattern_kind)
                            # TODO: improve this method of stopping the recursivity
                    context["classifiers"][ensemble_name]["thresholds"]["determine_threshold"] = 0
                    Ensemble(context, ensemble_name, temp_information, [pattern_kind])
                    temp_information.automatic_threshold_determine(context, ensemble_name)

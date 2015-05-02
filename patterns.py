# This Python file uses the following encoding: utf-8
# !/usr/local/bin/python3.3
####################################################
# <Copyright (C) 2012, 2013, 2014 Yeray Alvarez Romero>
# This file is part of MULLPY.
####################################################
from auxiliar import AutoVivification, path_exists
import copy
import numpy as np
import os
import re
import itertools
#######################################################################


class Pattern:
    """
    Class to define functions about getting information from files and build lists of patterns of each classifier
    """

    def __init__(self, context):
        self.patterns = AutoVivification()
        for classifier_name in context["classifier_list"]:
            for pattern_kind in context["patterns_texts"]:
                self.patterns[classifier_name][pattern_kind] = None

    ####################################################

    def modify_patterns_temporally(self, classifier_name, pattern_kind, new_pattern_list):
        self.patterns[classifier_name][pattern_kind] = new_pattern_list

    ####################################################

    def check_file_reading(self, context, classifier_name, pattern_kind):
        try:
            f = open(context["classifiers"][classifier_name]["patterns"][pattern_kind], "r")
        except IOError:
            raise NameError("Error ocurred trying to open file %s" % (
                context["classifiers"][classifier_name]["patterns"][pattern_kind]))
        return f

    ####################################################
    @staticmethod
    def create_new_patterns(context, classifier_name, pattern_kind, file_name):
        try:
            f = open(file_name, "w+")
        except IOError:
            raise NameError("Error ocurred trying to open file %s" % file_name)

        for feature_name in context["classifiers"][classifier_name]["features_names"]:
            f.write("@FEATURE {0}\n".format(feature_name))

        len_classes = len(context["classifiers"][classifier_name]["classes_names"])
        for instance in context["patterns"].patterns[classifier_name][pattern_kind]:
            #Write values
            for value in range(len(instance) - len_classes):
                f.write("{0}\t".format(instance[value]))
                #Write class
            for classes in range(len(instance) - len_classes, len(instance)):
                if context["ml_paradigm"] == "classification":
                    f.write("{0:.1f}\t".format(instance[classes]))
                    if classes != len(instance) - 1:
                        f.write("\t")
                else:
                    f.write("{0}\t".format(instance[classes]))
            f.write("\n")
        f.close()

    ####################################################
    @staticmethod
    def features_names(all_elements):
        """
        Features name are treated as capitalized letters
        :param all_elements:
        :return: The line number where the data start and the features names read on all_elements parameter
        """
        temp_feat_list = []
        i = 0
        res = re.search(r'@[F-f][E-e][A-a][T-t][U-u][R-r][E-e]', all_elements[i])
        while res is not None:
            temp_feat_list.append(re.sub("\n", "", re.sub(" ", "", all_elements[i][res.end():])).upper())
            i += 1
            if i > len(all_elements) - 1:
                raise NameError("Patterns error, there is no data included after features")
            res = re.search(r'@[F-f][E-e][A-a][T-t][U-u][R-r][E-e]', all_elements[i])

        return i, temp_feat_list

    ####################################################

    @staticmethod
    def check_file_type(context, classifier_name, pattern_kind):
        """
        Check the file type and define the separator of the features depending on that separator.
        Support for files of kind:
            - CSV, comma separated value
            - .pat file, an own type file similar to arff format.
        """
        #Check file type and define the separator of the patterns if not
        if "patterns_separator" not in context.keys():
            if ".pat" in context["classifiers"][classifier_name]["patterns"][pattern_kind]:
                patterns_separator = " "
            elif ".csv" in context["classifiers"][classifier_name]["patterns"][pattern_kind]:
                patterns_separator = ","
            else:
                raise ValueError("Not accepted file type")
            return patterns_separator
        else:
            return context["patterns_separator"]

    ####################################################
    @staticmethod
    def deployment_reorder_features(classifier_features_name, temp_feat_list, all_elements):
        temp = []
        for feature in classifier_features_name:
            temp.append(all_elements[0][temp_feat_list.index(feature)])
        return [temp]

    ####################################################
    @staticmethod
    def convert_data(patterns_separator, all_elements):
        """
        Convert to float data from strings and separate the elements by patterns_separator parameter
        :param patterns_separator: The character element to separate elements.
        :param all_elements: The matrix which contains all the info from the pattern file.
        """
        for i in range(len(all_elements)):
            all_elements[i] = list(
                map(np.float32,
                    re.sub(r" *\n", "", re.sub(r"  +", " ", re.sub(r"\t", patterns_separator,
                                                                   all_elements[i]))).split(patterns_separator)))

    ####################################################

    def extract(self, context, classifier_name, pattern_kind):
        """
        Extract all patterns kind for a classifier.
        """
        f = self.check_file_reading(context, classifier_name, pattern_kind)
        all_elements = f.readlines()

        patterns_separator = self.check_file_type(context, classifier_name, pattern_kind)

        if "deployment" not in context["execution_kind"] or pattern_kind != "test":
            len_classes = len(context["classifiers"][classifier_name]["classes_names"])
        else:
            len_classes = 0

        #List of features read from the file and the line number where the data start in all_elements
        i, temp_feat_list = self.features_names(all_elements)
        if context["ml_paradigm"] == "regression" and len(context["results"]["to_file"]["prediction_horizon"]) and \
                        pattern_kind == "validation":
            try:
                horizon_0 = context["results"]["to_file"]["prediction_horizon"][0]
                horizon_1 = context["results"]["to_file"]["prediction_horizon"][1]
                all_elements = all_elements[i + horizon_0:i + horizon_0 + horizon_1]
            except:
                raise Exception("Prediction Horizon is higher than instances in validation set")
        else:
            all_elements = all_elements[i:]
            #Convert data from strings to float type. To extend in cases where other types need to be converted
        self.convert_data(patterns_separator, all_elements)

        all_elements = self.manage_features(context, classifier_name, pattern_kind, temp_feat_list, all_elements)

        patterns_temp = np.ndarray(shape=(len(all_elements), len(all_elements[0])), dtype=np.float32)
        for i in range(len(all_elements)):
            temp = all_elements[i][:len(all_elements[i]) - len_classes]
            for j in range(len(temp)):
                patterns_temp[i][j] = temp[j]

            for j in range(len(all_elements[i]) - len_classes, len(all_elements[i])):
                if context["classifiers"][classifier_name]["patterns"]["range"] == [-1, 1]:
                    if all_elements[i][j] == 0:
                        patterns_temp[i][j] = np.float32(-1.0)
                    else:
                        patterns_temp[i][j] = np.float32(all_elements[i][j])
                elif context["classifiers"][classifier_name]["patterns"]["range"] == [0, 1]:
                    if all_elements[i][j] == -1.0:
                        patterns_temp[i][j] = np.float32(0.0)
                    else:
                        patterns_temp[i][j] = np.float32(all_elements[i][j])

        self.patterns[classifier_name][pattern_kind] = patterns_temp
        f.close()

    ####################################################

    def construct_ensemble_patterns_multiple_classifiers(self, context, ensemble_name, min_classifier_list):

        pattern_kind = "test"
        first_member = context["classifiers"][ensemble_name]["classifiers"][0]
        pattern_amount = len(context["patterns"].patterns[first_member][pattern_kind])
        classes_amount = len(context["classifiers"][ensemble_name]["classes_names"])

        self.patterns[ensemble_name]["learning"] = np.zeros(
            [len(context["patterns"].patterns[first_member]["learning"]), 2, classes_amount])
        self.patterns[ensemble_name]["validation"] = np.zeros(
            [len(context["patterns"].patterns[first_member]["validation"]), 2, classes_amount])
        self.patterns[ensemble_name]["test"] = np.zeros([pattern_amount, 2, classes_amount])

        for classifier_name in min_classifier_list:
            for pattern_number in range(pattern_amount):
                for i, class_text in enumerate(context["classifiers"][classifier_name]["classes_names"]):
                    position = context["classifiers"][ensemble_name]["classes_names"].index(class_text)
                    value = context["patterns"].patterns[classifier_name][pattern_kind][pattern_number][1][i]
                    #If the value was already token from other classifier, the value will be overwritten,
                    # but it is not a problem at first
                    self.patterns[ensemble_name]["test"][pattern_number][1][position] = value

    ####################################################

    def ensemble_copy(self, context, ensemble_name, deep=0):
        first_member = context["classifiers"][ensemble_name]["classifiers"][0]
        if deep:
            self.patterns[ensemble_name] = copy.deepcopy(self.patterns[first_member])
        else:
            self.patterns[ensemble_name] = self.patterns[first_member]

    ####################################################

    def filter_characteristics(self, classifier_name, pattern_kind, filter_list):
        """
        Filter instances indicated in filter_list without modifying the original pattern structure
        """
        #TODO: to reconvert to itertools.compress as in apply filter by instances
        instances_number = len(self.patterns[classifier_name][pattern_kind])
        new_pattern = []
        for i in range(instances_number):
            temp = []
            for j in filter_list:
                temp.append(self.patterns[classifier_name][pattern_kind][i][0][j])
            new_pattern.append([temp, self.patterns[classifier_name][pattern_kind][i][1]])

        return np.asarray(new_pattern)

    ####################################################

    def filter_instances(self, classifier_name, pattern_kind, filter_list):
        """
        Filter instances indicated in filter_list without modifying the original pattern structure
        """
        return np.asarray([self.patterns[classifier_name][pattern_kind][i] for i in
                           range(len(self.patterns[classifier_name][pattern_kind])) if i in filter_list])

    ####################################################
    def filter_classes(self, classifier_name, pattern_kind, filter_list):
        """
        Filter instances indicated in filter_list without modifying the original pattern structure
        """
        return np.asarray([instance for instance in self.patterns[classifier_name][pattern_kind]
                           if list(instance[1]) in filter_list])

    ####################################################

    def transform_multilabel_to_N_classes(self, context, classifier_name):
        """
        Transform the multilabel files into a n-classes problem.
        Convert a multilabel pattern file into multiple one-class pattern files
        """
        for pattern_kind in context["patterns_texts"]:
            for class_text in context["classifiers"][classifier_name]["classes_names"]:
                dir_name = context["general_path"] + "patterns/" + context["preprocess"][
                    "transform_multilabel_to_N_classes"]["new_set_name"] + '/'
                path_exists(dir_name)
                file_name = os.path.basename(context["classifiers"][classifier_name]["patterns"][pattern_kind])
                file_name = dir_name + file_name[:file_name.find(".pat")] + "_" + class_text + ".pat"
                try:
                    f = open(file_name, "w+")
                except IOError:
                    raise NameError("Error ocurred trying to open file %s" % file_name)

                for i in range(len(context["patterns"].patterns[classifier_name][pattern_kind])):
                    total = context["patterns"].patterns[classifier_name][pattern_kind][i]
                    #Write values
                    for value in total[0][:len(total[0])]:
                        f.write(str(value) + " ")
                        #Write class
                    classes = total[1]
                    if classes[context["classifiers"][classifier_name]["classes_names"].index(class_text)] == 1:
                        f.write("1\n")
                    else:
                        f.write("0\n")
                f.close()

    ####################################################

    def transform_N_classes_to_multilabel(self):
        pass

    ####################################################

    @staticmethod
    def build_features_filter(len_classes, selected_features, temp_feat_list):
        """
        Return a binary array that indicates which given features are found in the pattern file
        :param selected_features:
        :param temp_feat_list:
        :return:
        """
        feat_filter = []
        for feature in temp_feat_list:
            if feature not in selected_features:
                feat_filter.append(0)
            else:
                feat_filter.append(1)
        for classes in range(len_classes):
            feat_filter.append(1)
        return feat_filter

    ####################################################

    def apply_filter_by_instances(self, context, classifier_name, all_elements, temp_feat_list):
        """
        Feat_filter indicates the features that must be kept from the pattern file, depending on the features_names
        array.
        :param context:
        :param classifier_name:
        :param all_elements:
        :param temp_feat_list:
        :return:
        """
        feat_filter = self.build_features_filter(len(context["classifiers"][classifier_name]["classes_names"]),
                                                 context["classifiers"][classifier_name]["features_names"],
                                                 temp_feat_list)

        all_elements = np.asarray([list(itertools.compress(x, feat_filter)) for x in all_elements])
        temp_feat_list = list(itertools.compress(temp_feat_list, feat_filter))
        #all_elements contains now the same features than features_name from the classifier context, but unordered

        features_names = context["classifiers"][classifier_name]["features_names"]
        while temp_feat_list != features_names:
            for i, feature_true, feature_all_elements in zip(range(len(features_names)), features_names, temp_feat_list):
                if feature_true != feature_all_elements:
                    temp = copy.deepcopy(all_elements[:, i])
                    all_elements[:, i] = all_elements[:, temp_feat_list.index(feature_true)]
                    all_elements[:, temp_feat_list.index(feature_true)] = temp
                    temp_feature = copy.deepcopy(temp_feat_list[i])
                    old_position = temp_feat_list.index(feature_true)
                    temp_feat_list[i] = feature_true
                    temp_feat_list[old_position] = temp_feature
                    break

        return all_elements

    ####################################################

    def manage_features(self, context, classifier_name, pattern_kind, temp_feat_list, all_elements):
    #In deployment execution we are not sure that the order of the features given in the validation file
        # match with the order of the features in which the algorithm learnt
        if "features_names" in context["classifiers"][classifier_name].keys() and \
                        context["classifiers"][classifier_name]["features_names"] is not None:
            context["classifiers"][classifier_name]["features_names"] = \
                [x.upper() for x in context["classifiers"][classifier_name]["features_names"]]

        if "deployment" in context["execution_kind"] and pattern_kind == "test" and \
                        "features_names" in context["classifiers"][classifier_name].keys():
            all_elements = self.deployment_reorder_features(context["classifiers"][classifier_name]["features_names"],
                                                            temp_feat_list,
                                                            all_elements)

        if "features_names" in context["classifiers"][classifier_name].keys() and \
                        context["classifiers"][classifier_name]["features_names"] is not None and \
                        context["classifiers"][classifier_name]["features_names"] != temp_feat_list:

            all_elements = self.apply_filter_by_instances(context,
                                                          classifier_name,
                                                          all_elements,
                                                          temp_feat_list)

        if ("features_names" not in context["classifiers"][classifier_name].keys() or
                    "features_names" in context["classifiers"][classifier_name].keys() and
                        context["classifiers"][classifier_name]["features_names"] is None) \
                and len(temp_feat_list):
            context["classifiers"][classifier_name]["features_names"] = temp_feat_list

        return all_elements
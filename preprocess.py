# This Python file uses the following encoding: utf-8
# !/usr/local/bin/python3.3
####################################################
# <Copyright (C) 2012, 2013, 2014 Yeray Alvarez Romero>
# This file is part of MULLPY.
####################################################
import numpy as np
from patterns import Pattern
from auxiliar import AutoVivification, path_exists


class PreProcess():
    """
    Scheduler of the PreProcess execution.
    """

    def schedule(self, context):
        #TODO: connect to other libraries with a strong preprocessing library
        for preprocess_function in context["preprocess"].keys():
            if context["preprocess"][preprocess_function]["activate"]:
                getattr(self, preprocess_function)(context)

                #########################################################################

    @staticmethod
    def transform_multilabel_to_n_classes(context):
        for classifier_name in context["classifiers"].keys():
            Pattern(context).transform_multilabel_to_N_classes(context, classifier_name)

            #########################################################################

    @staticmethod
    def bagging(context, filters, lengths, total_length):
        learning_length = lengths["learning"]
        for i in range(context["preprocess"]["random_distribution"]["number_base_classifiers"]):
            temp = []
            while len(set(temp)) != learning_length:
                temp.append(np.random.randint(0, total_length))
            filters["learning"].append(temp)

            filters["validation"].append([x for x in range(total_length) if x not in set(filters["learning"][i])])

            #########################################################################

    @staticmethod
    def pasting_rvotes(context, filters, lengths, total_length):
        learning_length = lengths["learning"]
        for i in range(context["preprocess"]["random_distribution"]["number_base_classifiers"]):
            temp = []
            while len(temp) != learning_length:
                value = np.random.randint(0, total_length)
                if value not in temp:
                    temp.append(value)

            filters["learning"].append(temp)
            filters["validation"].append([x for x in range(total_length) if x not in temp])

    #########################################################################
    @staticmethod
    def all_features_combination(context, filters, characteristics_length):
        import itertools

        min_characteristics = context["preprocess"]["random_distribution"]["all_features_combination"][
            "min_characteristics"]
        max_characteristics = context["preprocess"]["random_distribution"]["all_features_combination"][
            "max_characteristics"]
        for characteristics_amount in range(min_characteristics, max_characteristics + 1):
            temporal = list(itertools.combinations(range(characteristics_length), characteristics_amount))
            for t in temporal:
                filters["learning"].append(list(t))

        # filters["test"] = filters["learning"]
        filters["validation"] = filters["learning"]

    #########################################################################
    @staticmethod
    def random_subspaces(context, filters, characteristics_length):
        for i in range(context["preprocess"]["random_distribution"]["number_base_classifiers"]):
            temp = []
            characteristics_amount = np.random.randint(
                context["preprocess"]["random_distribution"]["random_subspaces"]["min_characteristics"],
                characteristics_length)
            while len(temp) != characteristics_amount:
                temp.append(np.random.randint(0, characteristics_length))
            filters["learning"].append(temp)

            # filters["test"] = filters["learning"]
            filters["validation"] = filters["learning"]

    #########################################################################
    @staticmethod
    def classes_indexes(context, data_set):

        classes_indexes = AutoVivification()
        classes_texts = context["classifiers"][context["classifier_list"][0]]["classes_names"]
        len_inputs = len(data_set[0]) - len(classes_texts)
        for class_text in classes_texts:
            column = [data_set[i][len_inputs + classes_texts.index(class_text)] for i in range(len(data_set))]
            classes_indexes[class_text] = column

        return classes_indexes

    #########################################################################
    @staticmethod
    def classes_counter_indexes(context, data_set):

        classes_counter = AutoVivification()
        classes_indexes = AutoVivification()
        classes_texts = context["classifiers"][context["classifier_list"][0]]["classes_names"]
        len_inputs = len(data_set[0]) - len(classes_texts)

        for class_text in classes_texts:
            column = [data_set[i][len_inputs + classes_texts.index(class_text)] for i in range(len(data_set))]
            classes_counter[class_text] = np.sum(column)
            classes_indexes[class_text] = column

        return classes_counter, classes_indexes

    #########################################################################
    @staticmethod
    def forecasting_distribution(context, filters):
        data_set = context["patterns"].patterns[context["classifier_list"][0]]["learning"]
        validation_size = context["preprocess"]["random_distribution"]["forecasting_distribution"]["validation_size"]
        activate = context["preprocess"]["random_distribution"]["forecasting_distribution"]["walking_forward"]
        folds = context["preprocess"]["random_distribution"]["number_base_classifiers"]

        filters["learning"] = []
        filters["validation"] = []
        if activate is not 0:
            for fold in range(folds):
                filters["learning"].append([i for i in range(fold * validation_size,
                                                             len(data_set) - validation_size * (folds - fold))])

                filters["validation"].append([i for i in range(len(data_set) - validation_size * (folds - fold),
                                                               len(data_set) - validation_size * (folds - fold) +
                                                               validation_size)])
        else:
            filters["learning"].append([i for i in range(0, len(data_set) - validation_size)])
            filters["validation"].append([i for i in range(len(data_set) - validation_size, len(data_set))])

        return filters

    #########################################################################

    def binarize_data(self, context):
        from sklearn.preprocessing import LabelBinarizer

        pattern_kind = "learning"
        lb = LabelBinarizer()

        inputs_len = len(context["patterns"].patterns[context["classifier_list"][0]][pattern_kind][0]) - 1
        inputs = [x[:, range(inputs_len)] for x in
                  context["patterns"].patterns[context["classifier_list"][0]]["learning"]]

        outputs = context["patterns"].patterns[context["classifier_list"][0]][pattern_kind][:, -1]
        multilabel_outputs = [(x,) for x in outputs]
        lb.fit(multilabel_outputs)
        new_outputs = lb.transform(multilabel_outputs)

        context["patterns"].patterns[context["classifier_list"][0]][pattern_kind] = []
        for i, input in enumerate(inputs):
            temp = [x for x in inputs[i]]
            temp.extend(new_outputs[i])
            context["patterns"].patterns[context["classifier_list"][0]]["learning"].append(temp)

        dir_name = context["general_path"] + "patterns/" + context["classifiers"][context["classifier_list"][0]]["set"]
        file_name = dir_name + "/" + pattern_kind + "_binarized" + ".pat"
        context["patterns"].create_new_patterns(context, context["classifier_list"][0], pattern_kind, file_name)

    #########################################################################

    def k_fold(self, context, filters):
        classes_texts = context["classifiers"][context["classifier_list"][0]]["classes_names"]
        num_instances = sum([len(context["patterns"].patterns[context["classifier_list"][0]][x])
                             for x in context["patterns"].patterns[context["classifier_list"][0]]])

        data_set = None
        for i, filter_name in enumerate(context["patterns"].patterns[context["classifier_list"][0]].keys()):
            if i == 0:
                data_set = context["patterns"].patterns[context["classifier_list"][0]][filter_name]
            else:
                data_set = np.concatenate(data_set,
                                          context["patterns"].patterns[context["classifier_list"][0]][filter_name])

        total_classes_counter, classes_indexes = self.classes_counter_indexes(context, data_set)
        classes_counter = AutoVivification()
        min_limit_classes = np.min([total_classes_counter[class_counter] for class_counter in total_classes_counter])

        for i in range(context["preprocess"]["random_distribution"]["number_base_classifiers"]):
            total_indexes = []
            for j, filter_name in enumerate(["learning", "validation"]):
                aux_list = []
                aux_percent = context["preprocess"]["random_distribution"]["k_fold"]["percents"][filter_name]
                if j == len(context["preprocess"]["random_distribution"]["k_fold"]["percents"]) - 1:
                    filters[filter_name].append([x for x in range(len(data_set)) if x not in total_indexes])
                    break
                else:
                    if context["preprocess"]["random_distribution"]["k_fold"]["balanced"]:
                        total_instances = 0
                        for class_text in context["classifiers"][context["classifier_list"][0]]["classes_names"]:
                            classes_counter[filter_name][class_text] = np.ceil(aux_percent * min_limit_classes)
                            total_instances += classes_counter[filter_name][class_text]
                    else:
                        total_instances = np.ceil(aux_percent * num_instances)

                len_inputs = len(data_set[0]) - len(classes_texts)
                while len(aux_list) != total_instances:
                    value = np.random.randint(0, len(data_set))
                    if value not in total_indexes:
                        if context["preprocess"]["random_distribution"]["k_fold"]["balanced"]:
                            if classes_counter[filter_name][
                                classes_texts[list(data_set[value][len_inputs:]).index(1)]] > 0:
                                total_indexes.append(value)
                                aux_list.append(value)
                                classes_counter[filter_name][
                                    classes_texts[list(data_set[value][len_inputs:]).index(1)]] -= 1
                        else:
                            total_indexes.append(value)
                            aux_list.append(value)

                filters[filter_name].append(aux_list)

    #########################################################################

    @staticmethod
    def check_features_amount(context):
        classes_texts = context["classifiers"][context["classifier_list"][0]]["classes_names"]
        data_set = context["patterns"].patterns[context["classifier_list"][0]]["learning"]
        features_amount = len(data_set[0]) - len(classes_texts)

        for classifier_name in context["classifier_list"]:
            if features_amount != (len(context["patterns"].patterns[classifier_name]["learning"][0]) -
                                       len(classes_texts)):
                raise ValueError("Different lengths in learning patterns of classifier %s and %s" % (
                    context["classifier_list"][0], classifier_name))
        return features_amount

    #########################################################################

    def random_distribution(self, context):
        """
        Bagging methods come in many flavours but mostly differ from each other by the way they draw random subsets
         of the training set:

        -When random subsets of the dataset are drawn as random subsets of the samples, then this algorithm is known
        as Pasting Rvotes.
        -When samples are drawn with replacement, then the method is known as Bagging.
        -When random subsets of the dataset are drawn as random subsets of the features, then the method is known as
        Random Subspaces.
        -When base estimators are built on subsets of both samples and features, then the method is known as Random
        Patches.

        group_successive variable groups each X instances. Each of these successive instances has to be together in
        the sampling process
        """
        total_length = 0
        lengths = AutoVivification()
        for pattern_kind in context["patterns"].patterns[context["classifier_list"][0]]:
            lengths[pattern_kind] = len(context["patterns"].patterns[context["classifier_list"][0]][pattern_kind])
            total_length += lengths[pattern_kind]

        #Check if the length of patterns have the same size
        for classifier_name in context["classifier_list"]:
            for pattern_kind in context["patterns"].patterns[classifier_name]:
                if len(context["patterns"].patterns[classifier_name][pattern_kind]) != lengths[pattern_kind]:
                    raise ValueError(
                        'The length of the %s pattern of classifier %s has different size from others' % pattern_kind,
                        classifier_name)

        if context["preprocess"]["random_distribution"]["group_successive"]:
            total_length = int(total_length / context["preprocess"]["random_distribution"]["group_successive"])
            for pattern_kind in lengths:
                lengths[pattern_kind] = int(
                    lengths[pattern_kind] / context["preprocess"]["random_distribution"]["group_successive"])

        dir_name = context["general_path"] + "patterns/" + context["classifiers"][context["classifier_list"][0]]["set"]
        filters = AutoVivification()
        ###Specific kind of sampling###
        #############
        ######BAGGING
        #############
        if "bagging" in context["preprocess"]["random_distribution"] and \
                context["preprocess"]["random_distribution"]["bagging"]["activate"]:
            for pattern_kind in context["patterns_texts"]:
                filters[pattern_kind] = []
            self.bagging(context, filters, lengths, total_length)
            dir_name += "_bagging/"
        #############
        ######PASTING
        #############
        elif "pasting_Rvotes" in context["preprocess"]["random_distribution"] and \
                context["preprocess"]["random_distribution"]["pasting_Rvotes"]["activate"]:
            for pattern_kind in context["patterns_texts"]:
                filters[pattern_kind] = []
            self.pasting_rvotes(context, filters, lengths, total_length)
            dir_name += "_pasting_Rvotes/"
        #################
        #RANDOM SUBSPACES
        #################
        elif "random_subspaces" in context["preprocess"]["random_distribution"] and \
                context["preprocess"]["random_distribution"]["random_subspaces"]["activate"]:
            features_amount = self.check_features_amount(context)
            for pattern_kind in context["patterns_texts"]:
                filters[pattern_kind] = []
            self.random_subspaces(context, filters, features_amount)
            dir_name += "_random_subspaces/"
        #############
        #COMBINATIONS
        #############
        elif "all_features_combination" in context["preprocess"]["random_distribution"] and \
                context["preprocess"]["random_distribution"]["all_features_combination"]["activate"]:
            features_amount = self.check_features_amount(context)
            for pattern_kind in context["patterns_texts"]:
                filters[pattern_kind] = []
            self.all_features_combination(context, filters, features_amount)
            dir_name += "_features_combination/"
            context["preprocess"]["random_distribution"]["number_base_classifiers"] = len(filters["learning"])
        ###############
        #RANDOM PATCHES
        ###############
        elif "random_patches" in context["preprocess"]["random_distribution"] and \
                context["preprocess"]["random_distribution"]["random_patches"]["activate"]:
            dir_name += "_random_patches/"
        ###############
        #K-FOLD
        ###############
        elif "k_fold" in context["preprocess"]["random_distribution"] and \
                context["preprocess"]["random_distribution"]["k_fold"]["activate"]:
            for pattern_kind in context["preprocess"]["random_distribution"]["k_fold"]["percents"]:
                filters[pattern_kind] = []
            self.k_fold(context, filters)
            dir_name += "_k_fold/"
        ###############
        #Forecasting distribution
        ###############
        elif "forecasting_distribution" in context["preprocess"]["random_distribution"] and \
                context["preprocess"]["random_distribution"]["forecasting_distribution"]["activate"]:
            self.forecasting_distribution(context, filters)
            dir_name += "_walking_forward/"

            ###Common functions###
        elif "bagging" in context["preprocess"]["random_distribution"] and \
                context["preprocess"]["random_distribution"]["bagging"]["activate"] \
                or "pasting_Rvotes" in context["preprocess"]["random_distribution"] and \
                        context["preprocess"]["random_distribution"]["pasting_Rvotes"]["activate"]:
            if context["preprocess"]["random_distribution"]["group_successive"]:
                for kind_of in filters:
                    for filter in filters[kind_of]:
                        for i in range(len(filter)):
                            filter[i] = (
                                filter[i] * context["preprocess"]["random_distribution"]["group_successive"])
                            for j in range(1, context["preprocess"]["random_distribution"]["group_successive"]):
                                filter.append(filter[i] + j)

        path_exists(dir_name)

        self._generate_new_patterns_random_distribution(context, filters, dir_name)

    #########################################################################

    @staticmethod
    def _generate_new_patterns_random_distribution(context, filters, dir_name):
        for classifier_name in context["classifiers"].keys():
            all_patterns = [context["patterns"].patterns[classifier_name][pattern_kind][i]
                            for pattern_kind in context["patterns"].patterns[classifier_name].keys()
                            for i in range(len(context["patterns"].patterns[classifier_name][pattern_kind]))]

            for pattern_kind in filters:
                for number in range(context["preprocess"]["random_distribution"]["number_base_classifiers"]):
                    file_name = dir_name + "/" + pattern_kind + "_" + str(number) + ".pat"

                    if "random_subspaces" in context["preprocess"]["random_distribution"] and \
                            context["preprocess"]["random_distribution"]["random_subspaces"]["activate"] or \
                                            "all_features_combination" in context["preprocess"][
                                        "random_distribution"] and \
                                    context["preprocess"]["random_distribution"]["all_features_combination"][
                                        "activate"]:
                        temporal_pattern = context["patterns"].patterns[classifier_name][pattern_kind]
                        new_pattern = context["patterns"]. \
                            filter_characteristics(classifier_name, pattern_kind, filters[pattern_kind][number])
                        context["patterns"].modify_patterns_temporally(classifier_name, pattern_kind, new_pattern)
                        context["patterns"].create_new_patterns(context, classifier_name, pattern_kind, file_name)
                        context["patterns"].modify_patterns_temporally(classifier_name, pattern_kind,
                                                                       temporal_pattern)

                    else:
                        new_pattern = np.asarray([all_patterns[i] for i in filters[pattern_kind][number]])
                        context["patterns"].modify_patterns_temporally(classifier_name, pattern_kind, new_pattern)
                        context["patterns"].create_new_patterns(context, classifier_name, pattern_kind, file_name)

    #########################################################################

    @staticmethod
    def create_data_transformer(classifier_name, context, list_divided):
        from auxiliar import check_equal_classifier_patterns

        for pattern_kind in context["patterns_texts"]:
            for classifier_name_2 in list_divided:
                if check_equal_classifier_patterns(context, classifier_name, classifier_name_2, pattern_kind):
                    context["classifiers"][classifier_name]["transformer"] = \
                        context["classifiers"][classifier_name_2]["transformer"]
                    return

        from sklearn import preprocessing

        if "learning" not in context["patterns_texts"]:
            raise ValueError("Learning set is not defined in patterns_texts")
        learning_set = context["patterns"].patterns[classifier_name]["learning"]
        classes_texts = context["classifiers"][classifier_name]["classes_names"]
        len_inputs = len(learning_set[0]) - len(classes_texts)
        # classes_texts = context["classifiers"][classifier_name]["classes_names"]
        # if "deployment" in context["execution_kind"]:
        #     len_inputs = len(learning_set[0])
        # else:
        #     len_inputs = len(learning_set[0]) - len(classes_texts)
        #
        # #Check regression or classification type, to get all the features with class included or not
        # if context["ml_paradigm"] == "regression":
        #     inputs_learning = learning_set
        # elif context["ml_paradigm"] == "classification":
        #     inputs_learning = learning_set[:, range(len_inputs)]
        # else:
        #     raise Exception("bad definition of variable ml_paradigm")

        if "args" in context["classifiers"][classifier_name]["data_transformation"]:
            args = context["classifiers"][classifier_name]["data_transformation"]["args"]
        else:
            args = {}

        context["classifiers"][classifier_name]["transformer"] = \
            getattr(preprocessing, context["classifiers"][classifier_name]["data_transformation"]["kind"])(
                **args).fit(learning_set[:, range(len_inputs)])

    #########################################################################
    @staticmethod
    def apply_data_transformation(classifier_name, context, pattern_kind):
        """
        Performs the data transformation of a classifier and copy it from another classifier if exist and corresponds.
        :param classifier_name:
        :param context:
        :param list_divided:
        :return:
        """
        if "deployment" in context["execution_kind"]:
            if context["ml_paradigm"] == "classification":
                len_inputs = len(context["patterns"].patterns[classifier_name][pattern_kind][0])
        else:
            if context["ml_paradigm"] == "classification":
                len_classes = len(context["classifiers"][classifier_name]["classes_names"])
                len_inputs = len(context["patterns"].patterns[classifier_name]["learning"][0]) - len_classes

        for i, instance in enumerate(context["patterns"].patterns[classifier_name][pattern_kind]):
            if context["ml_paradigm"] == "regression":
                context["patterns"].patterns[classifier_name][pattern_kind] = \
                    context["classifiers"][classifier_name]["transformer"].transform(instance)
            elif context["ml_paradigm"] == "classification":
                instance[:len_inputs] = \
                    context["classifiers"][classifier_name]["transformer"].transform(instance[:len_inputs])
            else:
                raise NameError("ml_paradigm not valid")

    #########################################################################

    def create_data_transformation(self, classifier_name, list_divided, out_q, context):
        self.create_data_transformer(classifier_name[0], context, list_divided)

        if out_q is not None:
            out_q.put([context["patterns"].patterns, context["classifiers"]])
            out_q.close()

    #########################################################################
    @staticmethod
    def points2series(context):
        import pandas as pd
        from auxiliar import csv2pat
        import sys
        import os

        serie_points_amount = context["preprocess"]["points2series"]["serie_size"]
        input_file = context["preprocess"]["points2series"]["input_file"]
        output_file = context["preprocess"]["points2series"]["output_file"]
        class_variable = context["preprocess"]["points2series"]["class_variable"]
        series_limit = context["preprocess"]["points2series"]["series_limit"]
        # TODO: Add support for multiple class variables. Now classes_len = 1
        classes_len = 1
        defined_features_list = context["preprocess"]["points2series"]["columns"]

        if defined_features_list == "all":
            input_df = pd.read_csv(input_file)
            defined_features_list = input_df.columns
        else:
            defined_features_list.append(class_variable)
            input_df = pd.read_csv(input_file, usecols=defined_features_list)

        # We have to take only the (series_limit + series_size) last points of input_df
        input_df_last = input_df.iloc[len(input_df) - (series_limit + serie_points_amount):].reset_index(drop=True)

        # Building output columns list defined_features_list
        features_list = []
        for i in range(serie_points_amount):
            for j in range(len(defined_features_list)):
                features_list.append("%s_%d" % (defined_features_list[j].upper(), i))
                # Adding last column, that is class variable.
        if "deployment" not in context["execution_kind"]:
            features_list.append("%s_%s" % (class_variable.upper(), "CLASS"))

        output_df = pd.DataFrame(columns=features_list, dtype=np.float32)
        if "deployment" not in context["execution_kind"]:
            iteration = range(len(input_df_last) - serie_points_amount)
        else:
            iteration = range(1, len(input_df_last) - serie_points_amount + 1)
        for i in iteration:
            # Percentage completed
            if "deployment" not in context["execution_kind"]:
                sys.stdout.write("\r{0}".format("Loaded:%f%%" % (i * 100 / (len(input_df_last) - serie_points_amount))))
                sys.stdout.flush()
            #Iterate over a numpy row in order to optimize the performance
            row = np.zeros((1, len(features_list)), dtype=np.float32)
            j, z = 0, 0
            for j in range(serie_points_amount):
                for column in defined_features_list:
                    # We have to test if the exchange value was correctly given (between 1 and 2 in those dates)
                    row[0, z] = input_df_last.iloc[i + j][column]
                    z += 1
            if "deployment" not in context["execution_kind"]:
                row[0, z] = PreProcess.check_eurusd_values(input_df_last[class_variable][i + serie_points_amount])
            output_df.loc[i] = row
            #Check the variable series_limit and break the for if the amount of rows was reached
            if series_limit is not None and i + 1 >= series_limit:
                break

        #Create the dataFrame to output the csv
        # output_df = pd.DataFrame(matrix, columns=features_list)
        # Building csv and pat files
        file_name = output_file + ".csv"
        path_exists(os.path.dirname(file_name))
        output_df.to_csv(file_name, index=False)
        if context["preprocess"]["points2series"]["to_pat"]:
            csv2pat(file_name, classes_len)
        if not context["preprocess"]["points2series"]["to_csv"]:
            os.remove(file_name)
        # Displaying info
        serie_name = output_file[output_file.rfind("/") + 1:]
        serie_path = output_file[:output_file.rfind("/")]
        if "deployment" not in context["execution_kind"]:
            print("\n%s pattern files built at %s" % (serie_name, serie_path))

    #########################################################################

    @staticmethod
    def check_eurusd_values(value):
        # We have to test if the exchange value was correctly given (between 1 and 2 in those dates)
        return value
        if value > 1000:
            return value / 1000.
        else:
            return value


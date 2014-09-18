# This Python file uses the following encoding: utf-8
#!/usr/local/bin/python3.3
####################################################
#<Copyright (C) 2012, 2013, 2014 Yeray Alvarez Romero>
#This file is part of MULLPY.
####################################################
import os
import multiprocessing
import re
import sys
import numpy as np
import itertools
import collections
import copy
import warnings

from auxiliar import AutoVivification, path_exists
import classifiers
from classifier_info import ClassifiersInfo
from patterns import Pattern
from classifiers import Classifier
from ensembles import Ensemble
#######################################################################


class Process:
    """
    Scheduler of the execution Process.
    Initialize all the parameters related to the process, like:
        -Internal structure of context parameters
        -Construct the directory paths for each kind of files related to the classifiers
        -Extract the patterns
        -Initialize the classifiers classes
    -Include some methods to control the execution flow
    """

    def __init__(self, parameters):
        """
        Default initialization
        """
        self.context = copy.deepcopy(parameters)
        self.execution_list = ["preprocess", "reconfiguring", "learning", "results", "deployment_classification",
                               "deployment_regression"]
        self.manage_execution_kind_error()
        #If a classifier has classifiers means that is an ensemble

        self.context["classifier_list"] = [x for x in self.context["classifiers"].keys() if
                                           "classifiers" not in self.context["classifiers"][x].keys()]
        self.ensemble_generation()
        self.context["ensemble_list"] = [x for x in self.context["classifiers"].keys() if
                                         "classifiers" in self.context["classifiers"][x].keys()]
        self.order_classifier_list()
        #TODO: create a standard for the classifiers names
        #TODO: order the names of classifiers by default taking into account learning names and others criteria
        #TODO: Load classifiers when negative correlation learning is in execution. Means adding new members to the ensemble.
        self.ensemble_learning_process_names = ["NClearning", "RNClearning"]
        #Pattern structure initialization
        self.context["patterns"] = Pattern(self.context)
        self.pattern_range_check()
        # self.init_parallelized_classifiers(self.context["classifier_list"], None)
        classifier_name_list = self.statically_distribute_list(self.context["classifier_list"])
        self.generic_parallel_function(self.init_parallelized_classifiers, classifier_name_list,
                                       objectives=[self.context["classifiers"], self.context["patterns"].patterns],
                                       recolect=1,
                                       by_elements=1)
        ensemble_list = self.statically_distribute_list(self.context["ensemble_list"])
        self.generic_parallel_function(self.init_parallelized_ensembles, ensemble_list,
                                       objectives=[self.context["classifiers"], self.context["patterns"].patterns],
                                       recolect=1,
                                       by_elements=1)
    ####################################################

    def recursive_check_info_structure_and_append(self, objective, results):
        if isinstance(results, Pattern):
            for key in results.patterns:
                for patterns in results.patterns[key]:
                    if results.patterns[key][patterns] is not None:
                        if objective.patterns[key][patterns] is None:
                            objective.patterns[key][patterns] = results.patterns[key][patterns]
        else:
            for k, v in results.items():
                if isinstance(v, list):
                    if k in objective and type(objective[k]) == list:
                        if k == "features_names":
                            objective[k] = v
                        elif objective[k] != v:
                            objective[k] = (objective[k] + v)
                    else:
                        objective[k] = v
                elif isinstance(v, collections.Mapping):
                    r = self.recursive_check_info_structure_and_append(objective.get(k, {}), v)
                    if type(objective) is not type(r):
                        objective[k] = AutoVivification(r)
                    else:
                        objective[k] = r
                elif isinstance(v, Classifier):
                    objective[k] = v
                else:
                    objective[k] = results[k]
        return objective

    ####################################################

    def generic_parallel_function(self, function, working_list, **kwargs):
        """
        To parallel in a generic form a function.
        Args:
            Inputs:
                -The name of the function as a string
                -The list of classifiers to work with
                -keyword arguments:
                    -Arguments to that function
                    -Objective where to recolect the results
                    -If Recolect=1 then results are recolected.

        Objectives must be defined in the same order that the child process write the objects list in the out_q
        """
        #TODO: dynamically distribute the classifiers in list. Process may finish before others
        procs = AutoVivification()
        out_q = AutoVivification()

        if len(working_list):
            for list_divided, pos in zip(working_list, range(len(working_list))):
                args_list = [list_divided]
                out_q[pos] = multiprocessing.Queue()
                args_list.append(out_q[pos])
                if "args" in kwargs:
                    for arg in kwargs["args"]:
                        args_list.append(arg)

                args = tuple(args_list)

                p = multiprocessing.Process(
                    target=function,
                    args=args).start()
                procs[pos] = p

            for pos in procs:
                if kwargs["recolect"]:
                    if "by_elements" in kwargs:
                        #Recollect all classifier results
                        for classifier in working_list[pos]:
                            for result, objective in zip(self.manage_get_queue(out_q, pos), kwargs["objectives"]):
                                if None not in result:
                                    self.recursive_check_info_structure_and_append(objective[classifier], result)
                                else:
                                    if classifier in self.context["classifier_list"]:
                                        del (self.context["classifier_list"][
                                                 self.context["classifier_list"].index(classifier)])
                                    elif classifier in self.context["ensemble_list"]:
                                        del (self.context["ensemble_list"][
                                                 self.context["ensemble_list"].index(classifier)])

                    else:
                        for result, objective in zip(self.manage_get_queue(out_q, pos), kwargs["objectives"]):
                            self.recursive_check_info_structure_and_append(objective, result)
                else:
                    self.manage_get_queue(out_q, pos)

                out_q[pos].close()
                out_q[pos].join_thread()

    #########################################################################
    def manage_get_queue(self, out_q, pos):
        if "deployment" in self.context["execution_kind"]:
            try:
                temp = out_q[pos].get(timeout=5)
            except:
                for pos in out_q:
                    out_q[pos].cancel_join_thread()
                    out_q[pos].close()
                    out_q[pos].join_thread()
                sys.exit(0)
        else:
            temp = out_q[pos].get()
        return temp

    #########################################################################
    @staticmethod
    def manage_write_queue_child(out_q, list_to_write):
        if isinstance(out_q, type(multiprocessing.Queue())):
            out_q.put(list_to_write)
        elif isinstance(out_q, str):
            pass
        else:
            import warnings

            warnings.warn("Exception not controlled in function manage_write_queue_child of mullpy.py")

    #########################################################################

    def init_parallelized_ensembles(self, list_divided, out_q):
        for i, classifier_name in enumerate(list_divided):
            if self.context["interactive"]["activate"]:
                sys.stdout.write("\r{0}>".format("Loaded:%f%%" % ((float(i) / len(list_divided)) * 100)))
                sys.stdout.flush()
            if "learning" not in self.context["execution_kind"]:
                for classifier in self.context["classifiers"][classifier_name]["classifiers"]:
                    if classifier not in self.context["classifier_list"]:
                        raise ValueError("Ensemble %s needs the classifier %s and it was deleted from list" % (
                            classifier_name, classifier))

            self.patterns_files_extract(classifier_name, list_divided)
            if out_q is not None:
                list_to_write = [self.context["classifiers"][classifier_name],
                                 self.context["patterns"].patterns[classifier_name]]
                self.manage_write_queue_child(out_q, list_to_write)

    #########################################################################

    def init_parallelized_classifiers(self, list_divided, out_q):
        for i, classifier_name in enumerate(list_divided):
            if self.context["interactive"]["activate"]:
                sys.stdout.write("\r{0}>".format("Loaded:%f%%" % ((float(i) / len(list_divided)) * 100)))
                sys.stdout.flush()
            self.define_file_paths(classifier_name)
            self.check_config_files(classifier_name)
            if classifier_name not in self.context["classifier_list"]:
                #If the classifier was removed from the check_config_files function then jump iteration
                if out_q is not None:
                    list_to_write = [[None], [None]]
                    self.manage_write_queue_child(out_q, list_to_write)
                continue

            #Check and build patterns structure
            self.check_patterns_files(classifier_name)
            self.patterns_files_extract(classifier_name, list_divided)

            #CALL CLASSIFIER CLASS CONSTRUCTOR#
            #Some classifiers needs first the patterns extracted to define somethings like the number of
            # inputs in NN case
            self.base_classifier_scheduler_constructor(classifier_name)
            self.check_patterns_dimensions_classifiers_configuration(classifier_name)

            if out_q is not None:
                list_to_write = [self.context["classifiers"][classifier_name],
                                 self.context["patterns"].patterns[classifier_name]]
                self.manage_write_queue_child(out_q, list_to_write)

    #########################################################################

    # def points_to_classifier_order(self, y):
    #     """
    #     Return a list of integers to order the classifier_list
    #     """
    #     l = []
    #     while y.find("_") is not -1:
    #         l.append(int(y[y.find("_")+1]:y.find("_", y.find("_"))))
    #     return l

    #########################################################################

    def order_classifier_list(self):
        if len(self.context["classifier_list"]) > 1:
            try:
                self.context["classifier_list"] = \
                    [x for x in sorted(self.context["classifier_list"], key=lambda y: (y
                        # self.points_to_classifier_order(y)
                        # int(y[re.search(r'[0-9]+', y).start():re.search(r'[0-9]+', y).end()]),
                        # int(y[y.find("_") + 1:y.find("_", y.find("_") + 1)]),
                        # int(y[y.find("-", y.find("-") + 1) + 1:])
                    ))
                    ]
            except:
                self.context["classifier_list"] = \
                    [x for x in sorted(self.context["classifier_list"], key=lambda y:
                    int(y[re.search(r'[0-9]+', y).start():re.search(r'[0-9]+', y).end()]))
                    ]

    #########################################################################

    def define_file_paths(self, classifier_name):
        """
        Define the paths of each classifier and ensemble.
        Check if the path exist and create it if not
        """
        self.context["classifiers"][classifier_name]["paths"] = {}

        try:
            self.context["classifiers"][classifier_name]["paths"]["config_file"] = \
                self.context["general_path"] + "config_file/" + self.context["classifiers"][
                    classifier_name]["set"] + '/'
        except:
            print(classifier_name)

        self.context["classifiers"][classifier_name]["paths"]["results"] = \
            self.context["general_path"] + "results/" + self.context["classifiers"][classifier_name][
                "set"] + '/'
        self.context["classifiers"][classifier_name]["paths"]["patterns"] = \
            self.context["general_path"] + "patterns/" + self.context["classifiers"][classifier_name][
                "set"] + '/'
        #Check path and create if doesnt exists
        path_exists(self.context["classifiers"][classifier_name]["paths"]["results"])
        path_exists(self.context["classifiers"][classifier_name]["paths"]["config_file"])
        path_exists(self.context["classifiers"][classifier_name]["paths"]["patterns"])
        #Define files names
        self.context["classifiers"][classifier_name]["results"] = \
            self.context["classifiers"][classifier_name]["paths"]["results"] + self.context[
                "pattern_kind"] + "_" + classifier_name + "_" + self.context["result_name"]

    #########################################################################

    def check_config_files(self, classifier_name):
        """
        Check if the config file of the classifier exists
        There is no need of ensembles config files at the moment
        """
        self.context["classifiers"][classifier_name]["config_file"] = \
            self.context["classifiers"][classifier_name]["paths"]["config_file"] + classifier_name + ".dat"
        if os.path.isfile(self.context["classifiers"][classifier_name]["config_file"]):
            self.context["classifiers"][classifier_name]["overwrite"] = 0
            #If the execution implies modifying the file
            if "learning" in self.context["execution_kind"]:
                if self.context["execution_kind"] not in self.ensemble_learning_process_names:
                    if not self.context["automatic_continue_learning"]:
                        reply = input('You are trying to create the classifier %s file\
                                     structure with a name that already exists for the set %s. Do you want to overwrite'
                                      ' the existing classifier?. \
                         Type yes or no \n' % (classifier_name, self.context["classifiers"][classifier_name]["set"]))
                        if reply == "no":
                            raise NameError("Change the name of the classifier %s for the set %s" % (
                                classifier_name, self.context["classifiers"][classifier_name]["set"]))
                        elif reply == "yes":
                            self.context["classifiers"][classifier_name]["overwrite"] = 1
                        else:
                            raise NameError("Not a valid reply")
                    else:
                        if self.context["interactive"]["activate"]:
                            print("Classifier %s already exist and will be removed from list due to "
                                  "automatic_continue_learning variable activated" % classifier_name)
                        del (self.context["classifier_list"][self.context["classifier_list"].index(classifier_name)])

        elif "learning" not in self.context["execution_kind"] and "preprocess" not in self.context["execution_kind"]:
            raise NameError("Trying to execute a process over the classifier %s that doesnt exist at dir %s \n \
                            Create it with learning process and try again" % (classifier_name,
                                                                              self.context["classifiers"][
                                                                                  classifier_name]["config_file"]))

    #########################################################################

    def check_patterns_files(self, classifier_name):
        #Patterns files checks
        for file_type in self.context["patterns_texts"]:
            if self.context["classifiers"][classifier_name]["paths"]["patterns"] not in \
                    self.context["classifiers"][classifier_name]["patterns"][file_type]:
                #TODO: solve the problem that causes the re edition of the pattern path
                self.context["classifiers"][classifier_name]["patterns"][file_type] = \
                    self.context["classifiers"][classifier_name]["paths"]["patterns"] + \
                    self.context["classifiers"][classifier_name]["patterns"][file_type]
            if not os.path.isfile(self.context["classifiers"][classifier_name]["patterns"][file_type]):
                raise NameError('The pattern of kind %s doesnt exist as it passed for the classifier %s, at dir %s'
                                % (file_type, classifier_name,
                                   self.context["classifiers"][classifier_name]["patterns"][file_type]))

    #########################################################################

    def obtain_min_classifiers_list(self, ensemble_name):
        """
        Check if the classifiers´s classes has the same length that the ensemble´s classes, returning 1 or 0.
        Also return the minimum list of classifiers that can aggregate all the classes of the ensembles in the pattern
        construction.
        """
        lengths = 1
        min_classifiers_list = []

        for i, class_text in enumerate(self.context["classifiers"][ensemble_name]["classes_names"]):
            for classifier_name in self.context["classifiers"][ensemble_name]["classifiers"]:
                if class_text in self.context["classifiers"][classifier_name]["classes_names"]:
                    if classifier_name not in min_classifiers_list:
                        #Classes would be repeated among classifiers of this list
                        min_classifiers_list.append(classifier_name)
                if len(self.context["classifiers"][classifier_name]["classes_names"]) != len(
                        self.context["classifiers"][ensemble_name]["classes_names"]):
                    lengths = 0

        return lengths, min_classifiers_list

    #########################################################################

    def check_ensembles_classes(self, ensemble_name):
        """
        Check if all classifiers has the same classes in conjunction that the ensemble,
        raising error if we are not generating ensembles automatically.
        """
        values = np.zeros(len(self.context["classifiers"][ensemble_name]["classes_names"]))
        for i, class_text in enumerate(self.context["classifiers"][ensemble_name]["classes_names"]):
            for classifier_name in self.context["classifiers"][ensemble_name]["classifiers"]:
                if class_text in self.context["classifiers"][classifier_name]["classes_names"]:
                    values[i] = 1

        if np.sum(values) != len(values):
            if not self.context["ensembles_combination_on_fly"]["activate"] and \
                            "deployment" not in self.context["execution_kind"]:
                raise ValueError(
                    "Members of Ensemble %s does not contain the classes %s that is present in the ensemble"
                    % (ensemble_name,
                       [self.context["classifiers"][ensemble_name]["classes_names"]
                        [i] for i in values if values[i] == 0.]))
            else:
                return 0
        else:
            return 1

    #########################################################################

    def data_preprocess_transformation(self, classifier_name, list_divided, pattern_kind):
        if self.context["classifiers"][classifier_name]["data_transformation"] is not None:
            from preprocess import PreProcess

            if "transformer" not in self.context["classifiers"][classifier_name] or \
                            self.context["classifiers"][classifier_name]["transformer"] is None:
                if "deployment" not in self.context["execution_kind"]:
                    PreProcess().create_data_transformation([classifier_name], list_divided, None, self.context)
                else:
                    #Means we are in deployment so the transformation will occur in deployment_classification
                    return

            PreProcess().apply_data_transformation(classifier_name, self.context, pattern_kind)

    #########################################################################

    def patterns_files_extract(self, classifier_name, list_divided):
        ############################
        ##Extract patterns and kind#
        ############################
        from auxiliar import check_equal_classifier_patterns
        if classifier_name in self.context["classifier_list"]:
            for pattern_kind in self.context["patterns_texts"]:
                # if not self.context["results"]["instances_error"]["activate"] and not \
                #     self.context["results"]["classes_error"]["activate"]:
                # TODO: (Aaron) I added the if below because check_equal_classifier_patterns gave me problems with regression
                if self.context["execution_kind"] != "deployment_regression":
                    for classifier_name_2 in list_divided:
                        if check_equal_classifier_patterns(self.context,
                                                           classifier_name,
                                                           classifier_name_2,
                                                           pattern_kind):

                            self.context["patterns"].patterns[classifier_name][pattern_kind] = \
                                self.context["patterns"].patterns[classifier_name_2][pattern_kind]
                            if "transformer" in self.context["classifiers"][classifier_name_2] and \
                                            self.context["classifiers"][classifier_name_2]["transformer"] is not None:
                                self.context["classifiers"][classifier_name]["transformer"] = \
                                    self.context["classifiers"][classifier_name_2]["transformer"]
                            break

                if self.context["patterns"].patterns[classifier_name][pattern_kind] is None:
                    self.context["patterns"].extract(self.context, classifier_name, pattern_kind)
                    #Build the transformer if corresponds and transform it, only if it is the first classifier in
                    # creating a shared pattern or if the pattern is exclusive of this classifier
                    self.data_preprocess_transformation(classifier_name, list_divided, pattern_kind)

        elif classifier_name in self.context["ensemble_list"]:
            if self.context["ml_paradigm"] == "classification":
                self.check_ensembles_classes(classifier_name)
            lengths, min_classifiers_list = self.obtain_min_classifiers_list(classifier_name)

            if lengths:
                deep = 0
                self.context["patterns"].ensemble_copy(self.context, classifier_name, deep)
            else:
                self.context["patterns"].construct_ensemble_patterns_multiple_classifiers(self.context,
                                                                                          classifier_name,
                                                                                          min_classifiers_list)

                #########################################################################

    def check_patterns_dimensions_classifiers_configuration(self, classifier_name):
        classes_texts = self.context["classifiers"][classifier_name]["classes_names"]
        #Check that all patterns files has the same number of inputs
        for pattern_kind in self.context["patterns_texts"]:
            if "deployment" in self.context["execution_kind"]:
                if pattern_kind != "test":
                    input_dimension = len(self.context["patterns"].patterns[classifier_name]["learning"][0]) - len(
                        classes_texts)
                else:
                    input_dimension = len(self.context["patterns"].patterns[classifier_name]["test"][0])
            else:
                input_dimension = len(self.context["patterns"].patterns[classifier_name]["learning"][0]) - len(
                    classes_texts)

            if len(self.context["patterns"].patterns[classifier_name][pattern_kind][0][
                   :input_dimension]) != input_dimension:
                raise ValueError("Patterns of kind %s and %s of the classifier %s doesnt match"
                                 % (self.context["patterns_texts"][0], pattern_kind, classifier_name))

                #########################################################################

    def base_classifier_scheduler_constructor(self, classifier_name):
        self.context["classifiers"][classifier_name]["instance"] = getattr(classifiers, self.context["classifiers"][
            classifier_name]["classifier_kind"]["kind"])(self.context, classifier_name)
        if os.path.isfile(self.context["classifiers"][classifier_name]["config_file"]):
            if self.context["classifiers"][classifier_name]["overwrite"] == 0:
                old_context = \
                    self.context["classifiers"][classifier_name]["instance"].load_config_file(self.context,
                                                                                              classifier_name)
                # if type(old_context) != dict:
                #     #To deprecate
                #     old_context = old_context[1]

                if self.context["execution_kind"] != "reconfiguring":
                    self.context["classifiers"][classifier_name] = old_context
                if "instance" in old_context:
                    self.context["classifiers"][classifier_name]["instance"] = old_context["instance"]
                elif "instances" in old_context:
                    self.context["classifiers"][classifier_name]["instance"] = old_context["instances"]
                else:
                    raise ValueError("No instance was saved. Put it to learn again")

    #########################################################################

    def pattern_range_check(self):
        """
        -This function alters the pattern extraction depending on the activation function.

        -If the activation function is on the range_1_1 list, then the pattern extraction construct a list of patterns
        with the outputs in [-1,1] range. Otherwise a zero-one list is created.

        -Statistics and other functions are not affected (because it is controlled inside those functions) and always
        works with zero-one.
        """
        range_1_1 = ["tanh", "TanSig"]
        range_0_1 = ["sigmoid"]
        for classifier_name in self.context["classifier_list"]:
            try:
                if "transfer_function" in self.context["classifiers"][classifier_name]["classifier_kind"]:
                    #If empty, create default range
                    if type(self.context["classifiers"][classifier_name]["classifier_kind"]["transfer_function"]) \
                            == AutoVivification:
                        self.context["classifiers"][classifier_name]["patterns"]["range"] = [0, 1]
                    else:
                        if self.context["classifiers"][classifier_name]["classifier_kind"][
                            "transfer_function"] in range_1_1:
                            self.context["classifiers"][classifier_name]["patterns"]["range"] = [-1, 1]
                        elif self.context["classifiers"][classifier_name]["classifier_kind"][
                            "transfer_function"] in range_0_1:
                            self.context["classifiers"][classifier_name]["patterns"]["range"] = [0, 1]
                        else:
                            raise ValueError(
                                "Function not defined in pattern range check or even not implemented in classifiers")
                else:
                    if "kernel" in self.context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]:
                        self.context["classifiers"][classifier_name]["patterns"]["range"] = [0, 1]
                    #If empty, create default range
                    else:
                        self.context["classifiers"][classifier_name]["patterns"]["range"] = [0, 1]
            except:
                if "patterns" not in self.context["classifiers"][classifier_name]:
                    raise NameError("Patterns names were not defined in the configuration file")
                else:
                    self.context["classifiers"][classifier_name]["patterns"]["range"] = [0, 1]

        #TODO: Problems with ensembles that has classifiers with different ranges
        for ensemble_name in self.context["ensemble_list"]:
            if "patterns" not in self.context["classifiers"][ensemble_name].keys():
                self.context["classifiers"][ensemble_name]["patterns"] = {}
            self.context["classifiers"][ensemble_name]["patterns"]["range"] = [0, 1]
            for classifier_name in self.context["classifiers"][ensemble_name]["classifiers"]:
                if self.context["classifiers"][classifier_name]["patterns"]["range"] == [-1, 1]:
                    self.context["classifiers"][ensemble_name]["patterns"]["range"] = [-1, 1]

    #########################################################################

    def ensemble_generation_base(self, classifier_list, out_q):
        #  Taking each classifier from list
        for new in classifier_list:
            if type(new) == itertools.combinations:
                for x_tuple in new:
                    ensemble_name = "+".join([x for x in x_tuple])
                    self.context["classifiers"][ensemble_name] = {}
                    for key in self.context["ensembles_combination_on_fly"]["ensemble_model"].keys():
                        if key != "classifiers":
                            self.context["classifiers"][ensemble_name][key] = \
                                self.context["ensembles_combination_on_fly"]["ensemble_model"][key]
                        else:
                            self.context["classifiers"][ensemble_name][key] = list(x_tuple)

                    if not self.check_ensembles_classes(ensemble_name):
                        del (self.context["classifiers"][ensemble_name])

        list_to_write = [self.context]
        self.manage_write_queue_child(out_q, list_to_write)

    #########################################################################
    @staticmethod
    def automatic_ensemble_generation(given_classifier_list, specific_amounts, minimum_number=2):
        if len(specific_amounts):
            classifier_list = []
            for num_combination in specific_amounts:
                temp = itertools.combinations(given_classifier_list, num_combination)
                classifier_list.append(temp)
        else:
            classifier_list = []
            for num_combination in range(minimum_number,
                                         len(given_classifier_list) + 1):
                temp = itertools.combinations(given_classifier_list, num_combination)
                classifier_list.append(temp)
        return classifier_list

    #########################################################################

    def ensemble_generation(self):
        if "ensembles_combination_on_fly" in self.context and self.context["ensembles_combination_on_fly"][
            "activate"]:
            #  Taking each classifier from list
            classifier_list = self.automatic_ensemble_generation(
                self.context["classifier_list"],
                self.context["ensembles_combination_on_fly"]["specific_amounts"],
                self.context["ensembles_combination_on_fly"]["minimum_number"])

            classifier_name_list = self.statically_distribute_list(classifier_list)
            self.generic_parallel_function(self.ensemble_generation_base,
                                           classifier_name_list,
                                           objectives=[self.context],
                                           recolect=1)
            if self.context["interactive"]["activate"]:
                print("Ensembles generated automatically: %d" %
                      (len(self.context["classifiers"].keys()) - len(self.context["classifier_list"])))

    #########################################################################

    def curve_roc(self, classifier_name, information):

        pattern_kind = self.context["pattern_kind"]

        if classifier_name in self.context["classifier_list"]:
            information.build_real_outputs(self.context, classifier_name, pattern_kind)

        information.discretize_outputs_for_roc(self.context, classifier_name,
                                               information.info[classifier_name]["continuous_outputs"][
                                                   pattern_kind])

        information.build_roc(self.context, classifier_name,
                              self.context["patterns"].patterns[classifier_name][pattern_kind])

    #########################################################################

    def build_statistics(self, classifier_name, information, statistic_class, pattern_kind):
        if self.context["ml_paradigm"] == "classification":
            statistic_class.goodness(
                self.context,
                classifier_name,
                information.info[classifier_name]["discretized_outputs"][pattern_kind],
                self.context["patterns"].patterns[classifier_name][pattern_kind])

        for statistic in [x for x in self.context["results"]["to_file"]["measures"].keys()
                          if self.context["results"]["to_file"]["measures"][x]]:
            getattr(statistic_class, statistic)(classifier_name, self.context, information, pattern_kind)

        if "diversity_study" in self.context["results"]["to_file"] and \
                self.context["results"]["to_file"]["diversity_study"]["activate"]:
            statistic_class.build_list_oracle_outputs(classifier_name)

    #########################################################################

    def to_file_information(self, classifier_name, information, statistic_class):

        pattern_kind = self.context["pattern_kind"]

        if classifier_name in self.context["classifier_list"]:
            information.build_real_outputs(self.context, classifier_name, pattern_kind)
            if self.context["ml_paradigm"] == "classification":
                information.discretize_outputs(self.context, classifier_name, pattern_kind)

        self.build_statistics(classifier_name, information, statistic_class, pattern_kind)

    #########################################################################

    def learning_stability(self, classifier_name, information, statistic_class):

        pattern_kind = self.context["pattern_kind"]
        information.build_real_outputs(self.context, classifier_name, pattern_kind)

        information.discretize_outputs(self.context, classifier_name, pattern_kind)

        statistic_class.goodness(self.context, classifier_name,
                                 information.info[classifier_name]["discretized_outputs"][pattern_kind],
                                 self.context["patterns"].patterns[classifier_name][pattern_kind])

    #########################################################################

    def basic_ensemble_information_base(self, ensemble_name_list, out_q, information):
        """
    :param ensemble_name_list:
    :param out_q:
    :param information:
    """
        from statistics import Statistic

        statistic_class = Statistic()
        pattern_kind = [self.context["pattern_kind"]]
        for ensemble_name, i in zip(ensemble_name_list, range(len(ensemble_name_list))):
            if i % 10 == 1 and self.context["interactive"]["activate"]:
                sys.stdout.write("\r{0}>".format("Completed:%f%%" % ((float(i) / len(ensemble_name_list)) * 100)))
                sys.stdout.flush()

            if self.context["results"]["to_file"]["activate"]:
                Ensemble(self.context, ensemble_name, information, pattern_kind)
                self.to_file_information(ensemble_name, information, statistic_class)

            if self.context["results"]["roc"]["activate"]:
                self.context["classifiers"][ensemble_name]["outputs_kind"] = "continuous_outputs"
                Ensemble(self.context, ensemble_name, information, pattern_kind)
                self.curve_roc(ensemble_name, information)

            if self.context["results"]["classes_error"]["activate"]:
                Ensemble(self.context, ensemble_name, information, pattern_kind)
                self.classes_error(ensemble_name, information)

            if self.context["results"]["instances_error"]["activate"]:
                Ensemble(self.context, ensemble_name, information, pattern_kind)
                self.instances_error(ensemble_name, information)

            if self.context["results"]["scatter"]["activate"]:
                Ensemble(self.context, ensemble_name, information, pattern_kind)
                self.scatter(ensemble_name, information, statistic_class)

        list_to_write = [information.info, statistic_class.measures]
        self.manage_write_queue_child(out_q, list_to_write)

    #########################################################################

    def basic_classifier_information_base(self, classifier_name_list, out_q):
        from statistics import Statistic

        information = ClassifiersInfo()
        statistic_class = Statistic()

        for classifier_name, i in zip(classifier_name_list, range(len(classifier_name_list))):
            if i % 10 == 1 and self.context["interactive"]["activate"]:
                sys.stdout.write("\r{0}>".format("Completed:%f%%" % ((float(i) / len(classifier_name_list)) * 100)))
                sys.stdout.flush()

            if self.context["results"]["to_file"]["activate"]:
                self.to_file_information(classifier_name, information, statistic_class)

            if "roc" in self.context["results"] and self.context["results"]["roc"]["activate"]:
                self.curve_roc(classifier_name, information)

            if "classes_error" in self.context["results"] and self.context["results"]["classes_error"]["activate"]:
                self.classes_error(classifier_name, information)

            if "instances_error" in self.context["results"] and self.context["results"]["instances_error"][
                "activate"]:
                self.instances_error(classifier_name, information)

            if "learning_stability" in self.context["results"] and self.context["results"]["learning_stability"][
                "activate"]:
                self.learning_stability(classifier_name, information, statistic_class)

            if "configuration_evaluation" in self.context["results"] and \
                    self.context["results"]["configuration_evaluation"]["activate"]:
                self.configuration_evaluation(classifier_name, information, statistic_class)

            if "scatter" in self.context["results"] and self.context["results"]["scatter"]["activate"]:
                self.scatter(classifier_name, information, statistic_class)

        list_to_write = [information.info, statistic_class.measures]
        self.manage_write_queue_child(out_q, list_to_write)

    #########################################################################

    def information_builder(self):
        """
    Build basic classifiers and ensembles information.
    Work with all the elements defined in the context AutoViVification structure
    """
        from statistics import Statistic

        information = ClassifiersInfo()
        statistic_class = Statistic()

        if "weights_file" in self.context["results"] and self.context["results"]["weights_file"]["activate"]:
            #TODO: fix this function call
            self.classifier_info_file()

        classifier_name_list = self.statically_distribute_list(self.context["classifier_list"])
        self.generic_parallel_function(self.basic_classifier_information_base, classifier_name_list,
                                       objectives=[information.info, statistic_class.measures], recolect=1)

        ensemble_name_list = self.statically_distribute_list(self.context["ensemble_list"])
        self.generic_parallel_function(self.basic_ensemble_information_base, ensemble_name_list,
                                       objectives=[information.info, statistic_class.measures], args=[information],
                                       recolect=1)

        self.schedule_results_kind(statistic_class, information)

    #########################################################################

    def scatter(self, classifier_name, information, statistic_class):

        pattern_kind = self.context["pattern_kind"]

        if classifier_name in self.context["classifier_list"]:
            information.build_real_outputs(self.context, classifier_name, pattern_kind)
            information.discretize_outputs(self.context, classifier_name, pattern_kind)

        statistic_class.goodness(self.context, classifier_name,
                                 information.info[classifier_name]["discretized_outputs"][pattern_kind],
                                 self.context["patterns"].patterns[classifier_name][pattern_kind])

    #########################################################################

    def configuration_evaluation(self, classifier_name, information, statistic_class):
        """
    Scheduler to control de parametric class.
    """
        information.build_real_outputs(self.context, classifier_name, "validation")
        statistic_class.configuration_evaluation(self.context, classifier_name, information)

    #########################################################################

    def classes_error(self, classifier_name, information):
        """
    Scheduler to control the classes error graphic construction.
    """
        information.classes_error(self.context, classifier_name)

    #########################################################################

    def instances_error(self, classifier_name, information):
        """
    Scheduler to control the instances error graphic construction.
    """
        information.instances_error(self.context, classifier_name)

    #########################################################################

    def reconfiguring_base(self, classifier_list, out_q=""):
        for classifier_name in classifier_list:
            if self.context["classifiers"][classifier_name]["thresholds"][
                "determine_threshold"] == 1 and self.context["classifiers"][classifier_name]["thresholds"][
                    "threshold_kind"] != "manual":

                information = ClassifiersInfo()
                for pattern_kind in [x for x in self.context["patterns_texts"] if x != "test"]:
                    information.build_real_outputs(self.context, classifier_name, pattern_kind)
                information.automatic_threshold_determine(self.context, classifier_name)

            self.context["classifiers"][classifier_name]["instance"].save_config_file(self.context, classifier_name)

        list_to_write = [[None], [None]]
        self.manage_write_queue_child(out_q, list_to_write)

    #########################################################################

    def reconfiguring(self):
        """
    Used to adapt old data files structures.
    """
        classifier_name_list = self.statically_distribute_list(self.context["classifier_list"])
        self.generic_parallel_function(self.reconfiguring_base, classifier_name_list, recolect=0)

    ####################################################

    def discretize(self, threshold, value):
        """
    Used like a lambda function
    """
        if value < threshold:
            return 0
        else:
            return 1

    ####################################################

    def learning(self, list_divided, out_q):
        """
    Schedule the learning of an individual classifier
    """
        for i, classifier_name in enumerate(list_divided):
            if self.context["interactive"]["activate"]:
                sys.stdout.write("\r{0}>".format("Trained:%f%%" % ((float(i) / len(list_divided)) * 100)))
                sys.stdout.flush()
            if self.context["execution_kind"] == "iterative_train":
                self.context["classifiers"][classifier_name]["instance"].core_iterative_learning(self.context,
                                                                                                 classifier_name)
            #Iterative Learning save the structure into the function
            elif self.context["execution_kind"] == "learning_draw":
                self.context["classifiers"][classifier_name]["instance"].core_learning_draw(self.context,
                                                                                            classifier_name)
            elif self.context["execution_kind"] == "learning":
                self.context["classifiers"][classifier_name]["instance"].core_learning(self.context,
                                                                                       classifier_name)

            elif self.context["execution_kind"] == "evolutive_train":
                self.context["classifiers"][classifier_name]["instance"].core_evolutive_learning(self.context,
                                                                                                 classifier_name)
                #Common to learning process
            if self.context["execution_kind"] != "NClearning" and self.context["ml_paradigm"] == "classification":
                self.reconfiguring_base([classifier_name])  # Include the save_config_file
            else:
                self.context["classifiers"][classifier_name]["instance"].save_config_file(self.context,
                                                                                          classifier_name)

        list_to_write = [[None], [None]]
        self.manage_write_queue_child(out_q, list_to_write)

    ####################################################

    def statically_distribute_list(self, classifier_list):
        if type(classifier_list) is not list:
            classifier_list = list(classifier_list)

        if self.context["interactive"]["activate"]:
            if classifier_list == self.context["classifier_list"]:
                print("\nNumber of classifiers to compute:", len(classifier_list))
            elif classifier_list == self.context["ensemble_list"]:
                print("\nNumber of ensembles to compute:", len(classifier_list))
            else:
                print("\nNumber of elements to compute:", len(classifier_list))

        size = len(classifier_list) // multiprocessing.cpu_count()
        if size == 0:
            if len(classifier_list) != 0:
                size = 1
                return [classifier_list[i:i + size + 1] for i in range(0, len(classifier_list), size + 1)]
            else:
                return []
        else:
            return [classifier_list[i:i + size + 1] for i in range(0, len(classifier_list), size + 1)]

    ####################################################

    def parallel_negative_correlation_learning(self):
        classifier_name_list = self.statically_distribute_list(self.context["ensemble_list"])
        self.generic_parallel_function(self.correlation_learning, classifier_name_list, recolect=0)

    ####################################################

    def correlation_learning(self, ensemble_list, out_q=""):
        from statistics import Statistic

        for ensemble_name in ensemble_list:
            information = ClassifiersInfo()
            statistics = Statistic()
            if self.context["classifiers"][ensemble_name]["outputs_kind"] == "discretized_outputs":
                raise ValueError("Not possible to make a correlation learning with discretized outputs")

            outputs_kind = "continuous_outputs"

            frozen_list = list(self.context["classifiers"][ensemble_name]["classifiers"])
            for j, classifier_name in enumerate(frozen_list):
                self.context["classifiers"][ensemble_name]["classifiers"] = frozen_list[:j + 1]
                for i in range(self.context["classifiers"][ensemble_name]["learning_parameters"]["epochs"]):

                    information.build_real_outputs(self.context, classifier_name, "learning")
                    Ensemble(self.context, ensemble_name, information, ["learning"])
                    info = information.info[ensemble_name][outputs_kind]["learning"]
                    for x in range(len(info)):
                        for z in range(len(info[x])):
                            info[x][z] = Statistic().change_ranges(info[x][z],
                                                                   oldMin=0, oldMax=1, newMin=-1, newMax=1)

                    self.context["classifiers"][classifier_name]["instance"]. \
                        core_learning(self.context, classifier_name,
                                      ensemble_error=information.info[ensemble_name][outputs_kind]["learning"])

                if self.context["interactive"]["activate"]:
                    for classifier_name in self.context["classifiers"][ensemble_name]["classifiers"]:
                        information.build_real_outputs(self.context, classifier_name, "validation")
                    Ensemble(self.context, ensemble_name, information, ["learning", "validation"])
                    statistics.rms(ensemble_name,
                                   self.context,
                                   information,
                                   "learning")
                    statistics.rms(ensemble_name,
                                   self.context,
                                   information,
                                   "validation")
                    if self.context["interactive"]["activate"]:
                        print(ensemble_name + ": Finished classifier" + classifier_name + "learning error:",
                              str(statistics.measures[ensemble_name]["rms"]["learning"]),
                              "validation error:", str(statistics.measures[ensemble_name]["rms"]["validation"]),
                              "total_error:", (statistics.measures[ensemble_name]["rms"]["learning"] +
                                               statistics.measures[ensemble_name]["rms"]["validation"]))
                        ####
                self.reconfiguring_base([classifier_name])

        list_to_write = [[None], [None]]
        self.manage_write_queue_child(out_q, list_to_write)

    ####################################################

    def scheduler_learning(self):
        if len(self.context["classifier_list"]):
            if self.context["execution_kind"] == "NClearning":
                self.parallel_negative_correlation_learning()
            else:
                self.parallel_learning(self.context["classifier_list"])

    ####################################################

    def parallel_learning(self, init_classifier_list):
        """
    Calling the learning_base function from generic parallel function.
    Distribute the Classifier list from the arg into list of classifiers to send it to different process.
    """
        classifier_name_list = self.statically_distribute_list(init_classifier_list)
        self.generic_parallel_function(self.learning, classifier_name_list, recolect=0)

    #########################################################################

    def pairwise_diversity_scheduler(self, classifier_list, out_q, statistic_class):
        """
    For each kind of pairwise diversity measure marked, calculate into statistic_class structure
    """
        for function in self.context["results"]["to_file"]["diversity_study"]["pairwise_diversity"].keys():
            if self.context["results"]["to_file"]["diversity_study"]["pairwise_diversity"][function]:
                statistic_class.diversity_pairwise_structure(self.context, function, classifier_list)

        list_to_write = [statistic_class.measures]
        self.manage_write_queue_child(out_q, list_to_write)

    #############################################

    def non_pairwise_diversity_scheduler(self, classifier_list, out_q, statistic_class):
        """
    For each kind of non-pairwise diversity measure marked, calculate into statistic_class structure
    """
        for function in self.context["results"]["to_file"]["diversity_study"]["non_pairwise_diversity"].keys():
            if self.context["results"]["to_file"]["diversity_study"]["non_pairwise_diversity"][function]:
                statistic_class.diversity_non_pairwise_structure(self.context, function, classifier_list)

        list_to_write = [statistic_class.measures]
        self.manage_write_queue_child(out_q, list_to_write)

    #############################################

    def parallel_diversity(self, statistic_class):

        classifier_name_list = self.statically_distribute_list(self.context["classifiers"].keys())
        self.generic_parallel_function(self.pairwise_diversity_scheduler, classifier_name_list,
                                       objectives=[statistic_class.measures],
                                       args=[statistic_class], recolect=1)

    #########################################################################

    def parallel_non_pairwise_diversity(self, statistic_class):
        if self.context["results"]["to_file"]["diversity_study"]["over_ensembles"]:
            if not len(self.context["ensemble_list"]):
                if not self.context["ensembles_combination_on_fly"]["activate"]:
                    warnings.warn("ensembles_combination variable not activated but over_ensembles is activated to"
                                  "calculate the non_diversity_measures over it.")
            classifier_name_list = self.statically_distribute_list(self.context["ensemble_list"])
            self.generic_parallel_function(self.non_pairwise_diversity_scheduler, classifier_name_list,
                                           objectives=[statistic_class.measures], args=[statistic_class],
                                           recolect=1)
        else:
            out_q = ""
            self.non_pairwise_diversity_scheduler([self.context["classifier_list"]], out_q, statistic_class)

    #########################################################################

    def execution(self):
        """
        Main scheduler of the execution.
        """
        if "deployment" in self.context["execution_kind"]:
            if self.context["execution_kind"] == "deployment_regression":
                self.deployment_regression()
            elif self.context["execution_kind"] == "deployment_classification":
                self.deployment_classification()
            else:
                raise NameError("Execution kind has a deployment function that does not exists."
                                "Try with deployment_classification or deployment_regression")
        elif self.context["execution_kind"] == "preprocess":
            from preprocess import PreProcess

            PreProcess().schedule(self.context)
        elif self.context["execution_kind"] == "reconfiguring":
            self.reconfiguring()
        elif "learning" in self.context["execution_kind"]:
            self.scheduler_learning()
        elif self.context["execution_kind"] == "results":
            self.information_builder()
        else:
            self.manage_execution_kind_error()

    #########################################################################

    def aggregated_results(self, statistic_class):
        if self.context["results"]["to_file"]["activate"]:
            if "diversity_study" in self.context["results"]["to_file"] and \
                    self.context["results"]["to_file"]["diversity_study"]["activate"]:
                if sum(self.context["results"]["to_file"]["diversity_study"]["pairwise_diversity"].values()):
                    self.parallel_diversity(statistic_class)
                if sum(self.context["results"]["to_file"]["diversity_study"]["non_pairwise_diversity"].values()):
                    self.parallel_non_pairwise_diversity(statistic_class)

        elif self.context["results"]["configuration_evaluation"]["activate"]:
            statistic_class.best_choice()

    #########################################################################

    def schedule_results_kind(self, statistic_class, information):
        from presentations import Presentation
        #Build new summary results from generated information
        self.aggregated_results(statistic_class)

        #DETERMINE WHICH KIND OF RESULTS HAS TO BE CONSTRUCTED
        if self.context["results"]["to_file"]["activate"]:
            for file_type in [x for x in self.context["results"]["to_file"]["type_file"]
                              if self.context["results"]["to_file"]["type_file"][x] == 1]:
                getattr(Presentation(), file_type)(self.context, statistic_class.measures, information.info)
        elif self.context["results"]["validation_bars"]["activate"] or \
                self.context["results"]["validation_bars_3D"]["activate"]:
            Presentation().validation_bars(self.context, statistic_class.measures)
        elif self.context["results"]["rms_vs_E"]["activate"]:
            Presentation().error_vs_rms(self.context, statistic_class.measures)
        elif self.context["results"]["scatter"]["activate"]:
            Presentation().scatter(self.context, statistic_class.measures)
        elif self.context["results"]["roc"]["activate"]:
            Presentation().roc_curve(self.context, statistic_class.measures)
        elif self.context["results"]["classes_error"]["activate"]:
            Presentation().paint_classes_error(self.context, information)
        elif self.context["results"]["instances_error"]["activate"]:
            Presentation().paint_instances_error(self.context, information)
        elif self.context["results"]["learning_stability"]["activate"]:
            Presentation().learning_instability(self.context, statistic_class)
        elif self.context["results"]["configuration_evaluation"]["activate"]:
            Presentation().best_choice(self.context, statistic_class)
            #########################################################################
            #########################################################################
            #########################################################################
            #########################################################################
            #########################################################################

    #########################################################################
    def print_classification(self, classifier_name, information, pattern_kind):
        for i, class_name in enumerate(self.context["classifiers"][classifier_name]["classes_names"]):
            if information.info[classifier_name]["discretized_outputs"][pattern_kind][0][i]:
                print('{0} {1}'.format(
                    # classifier_name,
                    i,
                    (information.info[classifier_name]["continuous_outputs"][pattern_kind][0][i] * 100.),
                    # classifier_name
                    )
                )
            if information.info[classifier_name]["continuous_outputs"][pattern_kind][0][i] == 0.5:
                return

    #########################################################################

    def print_regression(self, classifier_name, information, pattern_kind):

        from statistics import Statistic
        d_change_pred, d_change_true = \
            Statistic().pre_forecasting_statistic(self.context, classifier_name, information, pattern_kind)

        predicted_value = information.info[classifier_name]["continuous_outputs"][pattern_kind][0][0]
        tendency = d_change_pred[0]
        if tendency < 0.0:
            tendency_to_print = "negative"
        elif tendency > 0.0:
            tendency_to_print = "positive"
        else:
            tendency_to_print = "same value"

        print("{0:.3f}".format(predicted_value))
        # print('Tendency: {0};\t Predicted value: {1}'.format(tendency_to_print, predicted_value))

    #########################################################################

    def deployment_regression(self):
        """
        Deployment execution of the intelligent system in regression context
        """
        pattern_kind = "test"
        if len(self.context["ensemble_list"]) > 0:
            #Ensemble decision
            if len(self.context["ensemble_list"]) > 1:
                raise ValueError("More than one ensemble are not admitted")
            else:
                ensemble_name = self.context["ensemble_list"][0]
                classifier_name_list = self.context["classifiers"][ensemble_name]["classifiers"]
                information = ClassifiersInfo()
                for classifier_name in classifier_name_list:
                    information.build_real_outputs(self.context, classifier_name, pattern_kind)
                # Building ensemble
                Ensemble(self.context, ensemble_name, information, [pattern_kind])
                self.print_regression(ensemble_name, information, pattern_kind)

        elif len(self.context["ensemble_list"]) == 0:
            # Monolithic classification
            classifier_name = self.context["classifier_list"][0]
            information = ClassifiersInfo()
            information.build_real_outputs(self.context, classifier_name, pattern_kind)
            self.print_regression(classifier_name, information, pattern_kind)

    #########################################################################

    def deployment_classification(self):
        """
        Deployment execution of the intelligent system
        """
        pattern_kind = "test"
        from preprocess import PreProcess
        if len(self.context["ensemble_list"]) > 0:
            #Ensemble decision
            if len(self.context["ensemble_list"]) > 1:
                raise ValueError("More than one ensemble is not permitted")
            else:
                ensemble_name = self.context["ensemble_list"][0]
                classifier_name_list = self.context["classifiers"][ensemble_name]["classifiers"]
                # if self.context["classifiers"][ensemble_name]["classifiers"] != self.context["classifier_list"]:
                #     classifier_name_list = self.context["classifiers"][ensemble_name]["classifiers"]
                information = ClassifiersInfo()
                for classifier_name in classifier_name_list:
                    PreProcess().apply_data_transformation(classifier_name, self.context, pattern_kind)
                    information.build_real_outputs(self.context, classifier_name, pattern_kind)
                    information.discretize_outputs(self.context, classifier_name, pattern_kind)
                    # self.print_classification(classifier_name, information, pattern_kind)
                    # Building ensemble
                Ensemble(self.context, ensemble_name, information, [pattern_kind])
                self.print_classification(ensemble_name, information, pattern_kind)

        elif len(self.context["ensemble_list"]) == 0:
            #Monolithic classification
            classifier_name = self.context["classifier_list"][0]
            information = ClassifiersInfo()
            PreProcess().apply_data_transformation(classifier_name, self.context, pattern_kind)
            information.build_real_outputs(self.context, classifier_name, pattern_kind)
            information.discretize_outputs(self.context, classifier_name, pattern_kind)
            self.print_classification(classifier_name, information, pattern_kind)

    #########################################################################

    def manage_execution_kind_error(self):
        if np.sum([1 if self.context["execution_kind"] in x else 0 for x in self.execution_list]) == 0:
            string_output = ""
            for element in self.execution_list:
                string_output += "\n" + element
            raise ValueError("No execution process defined. Select one from the next list: %s" % string_output)

# This Python file uses the following encoding: utf-8
# !/usr/local/bin/python3.3
####################################################
# <Copyright (C) 2012, 2013, 2014, 2015 Yeray Alvarez Romero>
# This file is part of MULLPY.
####################################################
import copy
from auxiliar import AutoVivification
import sys


def batch_execution(classifier_list, batch_len,
                    original_parameters, aux_parameters,
                    execution_kind, classifiers_kind,
                    k_folds=None):
    """
    :param classifier_list: The complete list of classifiers names to compute. If k_folds is activated, all the members
    has to passed in this list with a number between underscores in the name after the classifier kind as k_fold member.
    :param batch_len: The amount of classifiers to be computed at the same time (batch)
    :param original_parameters: The common parameters defined for all the classifiers
    :param aux_parameters: The AutoVivification structure that contains all the classifiers pre-structure to be computed
    :param execution_kind: The name of the execution kind as a string
    :param classifiers_kind: A list of strings that contains the types of classifiers to be computed
    :param k_folds: The amount of k_folds to execute. None by default or a integer.
            To control that the different folds of the same classifier go together
    :return: None
    """
    import mullpy
    import numpy as np

    if k_folds is not None and isinstance(k_folds, int):
        denominator = k_folds
        from presentations import Presentation
        classifier_list = list(set([Presentation.identify_cross_validation_members(x) for x in classifier_list]))
    else:
        denominator = 1

    import random
    random.shuffle(classifier_list)
    #Check if in classifier_list all the classifiers kind are defined
    checking_list = list(set([x[:x.find("_")] if x.find("_") != -1 else x for x in classifier_list]))
    for classifier_kind in classifiers_kind:
        if classifier_kind not in checking_list:
            raise Exception("All the classifiers kind was not defined in the example")

    for i in range(int(np.ceil(len(classifier_list) / (batch_len // denominator)))):
        j = 0
        parameters = copy.deepcopy(original_parameters)
        while len(parameters["classifiers"].keys()) < batch_len:
            classifier_name = classifier_list[i * (batch_len // denominator) + j]
            if k_folds is not None and isinstance(k_folds, int):
                for k_fold in range(k_folds):
                    if classifier_name.count("_") > 1:
                        cl_parameters = classifier_name[classifier_name.find("_"):]
                        new_classifier_name = classifier_name[:classifier_name.find("_")+1] + \
                                              str(k_fold) + cl_parameters
                    else:
                        new_classifier_name = classifier_name + "_" + str(k_fold)

                    if new_classifier_name not in aux_parameters["classifiers"].keys():
                        raise ValueError("{} not founded in aux_parameters. "
                                         "Review build_classifiers_models function".format(new_classifier_name))
                    parameters["classifiers"][new_classifier_name] = aux_parameters["classifiers"][new_classifier_name]
            else:
                if classifier_name not in aux_parameters["classifiers"].keys():
                        raise ValueError("{} not founded in aux_parameters. "
                                         "Review build_classifiers_models function".format(classifier_name))
                parameters["classifiers"][classifier_name] = aux_parameters["classifiers"][classifier_name]

            if classifier_list[i * (batch_len // denominator) + j] == classifier_list[-1]:
                break
            j += 1

        execution_class = mullpy.Process(parameters)
        execution_class.execution()

        len_classifier_list = len(classifier_list)
        if k_folds is not None and isinstance(k_folds, int):
            len_classifier_list *= k_folds

        sys.stdout.write("\r{0}:{1}% of classifiers {2} from a total of {3}>".format(
            execution_kind,
            int((float(i * batch_len + j) / len_classifier_list * 100)),
            "_".join([x for x in classifiers_kind]),
            len_classifier_list))
        sys.stdout.flush()
####################################################


def evolutive_execution(classifier_list, batch_len,
                        original_parameters, aux_parameters,
                        execution_kind, classifiers_kind,
                        k_folds=None):
    pass

####################################################


def check_pickable_process(classifier_list):
    import sys
    import pickle
    import os
    import classifiers
    from patterns import Pattern
    from classifiers import Classifier
    counter = 0
    for i, classifier_file in enumerate(classifier_list):
        f = open(classifier_file, 'rb')
        try:
            w = pickle.load(f)
        except:
            counter += 1
            print("Error loading the file of the classifier {0}".format(os.path.basename(classifier_file)))
        f.close()
        sys.stdout.write("\r{0}>".format("Checked:%f%%" % ((float(i) / len(classifier_list)) * 100)))
        sys.stdout.flush()


####################################################
def check_pickable_files(config_files_folder):
    import glob
    import multiprocessing
    classifier_list = glob.glob(config_files_folder + "*.dat")
    import random
    random.shuffle(classifier_list)
    size = len(classifier_list) // multiprocessing.cpu_count()
    classifier_distributed_list = [classifier_list[i:i + size + 1] for i in range(0, len(classifier_list), size + 1)]
    for sub_list in classifier_distributed_list:
        p = multiprocessing.Process(
            target=check_pickable_process,
            args=[sub_list],
            ).start()


####################################################
class ClassifierModels():
    def __init__(self, set_name, classes_text, threshold, data_transformation):
        self.models = AutoVivification()

        self.models["Cutoff"] = dict(set=set_name,
                                     classes_names=classes_text,
                                     classifier_kind={"kind": "Cutoff"},
                                     name_to_show="Cutoff",
                                     thresholds=copy.deepcopy(threshold),
                                     data_transformation=None)

        self.models["Lasso"] = dict(set=set_name,
                                    classes_names=classes_text,
                                    classifier_kind={"kind": "Lasso"},
                                    name_to_show="Lasso",
                                    learning_algorithm={'parameters':
                                                            dict(alpha=0.1,
                                                                 tol=0.01),
                                                        },
                                    thresholds=copy.deepcopy(threshold),
                                    data_transformation=copy.deepcopy(data_transformation))

        self.models["ElasticNet"] = dict(set=set_name,
                                         classes_names=classes_text,
                                         classifier_kind={"kind": "ElasticNet"},
                                         name_to_show="ElasticNet",
                                         learning_algorithm={'parameters':
                                                                 dict(alpha=0.1,
                                                                      tol=0.01,
                                                                      l1_ratio=0.001),
                                                             },
                                         thresholds=copy.deepcopy(threshold),
                                         data_transformation=copy.deepcopy(data_transformation))

        self.models["Ridge"] = dict(set=set_name,
                                    classes_names=classes_text,
                                    classifier_kind={"kind": "Ridge"},
                                    name_to_show="Ridge",
                                    learning_algorithm={'parameters':
                                                            dict(alpha=0.1,
                                                                 tol=0.01,
                                                                 solver="auto"),
                                                        },
                                    thresholds=copy.deepcopy(threshold),
                                    data_transformation=copy.deepcopy(data_transformation))

        self.models["ARDRegression"] = dict(set=set_name,
                                            classes_names=classes_text,
                                            classifier_kind={"kind": "ARDRegression"},
                                            name_to_show="ARDRegression",
                                            learning_algorithm={'parameters':
                                                                    dict(n_iter=int(1e3),
                                                                         tol=0.01,
                                                                         alpha_1=0.001,
                                                                         alpha_2=0.001,
                                                                         lambda_1=0.001,
                                                                         lambda_2=0.001,
                                                                         )
                                            },
                                            thresholds=copy.deepcopy(threshold),
                                            data_transformation=copy.deepcopy(data_transformation))

        self.models["LM_LinearRegression"] = dict(set=set_name,
                                                  classes_names=classes_text,
                                                  classifier_kind={"kind": "LM_LinearRegression"},
                                                  name_to_show="LM_LinearRegression",
                                                  thresholds=copy.deepcopy(threshold),
                                                  data_transformation=copy.deepcopy(data_transformation))

        self.models["KNeighborsRegressor"] = \
            dict(set=set_name,
                 classes_names=classes_text,
                 classifier_kind={"kind": "KNeighborsRegressor"},
                 name_to_show="KNeighborsRegressor",
                 learning_algorithm={'parameters':
                                         dict(n_neighbors=5,
                                              weights="uniform",
                                              algorithm="auto",
                                              leaf_size=30,
                                              metric="minkowski",
                                              p=2)
                 },
                 thresholds=copy.deepcopy(threshold),
                 data_transformation=copy.deepcopy(data_transformation))

        self.models["KNeighborsClassifier"] = \
            dict(set=set_name,
                 classes_names=classes_text,
                 classifier_kind={"kind": "KNeighborsClassifier"},
                 name_to_show="KNeighborsClassifier",
                 learning_algorithm={'parameters':
                                         dict(n_neighbors=5,
                                              weights="uniform",
                                              algorithm="auto",
                                              leaf_size=30,
                                              metric="minkowski",
                                              p=2)
                 },
                 thresholds=copy.deepcopy(threshold),
                 data_transformation=copy.deepcopy(data_transformation))

        self.models["RadiusNeighborsClassifier"] = \
            dict(set=set_name,
                 classes_names=classes_text,
                 classifier_kind={"kind": "RadiusNeighborsClassifier"},
                 name_to_show="RadiusNeighborsClassifier",
                 learning_algorithm={'parameters':
                                         dict(radius=5,
                                              weights="uniform",
                                              algorithm="auto",
                                              leaf_size=30,
                                              metric="minkowski",
                                              p=2)
                 },
                 thresholds=copy.deepcopy(threshold),
                 data_transformation=copy.deepcopy(data_transformation))

        self.models["Gaussian"] = \
            dict(set=set_name,
                 classes_names=classes_text,
                 classifier_kind={"kind": "Gaussian"},
                 name_to_show="Gaussian",
                 learning_algorithm={'parameters':
                                         dict(regr="quadratic",
                                              corr="generalized_exponential",
                                              normalize=False,
                                              beta0=1e-1,
                                              theta0=1e-1,
                                              thetaL=1e-1,
                                              thetaU=1e-1
                                         )},
                 thresholds=copy.deepcopy(threshold),
                 data_transformation=copy.deepcopy(data_transformation))

        self.models["BayesianRidge"] = \
            dict(set=set_name,
                 classes_names=classes_text,
                 classifier_kind={"kind": "BayesianRidge"},
                 name_to_show="BR",
                 learning_algorithm={'parameters':
                                         dict(n_iter=300,
                                              tol=1.e-3,
                                              alpha_1=1e-06,
                                              alpha_2=1e-06,
                                              lambda_1=1e-06,
                                              lambda_2=1e-06,
                                              fit_intercept=True,
                                              normalize=False)},
                 thresholds=copy.deepcopy(threshold),
                 data_transformation=copy.deepcopy(data_transformation))

        self.models["DTR"] = \
            dict(set=set_name,
                 classes_names=classes_text,
                 classifier_kind={"kind": "DTR"},
                 name_to_show="DT",
                 learning_algorithm={'parameters':
                                         dict(criterion="mse",
                                              max_depth=None,
                                              min_samples_split=2,
                                              min_samples_leaf=1,
                                              max_features="auto")},
                 thresholds=copy.deepcopy(threshold),
                 data_transformation=copy.deepcopy(data_transformation))

        self.models["GaussianNB"] = \
            dict(set=set_name,
                 classes_names=classes_text,
                 classifier_kind={"kind": "GaussianNB"},
                 name_to_show="GaussianNB",
                 thresholds=copy.deepcopy(threshold),
                 data_transformation=copy.deepcopy(data_transformation))

        self.models["BernoulliNB"] = \
            dict(set=set_name,
                 classes_names=classes_text,
                 classifier_kind={"kind": "BernoulliNB"},
                 name_to_show="BernoulliNB",
                 thresholds=copy.deepcopy(threshold),
                 data_transformation=copy.deepcopy(data_transformation))

        self.models["MultinomialNB"] = \
            dict(set=set_name,
                 classes_names=classes_text,
                 classifier_kind={"kind": "MultinomialNB"},
                 name_to_show="MultinomialNB",
                 thresholds=copy.deepcopy(threshold),
                 data_transformation=copy.deepcopy(data_transformation))

        self.models["DTClassifier"] = \
            dict(set=set_name,
                 classes_names=classes_text,
                 classifier_kind={"kind": "DTClassifier"},
                 name_to_show="DT",
                 learning_algorithm={'parameters':
                                         dict(criterion="gini",
                                              max_depth=None,
                                              min_samples_split=2,
                                              min_samples_leaf=1,
                                              max_features="auto")},
                 thresholds=copy.deepcopy(threshold),
                 data_transformation=copy.deepcopy(data_transformation))

        self.models["ExtraTreesClassifier"] = \
            dict(set=set_name,
                 classes_names=classes_text,
                 classifier_kind={"kind": "ExtraTreesClassifier"},
                 name_to_show="ExtraTreesClassifier",
                 learning_algorithm={'parameters':
                                         dict(n_estimators=10,
                                              bootstrap=False,
                                              criterion="gini",
                                              max_depth=None,
                                              min_samples_split=2,
                                              min_samples_leaf=1,
                                              max_features="auto")},
                 thresholds=copy.deepcopy(threshold),
                 data_transformation=copy.deepcopy(data_transformation))

        self.models["ExtraTreesRegressor"] = copy.deepcopy(self.models["ExtraTreesClassifier"])
        self.models["ExtraTreesRegressor"]["classifier_kind"]["kind"] = "ExtraTreesRegressor"
        self.models["ExtraTreesRegressor"]["name_to_show"] = "ExtraTreesRegressor"

        self.models["RandomForestRegressor"] = copy.deepcopy(self.models["ExtraTreesClassifier"])
        self.models["RandomForestRegressor"]["classifier_kind"]["kind"] = "RandomForestRegressor"
        self.models["RandomForestRegressor"]["name_to_show"] = "RandomForestRegressor"
        self.models["RandomForestRegressor"]["learning_algorithm"]['parameters']["criterion"] = "mse"

        self.models["RandomForestClassifier"] = copy.deepcopy(self.models["ExtraTreesClassifier"])
        self.models["RandomForestClassifier"]["classifier_kind"]["kind"] = "RandomForestClassifier"
        self.models["RandomForestClassifier"]["name_to_show"] = "RandomForestClassifier"

        self.models["RandomTreesEmbedding"] = \
            dict(set=set_name,
                 classes_names=classes_text,
                 classifier_kind={"kind": "RandomTreesEmbedding"},
                 name_to_show="RandomTreesEmbedding",
                 learning_algorithm={'parameters':
                                         dict(n_estimators=10,
                                              max_depth=None,
                                              min_samples_split=2,
                                              min_samples_leaf=1,
                                              )},
                 thresholds=copy.deepcopy(threshold),
                 data_transformation=copy.deepcopy(data_transformation))

        self.models["GradientBoostingClassifier"] = \
            dict(set=set_name,
                 classes_names=classes_text,
                 classifier_kind={"kind": "GradientBoostingClassifier"},
                 name_to_show="GradientBoostingClassifier",
                 learning_algorithm={'parameters':
                                         dict(
                                             loss='deviance',
                                             learning_rate=0.1,
                                             n_estimators=100,
                                             subsample=1.0,
                                             min_samples_split=2,
                                             min_samples_leaf=1,
                                             max_depth=3,
                                             max_features=None
                                         )},
                 thresholds=copy.deepcopy(threshold),
                 data_transformation=copy.deepcopy(data_transformation))

        self.models["GradientBoostingRegressor"] = \
            dict(set=set_name,
                 classes_names=classes_text,
                 classifier_kind={"kind": "GradientBoostingRegressor"},
                 name_to_show="GradientBoostingRegressor",
                 learning_algorithm={'parameters':
                                         dict(
                                             loss='ls',
                                             learning_rate=0.1,
                                             n_estimators=100,
                                             subsample=1.0,
                                             min_samples_split=2,
                                             min_samples_leaf=1,
                                             max_depth=3,
                                             max_features=None
                                         )},
                 thresholds=copy.deepcopy(threshold),
                 data_transformation=copy.deepcopy(data_transformation))

        self.models["SVR"] = \
            dict(set=set_name,
                 classes_names=classes_text,
                 classifier_kind={"kind": "SVR"},
                 name_to_show="SVR",
                 learning_algorithm={"parameters":
                                         dict(C=1e8,
                                              kernel="rbf",
                                              epsilon=0.1,
                                              cache_size=200,
                                              class_weight=None,
                                              coef0=0.0,
                                              degree=3,
                                              gamma=0.001,
                                              max_iter=-1,
                                              probability=True,
                                              random_state=None,
                                              shrinking=True,
                                              tol=0.00001,
                                              verbose=False)},
                 thresholds=copy.deepcopy(threshold),
                 data_transformation=copy.deepcopy(data_transformation))

        self.models["SVM"] = copy.deepcopy(self.models["SVR"])
        self.models["SVM"]["classifier_kind"]["kind"] = "SVM"
        self.models["SVM"]["name_to_show"] = "SVM"

        self.models["NuSVC"] = copy.deepcopy(self.models["SVR"])
        self.models["NuSVC"]["classifier_kind"]["kind"] = "NuSVC"
        self.models["NuSVC"]["name_to_show"] = "NuSVC"
        self.models["NuSVC"]["learning_algorithm"] = {"parameters":
                                         dict(nu=5,
                                              kernel="rbf",
                                              epsilon=0.1,
                                              cache_size=200,
                                              class_weight=None,
                                              coef0=0.0,
                                              degree=3,
                                              gamma=0.001,
                                              max_iter=-1,
                                              probability=True,
                                              random_state=None,
                                              shrinking=True,
                                              tol=0.00001,
                                              verbose=False)}

        self.models["NN"] = \
            dict(set=copy.deepcopy(set_name),
                 classes_names=copy.deepcopy(classes_text),
                 classifier_kind={"kind": "NN", "transfer_function": "tanh"},
                 name_to_show="NN",
                 configuration={"neurons": [6]},
                 learning_algorithm=dict(kind="backpropagate",
                                         early_stopping=dict(activate=1, learning=0.5, validation=0.5),
                                         parameters=dict(learning_rate=0.01,
                                                         momentum=0.1,
                                                         epochs=500,
                                                         objective_error=0.00001,
                                                         penalty_term=0.0,
                                                         )),
                 thresholds=copy.deepcopy(threshold),
                 data_transformation=copy.deepcopy(data_transformation))

        self.models["MLP"] = \
            dict(set=copy.deepcopy(set_name),
                 classes_names=copy.deepcopy(classes_text),
                 classifier_kind={"kind": "MLP", "transfer_function": "TanSig"},
                 name_to_show="MLP",
                 configuration={"neurons": [6]},
                 learning_algorithm=dict(kind="train_rprop",
                                         parameters=dict(learning_rate=0.001,
                                                         momentum=0.01,
                                                         epochs=500,
                                                         objective_error=0.00001,
                                                         error_function="SSE")),
                 thresholds=copy.deepcopy(threshold),
                 data_transformation=copy.deepcopy(data_transformation))

        #########################################
    def build_classifiers_models(self, parameters, patterns, classifiers_kind, k_fold, features_names=None):
        if "Cutoff" in classifiers_kind:
            new_classifier = copy.deepcopy(self.models["Cutoff"])
            new_classifier["patterns"] = copy.deepcopy(patterns)
            for_name_list = [new_classifier["classifier_kind"]["kind"], k_fold]
            new_name = "_".join([str(x) for x in for_name_list])
            parameters["classifiers"][new_name] = new_classifier

        if "MLP" in classifiers_kind:
            for epochs in [10, 50, 100, 200, 300]:
                for neurons_1 in range(1, 3):
                    new_classifier = copy.deepcopy(self.models["MLP"])
                    new_classifier["patterns"] = copy.deepcopy(patterns)
                    new_classifier["learning_algorithm"]["parameters"]["epochs"] = epochs
                    for_name_list = [new_classifier["classifier_kind"]["kind"],
                                     k_fold,
                                     epochs,
                                     neurons_1]
                    new_name = "_".join([str(x) for x in for_name_list])
                    new_classifier["name_to_show"] = new_name
                    new_classifier["configuration"]["neurons"] = [neurons_1]
                    parameters["classifiers"][new_name] = new_classifier

        if "NN" in classifiers_kind:
            for epochs in [200, 500, 700]:
                for neurons_1 in range(3, 7):
                    new_classifier = copy.deepcopy(self.models["NN"])
                    new_classifier["patterns"] = copy.deepcopy(patterns)
                    new_classifier["learning_algorithm"]["parameters"]["epochs"] = epochs
                    for_name_list = [new_classifier["classifier_kind"]["kind"],
                                     k_fold,
                                     epochs,
                                     neurons_1]
                    new_name = "_".join([str(x) for x in for_name_list])
                    new_classifier["name_to_show"] = new_name
                    new_classifier["configuration"]["neurons"] = [neurons_1]
                    parameters["classifiers"][new_name] = new_classifier

        if sum([1 for x in classifiers_kind if "NB" in x]):
            if "BernoulliNB" in classifiers_kind:
                new_classifier = copy.deepcopy(self.models["BernoulliNB"])
                new_classifier["patterns"] = copy.deepcopy(patterns)
                for_name_list = [new_classifier["classifier_kind"]["kind"],
                                 k_fold]
                new_name = "_".join([str(x) for x in for_name_list])
                new_classifier["name_to_show"] = new_name
                parameters["classifiers"][new_name] = new_classifier

            if "MultinomialNB" in classifiers_kind:
                new_classifier = copy.deepcopy(self.models["MultinomialNB"])
                new_classifier["patterns"] = copy.deepcopy(patterns)
                for_name_list = [new_classifier["classifier_kind"]["kind"],
                                 k_fold]
                new_name = "_".join([str(x) for x in for_name_list])
                new_classifier["name_to_show"] = new_name
                new_classifier["data_transformation"] = dict(kind="MinMaxScaler", args=dict(feature_range=[0.0, 1.0]))
                parameters["classifiers"][new_name] = new_classifier

            if "GaussianNB" in classifiers_kind:
                new_classifier = copy.deepcopy(self.models["GaussianNB"])
                new_classifier["patterns"] = copy.deepcopy(patterns)
                for_name_list = [new_classifier["classifier_kind"]["kind"],
                                 k_fold]
                new_name = "_".join([str(x) for x in for_name_list])
                new_classifier["name_to_show"] = new_name
                parameters["classifiers"][new_name] = new_classifier

        if "SVR" in classifiers_kind:
            for C in [1e2, 1e3, 1e4]:
                for epsilon in [1e-4, 1e-2, 1e-1]:
                    for kernel in ['poly', 'rbf', 'linear']:
                        for degree in [3, 4, 5]:
                            for gamma in [1e-3, 1e-1, 1.e1]:
                                for tol in [1e-7]:
                                    if kernel != "rbf":
                                        for coef0 in [1e-1, 0.0, 1e1]:
                                            new_classifier = copy.deepcopy(self.models["SVR"])
                                            new_classifier["patterns"] = copy.deepcopy(patterns)

                                            new_classifier["learning_algorithm"]["parameters"]["coef0"] = coef0
                                            new_classifier["learning_algorithm"]["parameters"]["tol"] = tol
                                            new_classifier["learning_algorithm"]["parameters"]["gamma"] = gamma
                                            new_classifier["learning_algorithm"]["parameters"]["degree"] = degree
                                            new_classifier["learning_algorithm"]["parameters"]["kernel"] = kernel
                                            new_classifier["learning_algorithm"]["parameters"]["C"] = C
                                            new_classifier["learning_algorithm"]["parameters"]["epsilon"] = epsilon

                                            for_name_list = [new_classifier["classifier_kind"]["kind"],
                                                             k_fold,
                                                             C,
                                                             epsilon,
                                                             kernel,
                                                             coef0,
                                                             degree,
                                                             gamma,
                                                             tol]
                                            new_name = "_".join([str(x) for x in for_name_list])
                                            new_classifier["name_to_show"] = new_name
                                            parameters["classifiers"][new_name] = new_classifier
                                    else:

                                        new_classifier = copy.deepcopy(self.models["SVR"])
                                        new_classifier["patterns"] = copy.deepcopy(patterns)

                                        new_classifier["learning_algorithm"]["parameters"]["tol"] = tol
                                        new_classifier["learning_algorithm"]["parameters"]["gamma"] = gamma
                                        new_classifier["learning_algorithm"]["parameters"]["degree"] = degree
                                        new_classifier["learning_algorithm"]["parameters"]["kernel"] = kernel
                                        new_classifier["learning_algorithm"]["parameters"]["C"] = C
                                        new_classifier["learning_algorithm"]["parameters"]["epsilon"] = epsilon
                                        for_name_list = [new_classifier["classifier_kind"]["kind"],
                                                         k_fold,
                                                         C,
                                                         epsilon,
                                                         kernel,
                                                         degree,
                                                         gamma,
                                                         tol]
                                        new_name = "_".join([str(x) for x in for_name_list])
                                        new_classifier["name_to_show"] = new_name
                                        parameters["classifiers"][new_name] = new_classifier

        if "SVM" in classifiers_kind:
            for C in [1, 1e1, 1e2, 1e3, 1e4, 1e5]:
                for kernel in ['rbf', "linear"]:
                    for degree in [3, 4, 5, 6, 7, 9]:
                        for gamma in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.1, 10.]:
                            for max_iter in [5e6]:
                                if kernel != "rbf":
                                    for coef0 in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.1, 10.]:
                                        new_classifier = copy.deepcopy(self.models["SVM"])
                                        new_classifier["patterns"] = copy.deepcopy(patterns)

                                        new_classifier["learning_algorithm"]["parameters"]["coef0"] = coef0
                                        new_classifier["learning_algorithm"]["parameters"]["max_iter"] = max_iter
                                        new_classifier["learning_algorithm"]["parameters"]["gamma"] = gamma
                                        new_classifier["learning_algorithm"]["parameters"]["degree"] = degree
                                        new_classifier["learning_algorithm"]["parameters"]["kernel"] = kernel
                                        new_classifier["learning_algorithm"]["parameters"]["C"] = C

                                        for_name_list = [new_classifier["classifier_kind"]["kind"],
                                                         k_fold,
                                                         C,
                                                         kernel,
                                                         coef0,
                                                         degree,
                                                         gamma,
                                                         max_iter]
                                        new_name = "_".join([str(x) for x in for_name_list])
                                        new_classifier["name_to_show"] = new_name
                                        parameters["classifiers"][new_name] = new_classifier
                                else:
                                    new_classifier = copy.deepcopy(self.models["SVM"])
                                    new_classifier["patterns"] = copy.deepcopy(patterns)

                                    new_classifier["learning_algorithm"]["parameters"]["max_iter"] = max_iter
                                    new_classifier["learning_algorithm"]["parameters"]["gamma"] = gamma
                                    new_classifier["learning_algorithm"]["parameters"]["degree"] = degree
                                    new_classifier["learning_algorithm"]["parameters"]["kernel"] = kernel
                                    new_classifier["learning_algorithm"]["parameters"]["C"] = C
                                    for_name_list = [new_classifier["classifier_kind"]["kind"],
                                                     k_fold,
                                                     C,
                                                     kernel,
                                                     degree,
                                                     gamma,
                                                     max_iter]
                                    new_name = "_".join([str(x) for x in for_name_list])
                                    new_classifier["name_to_show"] = new_name
                                    parameters["classifiers"][new_name] = new_classifier

        if "NuSVC" in classifiers_kind:
            for nu in [1e-2, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
                for kernel in ['poly', 'rbf', "linear", "sigmoid"]:
                    for degree in [3, 4, 5, 6, 7, 9]:
                        for gamma in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 10.]:
                            for max_iter in [5e6]:
                                if kernel != "rbf":
                                    for coef0 in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 10.]:
                                        new_classifier = copy.deepcopy(self.models["NuSVC"])
                                        new_classifier["patterns"] = copy.deepcopy(patterns)

                                        new_classifier["learning_algorithm"]["parameters"]["coef0"] = coef0
                                        new_classifier["learning_algorithm"]["parameters"]["max_iter"] = max_iter
                                        new_classifier["learning_algorithm"]["parameters"]["gamma"] = gamma
                                        new_classifier["learning_algorithm"]["parameters"]["degree"] = degree
                                        new_classifier["learning_algorithm"]["parameters"]["kernel"] = kernel
                                        new_classifier["learning_algorithm"]["parameters"]["nu"] = nu

                                        for_name_list = [new_classifier["classifier_kind"]["kind"],
                                                         k_fold,
                                                         nu,
                                                         kernel,
                                                         coef0,
                                                         degree,
                                                         gamma,
                                                         max_iter]
                                        new_name = "_".join([str(x) for x in for_name_list])
                                        new_classifier["name_to_show"] = new_name
                                        parameters["classifiers"][new_name] = new_classifier
                                else:
                                    new_classifier = copy.deepcopy(self.models["NuSVC"])
                                    new_classifier["patterns"] = copy.deepcopy(patterns)

                                    new_classifier["learning_algorithm"]["parameters"]["max_iter"] = max_iter
                                    new_classifier["learning_algorithm"]["parameters"]["gamma"] = gamma
                                    new_classifier["learning_algorithm"]["parameters"]["degree"] = degree
                                    new_classifier["learning_algorithm"]["parameters"]["kernel"] = kernel
                                    new_classifier["learning_algorithm"]["parameters"]["nu"] = nu
                                    for_name_list = [new_classifier["classifier_kind"]["kind"],
                                                     k_fold,
                                                     nu,
                                                     kernel,
                                                     degree,
                                                     gamma,
                                                     max_iter]
                                    new_name = "_".join([str(x) for x in for_name_list])
                                    new_classifier["name_to_show"] = new_name
                                    parameters["classifiers"][new_name] = new_classifier

        if "DTR" in classifiers_kind:
            for max_depth in [None, 5, 10, 20]:
                for min_samples_split in [2, 5, 10, 20]:
                    for min_samples_leaf in [1, 3, 5, 10]:
                        new_classifier = copy.deepcopy(self.models["DTR"])
                        new_classifier["patterns"] = copy.deepcopy(patterns)

                        new_classifier["learning_algorithm"]["parameters"]["max_depth"] = max_depth
                        new_classifier["learning_algorithm"]["parameters"]["min_samples_split"] = min_samples_split
                        new_classifier["learning_algorithm"]["parameters"]["min_samples_leaf"] = min_samples_leaf

                        for_name_list = [new_classifier["classifier_kind"]["kind"],
                                         k_fold,
                                         max_depth,
                                         min_samples_split,
                                         min_samples_leaf]
                        new_name = "_".join([str(x) for x in for_name_list])

                        new_classifier["name_to_show"] = new_name
                        parameters["classifiers"][new_name] = new_classifier

        if "DTClassifier" in classifiers_kind:
            for criterion in ["gini", "entropy"]:
                for max_depth in [None, 5, 10, 20, 30]:
                    for min_samples_split in [2, 5, 10, 20, 30]:
                        for min_samples_leaf in [1, 3, 5, 10, 20]:
                            new_classifier = copy.deepcopy(self.models["DTClassifier"])
                            new_classifier["patterns"] = copy.deepcopy(patterns)

                            new_classifier["learning_algorithm"]["parameters"]["criterion"] = criterion
                            new_classifier["learning_algorithm"]["parameters"]["max_depth"] = max_depth
                            new_classifier["learning_algorithm"]["parameters"]["min_samples_split"] = min_samples_split
                            new_classifier["learning_algorithm"]["parameters"]["min_samples_leaf"] = min_samples_leaf

                            for_name_list = [new_classifier["classifier_kind"]["kind"],
                                             k_fold,
                                             criterion,
                                             max_depth,
                                             min_samples_split,
                                             min_samples_leaf]
                            new_name = "_".join([str(x) for x in for_name_list])

                            new_classifier["name_to_show"] = new_name
                            parameters["classifiers"][new_name] = new_classifier

        if "ExtraTreesClassifier" in classifiers_kind:
            for n_estimators in [10, 20, 30, 50]:
                for criterion in ["gini", "entropy"]:
                    for max_depth in [None, 5, 10, 20, 30]:
                        for min_samples_split in [10, 20, 30]:
                            for min_samples_leaf in [5, 10, 20]:
                                for bootstrap in [True, False]:
                                    new_classifier = copy.deepcopy(self.models["ExtraTreesClassifier"])
                                    new_classifier["patterns"] = copy.deepcopy(patterns)

                                    new_classifier["learning_algorithm"]["parameters"]["n_estimators"] = n_estimators
                                    new_classifier["learning_algorithm"]["parameters"]["bootstrap"] = bootstrap
                                    new_classifier["learning_algorithm"]["parameters"]["criterion"] = criterion
                                    new_classifier["learning_algorithm"]["parameters"]["max_depth"] = max_depth
                                    new_classifier["learning_algorithm"]["parameters"][
                                        "min_samples_split"] = min_samples_split
                                    new_classifier["learning_algorithm"]["parameters"][
                                        "min_samples_leaf"] = min_samples_leaf

                                    for_name_list = [new_classifier["classifier_kind"]["kind"],
                                                     k_fold,
                                                     n_estimators,
                                                     bootstrap,
                                                     criterion,
                                                     max_depth,
                                                     min_samples_split,
                                                     min_samples_leaf]
                                    new_name = "_".join([str(x) for x in for_name_list])

                                    new_classifier["name_to_show"] = new_name
                                    parameters["classifiers"][new_name] = new_classifier

        if "RandomTreesEmbedding" in classifiers_kind:
            for n_estimators in [10, 20, 30, 40, 50, 60]:
                for max_depth in [None, 5, 10, 20, 30]:
                    for min_samples_split in [10, 20, 30]:
                        for min_samples_leaf in [5, 10, 20]:
                            new_classifier = copy.deepcopy(self.models["RandomTreesEmbedding"])
                            new_classifier["patterns"] = copy.deepcopy(patterns)

                            new_classifier["learning_algorithm"]["parameters"]["n_estimators"] = n_estimators
                            new_classifier["learning_algorithm"]["parameters"]["max_depth"] = max_depth
                            new_classifier["learning_algorithm"]["parameters"]["min_samples_split"] = min_samples_split
                            new_classifier["learning_algorithm"]["parameters"]["min_samples_leaf"] = min_samples_leaf

                            for_name_list = [new_classifier["classifier_kind"]["kind"],
                                             k_fold,
                                             n_estimators,
                                             max_depth,
                                             min_samples_split,
                                             min_samples_leaf]
                            new_name = "_".join([str(x) for x in for_name_list])

                            new_classifier["name_to_show"] = new_name
                            parameters["classifiers"][new_name] = new_classifier

        if "RandomForestClassifier" in classifiers_kind:
            for n_estimators in [10, 20, 30]:
                # for criterion in ["gini", "entropy"]:
                for criterion in ["gini"]:
                    for max_depth in [None, 5, 10, 20]:
                        for min_samples_split in [10, 20, 30]:
                            for min_samples_leaf in [5, 10, 20]:
                                for bootstrap in [True]:
                                    new_classifier = copy.deepcopy(self.models["RandomForestClassifier"])
                                    new_classifier["patterns"] = copy.deepcopy(patterns)

                                    new_classifier["learning_algorithm"]["parameters"]["n_estimators"] = n_estimators
                                    new_classifier["learning_algorithm"]["parameters"]["bootstrap"] = bootstrap
                                    new_classifier["learning_algorithm"]["parameters"]["criterion"] = criterion
                                    new_classifier["learning_algorithm"]["parameters"]["max_depth"] = max_depth
                                    new_classifier["learning_algorithm"]["parameters"][
                                        "min_samples_split"] = min_samples_split
                                    new_classifier["learning_algorithm"]["parameters"][
                                        "min_samples_leaf"] = min_samples_leaf

                                    for_name_list = [new_classifier["classifier_kind"]["kind"],
                                                     k_fold,
                                                     n_estimators,
                                                     bootstrap,
                                                     criterion,
                                                     max_depth,
                                                     min_samples_split,
                                                     min_samples_leaf]
                                    new_name = "_".join([str(x) for x in for_name_list])

                                    new_classifier["name_to_show"] = new_name
                                    parameters["classifiers"][new_name] = new_classifier

        if "RandomForestRegressor" in classifiers_kind:
            for n_estimators in [10, 20, 30]:
                for loss_function in ["mse"]:
                    # for max_depth in [None]:
                    for max_depth in [None, 10, 20, 30]:
                        # for min_samples_split in [10]:
                        for min_samples_split in [10, 20, 30]:
                            # for min_samples_leaf in [5, 10]:
                            for min_samples_leaf in [5, 10, 20]:
                                # for bootstrap in [True, False]:
                                for bootstrap in [True]:
                                    new_classifier = copy.deepcopy(self.models["RandomForestRegressor"])
                                    new_classifier["patterns"] = copy.deepcopy(patterns)

                                    new_classifier["learning_algorithm"]["parameters"]["n_estimators"] = n_estimators
                                    new_classifier["learning_algorithm"]["parameters"]["bootstrap"] = bootstrap
                                    new_classifier["learning_algorithm"]["parameters"]["loss_function"] = loss_function
                                    new_classifier["learning_algorithm"]["parameters"]["max_depth"] = max_depth
                                    new_classifier["learning_algorithm"]["parameters"][
                                        "min_samples_split"] = min_samples_split
                                    new_classifier["learning_algorithm"]["parameters"][
                                        "min_samples_leaf"] = min_samples_leaf

                                    for_name_list = [new_classifier["classifier_kind"]["kind"],
                                                     k_fold,
                                                     n_estimators,
                                                     bootstrap,
                                                     loss_function,
                                                     max_depth,
                                                     min_samples_split,
                                                     min_samples_leaf]
                                    new_name = "_".join([str(x) for x in for_name_list])

                                    new_classifier["name_to_show"] = new_name
                                    parameters["classifiers"][new_name] = new_classifier

        if "ExtraTreesClassifier" in classifiers_kind:
            for n_estimators in [10, 20, 30, 50]:
                for criterion in ["gini", "entropy"]:
                    for max_depth in [None, 5, 10, 20, 30]:
                        for min_samples_split in [10, 20, 30]:
                            for min_samples_leaf in [5, 10, 20]:
                                for bootstrap in [True, False]:
                                    new_classifier = copy.deepcopy(self.models["ExtraTreesClassifier"])
                                    new_classifier["patterns"] = copy.deepcopy(patterns)

                                    new_classifier["learning_algorithm"]["parameters"]["n_estimators"] = n_estimators
                                    new_classifier["learning_algorithm"]["parameters"]["bootstrap"] = bootstrap
                                    new_classifier["learning_algorithm"]["parameters"]["criterion"] = criterion
                                    new_classifier["learning_algorithm"]["parameters"]["max_depth"] = max_depth
                                    new_classifier["learning_algorithm"]["parameters"][
                                        "min_samples_split"] = min_samples_split
                                    new_classifier["learning_algorithm"]["parameters"][
                                        "min_samples_leaf"] = min_samples_leaf

                                    for_name_list = [new_classifier["classifier_kind"]["kind"],
                                                     k_fold,
                                                     n_estimators,
                                                     bootstrap,
                                                     criterion,
                                                     max_depth,
                                                     min_samples_split,
                                                     min_samples_leaf]
                                    new_name = "_".join([str(x) for x in for_name_list])

                                    new_classifier["name_to_show"] = new_name
                                    parameters["classifiers"][new_name] = new_classifier

        if "GradientBoostingClassifier" in classifiers_kind:
            for learning_rate in [0.05, 0.75, 0.1]:
                for n_estimators in [10, 20, 30]:
                    for max_depth in [None, 5, 10, 20]:
                        for min_samples_split in [10, 20, 30]:
                            for min_samples_leaf in [5, 10, 20]:
                                for subsample in [0.1, 0.2, 0.3, 0.5]:
                                    new_classifier = copy.deepcopy(
                                        self.models["GradientBoostingClassifier"])
                                    new_classifier["patterns"] = copy.deepcopy(patterns)

                                    new_classifier["learning_algorithm"]["parameters"]["learning_rate"] = learning_rate
                                    new_classifier["learning_algorithm"]["parameters"]["n_estimators"] = n_estimators
                                    new_classifier["learning_algorithm"]["parameters"]["max_depth"] = max_depth
                                    new_classifier["learning_algorithm"]["parameters"][
                                        "min_samples_split"] = min_samples_split
                                    new_classifier["learning_algorithm"]["parameters"][
                                        "min_samples_leaf"] = min_samples_leaf
                                    new_classifier["learning_algorithm"]["parameters"]["subsample"] = subsample

                                    for_name_list = [new_classifier["classifier_kind"]["kind"],
                                                     k_fold,
                                                     learning_rate,
                                                     n_estimators,
                                                     max_depth,
                                                     min_samples_split,
                                                     min_samples_leaf,
                                                     subsample]
                                    new_name = "_".join([str(x) for x in for_name_list])

                                    new_classifier["name_to_show"] = new_name
                                    parameters["classifiers"][new_name] = new_classifier

        if "GradientBoostingRegressor" in classifiers_kind:
            for loss_function in ['ls', 'lad', 'huber', 'quantile']:
                for learning_rate in [0.1, 0.3, 0.5]:
                    for n_estimators in [10, 20, 30, 50]:
                        for max_depth in [None, 5, 10, 20, 30]:
                            for min_samples_split in [10, 20, 30]:
                                for min_samples_leaf in [5, 10, 20]:
                                    for subsample in [0.1, 0.5, 1.0]:
                                        new_classifier = copy.deepcopy(
                                            self.models["GradientBoostingRegressor"])
                                        new_classifier["patterns"] = copy.deepcopy(patterns)

                                        new_classifier["learning_algorithm"]["parameters"][
                                            "loss_function"] = loss_function
                                        new_classifier["learning_algorithm"]["parameters"][
                                            "learning_rate"] = learning_rate
                                        new_classifier["learning_algorithm"]["parameters"][
                                            "n_estimators"] = n_estimators
                                        new_classifier["learning_algorithm"]["parameters"]["max_depth"] = max_depth
                                        new_classifier["learning_algorithm"]["parameters"][
                                            "min_samples_split"] = min_samples_split
                                        new_classifier["learning_algorithm"]["parameters"][
                                            "min_samples_leaf"] = min_samples_leaf
                                        new_classifier["learning_algorithm"]["parameters"]["subsample"] = subsample

                                        for_name_list = [new_classifier["classifier_kind"]["kind"],
                                                         k_fold,
                                                         loss_function,
                                                         learning_rate,
                                                         n_estimators,
                                                         max_depth,
                                                         min_samples_split,
                                                         min_samples_leaf,
                                                         subsample]
                                        new_name = "_".join([str(x) for x in for_name_list])

                                        new_classifier["name_to_show"] = new_name
                                        parameters["classifiers"][new_name] = new_classifier

        if "BayesianRidge" in classifiers_kind:
            for n_iter in [100, 300, 500]:
                for alpha_1 in [1.e-9, 1.e-6, 1.e-3]:
                    for alpha_2 in [1.e-9, 1.e-6, 1.e-3]:
                        for lambda_1 in [1.e-9, 1.e-6, 1.e-3]:
                            for lambda_2 in [1.e-9, 1.e-6, 1.e-3]:
                                new_classifier = copy.deepcopy(self.models["BayesianRidge"])
                                new_classifier["patterns"] = copy.deepcopy(patterns)

                                new_classifier["learning_algorithm"]["parameters"]["n_iter"] = n_iter
                                new_classifier["learning_algorithm"]["parameters"]["alpha1"] = alpha_1
                                new_classifier["learning_algorithm"]["parameters"]["alpha2"] = alpha_2
                                new_classifier["learning_algorithm"]["parameters"]["lambda1"] = lambda_1
                                new_classifier["learning_algorithm"]["parameters"]["lambda2"] = lambda_2

                                for_name_list = [new_classifier["classifier_kind"]["kind"],
                                                 k_fold,
                                                 n_iter,
                                                 alpha_1,
                                                 alpha_2,
                                                 lambda_1,
                                                 lambda_2]
                                new_name = "_".join([str(x) for x in for_name_list])
                                new_classifier["name_to_show"] = new_name
                                parameters["classifiers"][new_name] = new_classifier

        if "Gaussian" in classifiers_kind:
            # for regr in ["constant", "linear", "quadratic"]:
            for regr in ["quadratic"]:
                # for corr in ["absolute_exponential", 'generalized_exponential', 'squared_exponential', 'cubic', 'linear']:
                for corr in ['linear']:
                    # for beta0 in [[1e-3], [1e-1], [1.e2]]:
                    for beta0 in [[1e-3]]:
                        for theta0 in [1e-3]:
                            for thetaL in [1e1]:
                                for thetaU in [1e2]:
                                    new_classifier = copy.deepcopy(self.models["Gaussian"])
                                    new_classifier["patterns"] = copy.deepcopy(patterns)

                                    new_classifier["learning_algorithm"]["parameters"]["thetaU"] = thetaU
                                    new_classifier["learning_algorithm"]["parameters"]["thetaL"] = thetaL
                                    new_classifier["learning_algorithm"]["parameters"]["theta0"] = theta0
                                    new_classifier["learning_algorithm"]["parameters"]["beta0"] = beta0
                                    new_classifier["learning_algorithm"]["parameters"]["corr"] = corr
                                    new_classifier["learning_algorithm"]["parameters"]["regr"] = regr

                                    for_name_list = [new_classifier["classifier_kind"]["kind"],
                                                     k_fold,
                                                     regr,
                                                     corr,
                                                     beta0,
                                                     theta0,
                                                     thetaL,
                                                     thetaU]
                                    new_name = "_".join([str(x) for x in for_name_list])
                                    new_classifier["name_to_show"] = new_name
                                    parameters["classifiers"][new_name] = new_classifier

        if "KNeighborsRegressor" in classifiers_kind:
            for n_neighbors in [5, 10, 15]:
                for weights in ["uniform", 'distance']:
                    for leaf_size in [20, 30, 40, 50, 60]:
                        for p in [1, 2, 3, 4, 5]:
                            for metric in ["minkowski"]:
                                new_classifier = copy.deepcopy(self.models["KNeighborsRegressor"])
                                new_classifier["patterns"] = copy.deepcopy(patterns)

                                new_classifier["learning_algorithm"]["parameters"]["metric"] = metric
                                new_classifier["learning_algorithm"]["parameters"]["p"] = p
                                new_classifier["learning_algorithm"]["parameters"]["leaf_size"] = leaf_size
                                new_classifier["learning_algorithm"]["parameters"]["weights"] = weights
                                new_classifier["learning_algorithm"]["parameters"]["n_neighbors"] = n_neighbors

                                for_name_list = [new_classifier["classifier_kind"]["kind"],
                                                 k_fold,
                                                 metric,
                                                 p,
                                                 leaf_size,
                                                 weights,
                                                 n_neighbors]
                                new_name = "_".join([str(x) for x in for_name_list])
                                new_classifier["name_to_show"] = new_name
                                parameters["classifiers"][new_name] = new_classifier

        if "LM_LinearRegression" in classifiers_kind:
            new_classifier = copy.deepcopy(self.models["LM_LinearRegression"])
            new_classifier["patterns"] = copy.deepcopy(patterns)

            for_name_list = [new_classifier["classifier_kind"]["kind"], k_fold]
            new_name = "_".join([str(x) for x in for_name_list])

            parameters["classifiers"][new_name] = new_classifier

        if "ARDRegression" in classifiers_kind:
            for n_iter in [1e2, 3e2]:
                for tol in [1e-1, 1e-2]:
                    for alpha_1 in [1e-5, 1e-6, 1e-7]:
                        for alpha_2 in [1e-5, 1e-6, 1e-7]:
                            for lambda_1 in [1e-5, 1e-6, 1e-7]:
                                for lambda_2 in [1e-5, 1e-6, 1e-7]:
                                    new_classifier = copy.deepcopy(self.models["ARDRegression"])
                                    new_classifier["patterns"] = copy.deepcopy(patterns)

                                    new_classifier["learning_algorithm"]["parameters"]["n_iter"] = int(n_iter)
                                    new_classifier["learning_algorithm"]["parameters"]["tol"] = tol
                                    new_classifier["learning_algorithm"]["parameters"]["alpha_1"] = alpha_1
                                    new_classifier["learning_algorithm"]["parameters"]["alpha_2"] = alpha_2
                                    new_classifier["learning_algorithm"]["parameters"]["lambda_1"] = lambda_1
                                    new_classifier["learning_algorithm"]["parameters"]["lambda_2"] = lambda_2

                                    for_name_list = [new_classifier["classifier_kind"]["kind"],
                                                     k_fold,
                                                     n_iter,
                                                     tol,
                                                     alpha_1,
                                                     alpha_2,
                                                     lambda_1,
                                                     lambda_2]
                                    new_name = "_".join([str(x) for x in for_name_list])
                                    parameters["classifiers"][new_name] = new_classifier

        if "Ridge" in classifiers_kind:
            for solver in ['auto', 'svd', 'dense_cholesky', 'lsqr', 'sparse_cg']:
                for tol in [1e-1, 1e-2]:
                    for alpha in [1e-5, 1e-6, 1e-7]:
                        new_classifier = copy.deepcopy(self.models["Ridge"])
                        new_classifier["patterns"] = copy.deepcopy(patterns)

                        new_classifier["learning_algorithm"]["parameters"]["solver"] = solver
                        new_classifier["learning_algorithm"]["parameters"]["tol"] = tol
                        new_classifier["learning_algorithm"]["parameters"]["alpha"] = alpha

                        for_name_list = [new_classifier["classifier_kind"]["kind"],
                                         k_fold,
                                         solver,
                                         tol,
                                         alpha]
                        new_name = "_".join([str(x) for x in for_name_list])
                        parameters["classifiers"][new_name] = new_classifier

        if "Lasso" in classifiers_kind:
            for tol in [1e-1, 1e-2]:
                for alpha in [1e-5, 1e-6, 1e-7]:
                    new_classifier = copy.deepcopy(self.models["Lasso"])
                    new_classifier["patterns"] = copy.deepcopy(patterns)

                    new_classifier["learning_algorithm"]["parameters"]["tol"] = tol
                    new_classifier["learning_algorithm"]["parameters"]["alpha"] = alpha

                    for_name_list = [new_classifier["classifier_kind"]["kind"],
                                     k_fold,
                                     tol,
                                     alpha
                                     ]
                    new_name = "_".join([str(x) for x in for_name_list])
                    parameters["classifiers"][new_name] = new_classifier

        if "ElasticNet" in classifiers_kind:
            for tol in [1e-1, 1e-2]:
                for alpha in [1e-5, 1e-6, 1e-7]:
                    for l1_ratio in [1e-5, 1e-6, 1e-7]:
                        new_classifier = copy.deepcopy(self.models["ElasticNet"])
                        new_classifier["patterns"] = copy.deepcopy(patterns)

                        new_classifier["learning_algorithm"]["parameters"]["tol"] = tol
                        new_classifier["learning_algorithm"]["parameters"]["alpha"] = alpha
                        new_classifier["learning_algorithm"]["parameters"]["l1_ratio"] = l1_ratio

                        for_name_list = [new_classifier["classifier_kind"]["kind"],
                                         k_fold,
                                         tol,
                                         alpha,
                                         l1_ratio
                                         ]
                        new_name = "_".join([str(x) for x in for_name_list])
                        parameters["classifiers"][new_name] = new_classifier

        if "KNeighborsClassifier" in classifiers_kind:
            for n_neighbors in [5, 10, 15]:
                for weights in ["uniform", 'distance']:
                    for leaf_size in [20, 30, 40, 50, 60]:
                        for p in [1, 2, 3, 4, 5]:
                            for metric in ["minkowski"]:
                                new_classifier = copy.deepcopy(self.models["KNeighborsClassifier"])
                                new_classifier["patterns"] = copy.deepcopy(patterns)

                                new_classifier["learning_algorithm"]["parameters"]["metric"] = metric
                                new_classifier["learning_algorithm"]["parameters"]["p"] = p
                                new_classifier["learning_algorithm"]["parameters"]["leaf_size"] = leaf_size
                                new_classifier["learning_algorithm"]["parameters"]["weights"] = weights
                                new_classifier["learning_algorithm"]["parameters"]["n_neighbors"] = n_neighbors

                                for_name_list = [new_classifier["classifier_kind"]["kind"],
                                                 k_fold,
                                                 n_neighbors,
                                                 weights,
                                                 leaf_size,
                                                 p,
                                                 metric]
                                new_name = "_".join([str(x) for x in for_name_list])
                                new_classifier["name_to_show"] = new_name
                                parameters["classifiers"][new_name] = new_classifier

        for classifier_name in parameters["classifiers"].keys():
            parameters["classifiers"][classifier_name]["features_names"] = features_names


#########################################

def delete_classifiers_not_selected(config_files_folder, results_file, results_folder, kfolds=None):
    import os
    import glob

    files = glob.glob(config_files_folder + "*.dat")
    if not isinstance(results_file, list):
        results_file = [results_file]

    #Acumulative variable
    lines = []
    for file_name in results_file:
        f = open(results_folder + file_name)

        f_lines = f.readlines()
        if isinstance(kfolds, int) and kfolds > 0:
            for line in f_lines:
                for fold in range(kfolds):
                    if line.count("_") > 1:
                        cl_parameters = line[line.find("_"):line.find(":")]
                        new_classifier_name = line[:line.find("_")+1] + str(fold) + cl_parameters + ".dat"
                    else:
                        new_classifier_name = line[:line.find(":")] + "_" + str(fold) + ".dat"
                    lines.append(new_classifier_name)
        elif kfolds is None:
            for line in f_lines:
                lines.append(line)
        f.close()

    count = 0
    for config_file_name in files:
        if os.path.basename(config_file_name) not in lines:
            count += 1
            # os.remove(config_file_name)
            pass

    print("Deleted {0} classifiers from a total of {1}".format(count, len(files)))

#########################################
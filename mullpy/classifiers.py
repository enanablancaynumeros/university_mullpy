# This Python file uses the following encoding: utf-8
# !/usr/local/bin/python3.3
####################################################
# This file is part of MULLPY.
#
#    MULLPY is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MULLPY is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with MULLPY.  If not, see <http://www.gnu.org/licenses/>.
####################################################
import copy
import pickle
import os

from mullpy.statistics import Statistics

import numpy as np

from mullpy.classifier_info import ClassifiersInfo
from mullpy.auxiliar import AutoVivification


####################################################


class Classifier:
    """
    Main class where are defined common functions within different kinds of classifiers.
    Every classifier inherit from this main class.
    It gives subclasses:
    a) A generic learning algorithm
    b) Some maths functions
    c) Build real outputs function
    """
    ####################################################
    def gaussian(self, base, width, input):
        """Return gaussian radial function.
            Args:
            radial: (num, num) of gaussian (base, width^2) pair
            input: input
            Returns:
            num of gaussian output
        """
        y = np.exp(-1 / width / 2 * np.power(input - base, 2))
        return y

    ####################################################
    def sign(self, x):
        if x < 0:
            return -1
        elif x > 0:
            return 1
        elif x == 0:
            return 0

    ####################################################
    def sigmoid(self, x):
        """Implementation of the sigmoid function"""
        p = 1.0
        return 1.0 / (1.0 + np.exp(-p * x))

    ####################################################
    def dsigmoid(self, x):
        """
        Sigmoid derivate function.
        """
        p = 1.0
        return p * (x - np.power(x, 2))

    ####################################################
    def tanh(self, y):
        """
        Tanh derivate function.
        """
        return np.tanh(y)

    ####################################################
    def dtanh(self, y):
        """
        Tanh derivate function.
        """
        return 1.0 - np.square(y)

    ####################################################
    def real_outputs(self, context, classifier_name, patterns):
        """
        Every kind of sub-classifier must have a real_outputs function which return a list of real outputs given
        by input pattern named predict.
        """
        classes_texts = context["classifiers"][classifier_name]["classes_names"]
        if "deployment" in context["execution_kind"]:
            # if context["execution_kind"] == "deployment_classification":
            len_inputs = len(patterns[0])
        else:
            len_inputs = len(patterns[0]) - len(classes_texts)
        return [self.predict(context, classifier_name, p[:len_inputs]) for p in patterns]

    ####################################################
    def core_learning(self, context, classifier_name, **kwargs):
        """
        Core learning. Proceed as a generic learning process for every classifier that has no itself core learning
        process.
        Parameters:
        a)context
        b)classifier name
        c) Kwargs:
            -) "ensemble_error": In the case of negative correlation learning (NCL) procedure,
            the ensemble call to each classifier to learn during 1 epoch.
            The NCL needs the ensemble output to be part of the learning process of each classifier.

        Core learning calls to the learning process of each classifier, controlling the number of epochs and the
        objective error determined. At this moment, the core learning process stop when the sum of the learning RMS
        and the test RMS rise the objective error. If the total error increases, the weights of the best error achieved
        is recorded, recovering it at the end of the process.

        If "plot_interactive" is marked to one, a graphic is plotted and saved into a file with the same name
        of the classifier.
        """
        statistics = Statistics()
        information = ClassifiersInfo()

        objective_error = context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["objective_error"]
        epochs = context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["epochs"]
        if epochs > 1:
            best_classifier = copy.deepcopy(context["classifiers"][classifier_name]["instance"])

        i = 0
        early_stopping = context["classifiers"][classifier_name]["learning_algorithm"]["early_stopping"]["activate"]
        if context["classifiers"][classifier_name]["learning_algorithm"]["early_stopping"]["activate"]:
            learning_percent = context["classifiers"][classifier_name]["learning_algorithm"]["early_stopping"][
                "learning"]
            validation_percent = context["classifiers"][classifier_name]["learning_algorithm"]["early_stopping"][
                "validation"]
            best_error = 10000000.
            iteration_best_error = 0
            validation_error = 10000000.

        if context["plot_training"]["activate"]:
            learning_accumulated_error = np.zeros(int(epochs / context["plot_training"]["times"]))
            validation_accumulated_error = np.zeros(int(epochs / context["plot_training"]["times"]))

        while validation_error >= objective_error and i < epochs:
            if "ensemble_error" in kwargs:
                learning_error = self.learning(context, classifier_name, ensemble_error=kwargs["ensemble_error"],
                                               epoch=i)
            if len(kwargs.keys()) and "ensemble_error" in kwargs and "classifier_error" in kwargs:
                learning_error = self.learning(context, classifier_name, ensemble_error=kwargs["ensemble_error"],
                                               epoch=i)
            else:
                learning_error = self.learning(context, classifier_name, epoch=i)

            if context["interactive"]["activate"] == 1 and i % context["interactive"]["epochs"] == 0 and i > 0:
                information.build_real_outputs(context, classifier_name, "learning")
                statistics.rms(classifier_name, context, information, "learning")
                learning_error = statistics.measures[classifier_name]["rms"]["learning"]
                information.build_real_outputs(context, classifier_name, "validation")
                statistics.rms(classifier_name, context, information, "validation")
                validation_error = statistics.measures[classifier_name]["rms"]["validation"]
                print("%s: epoch:%d, learning_error:%f, validation_error:%f" %
                      (classifier_name, i + 1, learning_error, validation_error))

            if context["plot_training"]["activate"] == 1 and i % context["plot_training"]["times"] == 0:
                if context["interactive"]["activate"] == 1 and i % context["interactive"]["epochs"] == 0:
                    learning_accumulated_error[int(i / context["plot_training"]["times"])] = learning_error
                    validation_accumulated_error[int(i / context["plot_training"]["times"])] = validation_error
                else:
                    information.build_real_outputs(context, classifier_name, "learning")
                    statistics.rms(classifier_name, context, information, "learning")
                    learning_error = statistics.measures[classifier_name]["rms"]["learning"]
                    information.build_real_outputs(context, classifier_name, "validation")
                    statistics.rms(classifier_name, context, information, "validation")
                    validation_error = statistics.measures[classifier_name]["rms"]["validation"]
                    learning_accumulated_error[int(i / context["plot_training"]["times"])] = learning_error
                    validation_accumulated_error[int(i / context["plot_training"]["times"])] = validation_error

            #To give also relevance to the learning error
            if early_stopping:
                stop_criteria_error = learning_percent * learning_error + validation_percent * validation_error

            if early_stopping and stop_criteria_error < best_error and context[
                "execution_kind"] == "learning" and epochs > 1:
                iteration_best_error = i
                best_error = stop_criteria_error
                best_classifier = copy.deepcopy(context["classifiers"][classifier_name]["instance"])

            i += 1

        #Return the value of the last epoch plus one
        if context["execution_kind"] != "NClearning":
            if context["interactive"]["activate"]:
                if early_stopping:
                    print("Best error of classifier {0}:{1} on epoch {2}".format(
                        classifier_name,
                        str(best_error),
                        str(iteration_best_error + 1)))
                    context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                        "epochs"] = iteration_best_error + 1
                    context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                        "objetive_error"] = best_error
                    context["classifiers"][classifier_name]["instance"] = best_classifier
                else:
                    information.build_real_outputs(context, classifier_name, "learning")
                    statistics.rms(classifier_name, context, information, "learning")
                    learning_error = statistics.measures[classifier_name]["rms"]["learning"]
                    information.build_real_outputs(context, classifier_name, "validation")
                    statistics.rms(classifier_name, context, information, "validation")
                    validation_error = statistics.measures[classifier_name]["rms"]["validation"]
                    print("Final error of classifier {0}: learning={1}, validation={2}".format(
                        classifier_name,
                        str(learning_error),
                        str(validation_error)))

        if context["plot_training"]["activate"]:
            from mullpy.presentations import Presentation

            Presentation().learning_graphic(
                context,
                classifier_name,
                learning_accumulated_error,
                validation_accumulated_error)

    ####################################################
    def save_config_file(self, context, classifier_name):
        """
        Save all data structure of a classifier
        """
        w = context["classifiers"][classifier_name]
        f = open(context["classifiers"][classifier_name]["config_file"], 'wb')
        pickle.dump(w, f, pickle.HIGHEST_PROTOCOL)
        # try:
        #     pickle.load(f)
        # except:
        #     print("Error saving the file of the classifier {0}. Not pickable".format(
        #         context["classifiers"][classifier_name]["config_file"]))
        f.close()

    ####################################################
    def load_config_file(self, context, classifier_name):
        """
        Recovery all the information from the file and initialize the attributes.
        """
        f = open(context["classifiers"][classifier_name]["config_file"], 'rb')
        try:
            w = pickle.load(f)
        except ValueError:
            import os

            print("Found a different pickle protocol and proceeding to remove the old/new by the present protocol.")
            os.remove(context["classifiers"][classifier_name]["config_file"])
            error = "Error loading the file of the classifier {0}. Just restart the process".format(classifier_name)
            raise NameError(error)
        except:
            raise NameError("Error loading the file of the classifier {0}".format(classifier_name))
        f.close()

        return w


####################################################


# class Neurolab(Classifier):
#     """
#     """
#
#
# ####################################################
#
#
# class Kohonen(Neurolab):
#     """
#
#     """
#
#     def func(self):
#         pass
#
# ####################################################
#
#
# class Elman(Neurolab):
#     """
#
#     """
#
#     def func(self):
#         pass
#
# ####################################################
#
#
# class Perceptron(Neurolab):
#     """
#
#     """
#
#     def func(self):
#         pass
#
# ####################################################
#
#
# class Lvq(Neurolab):
#     """
#
#     """
#
#     def func(self):
#         pass
#
# ####################################################
#
#
# class Hopfield(Neurolab):
#     """
#
#     """
#
#     def func(self):
#         pass
#
# ####################################################
#
#
# class Hemming(Neurolab):
#     """
#
#     """
#
#     def func(self):
#         pass

####################################################


# class MLP_(Neurolab):
#     """
#
#     """
#
#     def __init__(self, context, classifier_name):
#         import neurolab as nl
#
#         self.ni = len(context["patterns"].patterns[classifier_name][context["patterns_texts"][0]][0][0])
#         def_layers = context["classifiers"][classifier_name]["configuration"]["neurons"]
#         self.no = len(context["classifiers"][classifier_name]["classes_names"])
#
#         def_layers.insert(0, self.ni)
#         def_layers.append(self.no)
#
#         initialization_weight_values = [[-1, 1]] * self.ni
#         transfer_function_array = \
#             [getattr(nl.trans, context["classifiers"][classifier_name]["classifier_kind"]["transfer_function"])()] * \
#             len(def_layers)
#
#         layers = []
#         for i, nn in enumerate(def_layers):
#             layer_ci = def_layers[i - 1] if i > 0 else self.ni
#             l = nl.layer.Perceptron(layer_ci, nn, transfer_function_array[i])
#             l.initf = nl.init.initnw
#             layers.append(l)
#         connect = [[i - 1] for i in range(len(layers) + 1)]
#         from neurolab import core
#
#         self.classifier = core.Net(initialization_weight_values,
#                                    self.no,
#                                    layers,
#                                    connect,
#                                    context["classifiers"][classifier_name]["learning_algorithm"]["kind"],
#                                    context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
#                                        "error_function"])
#
#     ####################################################
#     def core_learning(self, context, classifier_name, **kwargs):
#         inputs = list(context["patterns"].patterns[classifier_name]["learning"][:, 0])
#         outputs = list(context["patterns"].patterns[classifier_name]["learning"][:, 1])
#         validation = context["patterns"].patterns[classifier_name]["validation"]
#         epochs = context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["epochs"]
#         goal = context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["objective_error"]
#         if context["interactive"]["activate"]:
#             show = context["interactive"]["epochs"]
#         else:
#             show = 0
#
#         self.classifier.train(inputs, outputs, epochs=epochs, show=show, goal=goal)
#
#     ####################################################
#     def predict(self, context, classifier_name, inputs):
#         return [y for x in self.classifier.sim([inputs]) for y in x]

####################################################


class MLP(Classifier):
    def __init__(self, context, classifier_name):
        import neurolab as nl

        self.no = len(context["classifiers"][classifier_name]["classes_names"])
        self.ni = len(context["patterns"].patterns[classifier_name][context["patterns_texts"][0]][0]) - self.no
        def_layers = context["classifiers"][classifier_name]["configuration"]["neurons"]

        def_layers.insert(0, self.ni)
        def_layers.append(self.no)

        initialization_weight_values = [[-1, 1]] * self.ni

        self.classifier = nl.net.newff(initialization_weight_values, def_layers)

    ####################################################
    def core_learning(self, context, classifier_name, **kwargs):
        if "features_names" in context["classifiers"][classifier_name]:
            len_inputs = len(context["classifiers"][classifier_name]["features_names"])
        else:
            len_inputs = self.ni

        inputs = context["patterns"].patterns[classifier_name]["learning"][:, range(len_inputs)]
        outputs = context["patterns"].patterns[classifier_name]["learning"][:, range(len_inputs, self.no + len_inputs)]
        # validation = context["patterns"].patterns[classifier_name]["validation"]
        epochs = context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["epochs"]
        goal = context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["objective_error"]
        if context["interactive"]["activate"]:
            show = context["interactive"]["epochs"]
        else:
            show = 0

        self.classifier.train(inputs, outputs, epochs=epochs, show=show, goal=goal)

    ####################################################
    def predict(self, context, classifier_name, inputs):
        return [y for x in self.classifier.sim([inputs]) for y in x]


####################################################


class Cutoff(Classifier):
    def __init__(self, context, classifier_name):
        pass

    def predict(self, context, classifier_name, inputs):
        if sum((inputs[3], inputs[5], inputs[6], inputs[7])) < 29:
            outputs = [0., 1.]
        else:
            outputs = [1., 0.]
        return outputs

    def core_learning(self, context, classifier_name, **kwargs):
        pass


####################################################


class Sklearn(Classifier):
    """
    Generic functions to use the learners from the scikit-learning library.
    """

    ####################################################

    def predict(self, context, classifier_name, inputs):
        """
        Every scikit-learning algorithm has its own predict function. This provide a general framework to use them
        through this main class.
        """
        outputs = [y for x in self.classifier.predict_proba([inputs]) for y in x]
        return outputs

    ####################################################

    def core_learning(self, context, classifier_name, **kwargs):
        """
        Every scikit-learning ML algorithm has it own classifier learning process, but many of them needs to
        convert the pattern structure to a contiguous float64 elements, like the SVM libray.
        """
        classes_len = len(context["classifiers"][classifier_name]["classes_names"])
        total_len = len(context["patterns"].patterns[classifier_name]["learning"][0])
        len_inputs = total_len - classes_len
        inputs = context["patterns"].patterns[classifier_name]["learning"][:, range(len_inputs)]
        outputs = context["patterns"].patterns[classifier_name]["learning"][:, range(len_inputs, total_len)]

        self.classifier.fit(inputs, outputs)


####################################################


class RandomForestRegressor(Sklearn):
    def __init__(self, context, classifier_name):
        from sklearn.ensemble import RandomForestRegressor

        self.classifier = RandomForestRegressor(
            n_estimators=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["n_estimators"],
            criterion=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["criterion"],
            max_depth=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["max_depth"],
            min_samples_split=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                "min_samples_split"],
            min_samples_leaf=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                "min_samples_leaf"],
            max_features=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["max_features"],
            bootstrap=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["bootstrap"],
            random_state=None,
            compute_importances=None)

    ####################################################

    def predict(self, context, classifier_name, inputs):
        print(inputs)
        outputs = [x for x in self.classifier.predict([inputs])]
        return outputs

    ####################################################

    def core_learning(self, context, classifier_name, **kwargs):
        classes_len = len(context["classifiers"][classifier_name]["classes_names"])
        total_len = len(context["patterns"].patterns[classifier_name]["learning"][0])
        len_inputs = total_len - classes_len
        inputs = context["patterns"].patterns[classifier_name]["learning"][:, range(len_inputs)]
        outputs = [x[0] for x in
                   context["patterns"].patterns[classifier_name]["learning"][:, range(len_inputs, total_len)]]

        self.classifier.fit(inputs, outputs)


####################################################


class ExtraTreesClassifier(Sklearn):
    def __init__(self, context, classifier_name):
        from sklearn.ensemble import ExtraTreesClassifier

        self.classifier = ExtraTreesClassifier(
            n_estimators=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["n_estimators"],
            criterion=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["criterion"],
            max_depth=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["max_depth"],
            min_samples_split=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                "min_samples_split"],
            min_samples_leaf=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                "min_samples_leaf"],
            max_features=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["max_features"],
            bootstrap=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["bootstrap"],
            random_state=None,
            compute_importances=None)

    ####################################################

    def predict(self, context, classifier_name, inputs):
        outputs = [y[1] for x in self.classifier.predict_proba([inputs]) for y in x]
        return outputs


####################################################


class ExtraTreesRegressor(Sklearn):
    def __init__(self, context, classifier_name):
        from sklearn.ensemble import ExtraTreesRegressor

        self.classifier = ExtraTreesRegressor(
            n_estimators=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["n_estimators"],
            criterion=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["criterion"],
            max_depth=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["max_depth"],
            min_samples_split=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                "min_samples_split"],
            min_samples_leaf=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                "min_samples_leaf"],
            max_features=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["max_features"],
            bootstrap=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["bootstrap"],
            random_state=None,
            compute_importances=None)

    ####################################################

    def predict(self, context, classifier_name, inputs):
        outputs = [y[1] for x in self.classifier.predict([inputs]) for y in x]
        return outputs

    ####################################################

    def core_learning(self, context, classifier_name, **kwargs):
        classes_len = len(context["classifiers"][classifier_name]["classes_names"])
        total_len = len(context["patterns"].patterns[classifier_name]["learning"][0])
        len_inputs = total_len - classes_len
        inputs = context["patterns"].patterns[classifier_name]["learning"][:, range(len_inputs)]
        outputs = [x[0] for x in
                   context["patterns"].patterns[classifier_name]["learning"][:, range(len_inputs, total_len)]]

        self.classifier.fit(inputs, outputs)


####################################################
class RandomTreesEmbedding(Sklearn):
    def __init__(self, context, classifier_name):
        from sklearn.ensemble import RandomTreesEmbedding

        self.classifier = RandomTreesEmbedding(
            n_estimators=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["n_estimators"],
            max_depth=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["max_depth"],
            min_samples_split=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                "min_samples_split"],
            min_samples_leaf=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                "min_samples_leaf"],
            random_state=None)


####################################################


class RandomForestClassifier(Sklearn):
    def __init__(self, context, classifier_name):
        from sklearn.ensemble import RandomForestClassifier

        self.classifier = RandomForestClassifier(
            n_estimators=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["n_estimators"],
            criterion=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["criterion"],
            max_depth=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["max_depth"],
            min_samples_split=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                "min_samples_split"],
            min_samples_leaf=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                "min_samples_leaf"],
            max_features=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["max_features"],
            bootstrap=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["bootstrap"],
            random_state=None,
            compute_importances=None)

    ############################
    def predict(self, context, classifier_name, inputs):
        outputs = [y[1] for x in self.classifier.predict_proba([inputs]) for y in x]
        return outputs


####################################################


class GradientBoostingClassifier(Sklearn):
    def __init__(self, context, classifier_name):
        from sklearn.ensemble import GradientBoostingClassifier

        self.classifier = GradientBoostingClassifier(
            loss=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["loss"],
            learning_rate=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["learning_rate"],
            n_estimators=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["n_estimators"],
            subsample=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["subsample"],
            min_samples_split=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                "min_samples_split"],
            min_samples_leaf=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                "min_samples_leaf"],
            max_depth=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["max_depth"],
            max_features=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["max_features"])

    ####################################################

    def core_learning(self, context, classifier_name, **kwargs):
        classes_len = len(context["classifiers"][classifier_name]["classes_names"])
        total_len = len(context["patterns"].patterns[classifier_name]["learning"][0])
        len_inputs = total_len - classes_len
        inputs = context["patterns"].patterns[classifier_name]["learning"][:, range(len_inputs)]
        outputs = [x[1] for x in
                   context["patterns"].patterns[classifier_name]["learning"][:, range(len_inputs, total_len)]]
        self.classifier.fit(inputs, outputs)

    ############################
    def predict(self, context, classifier_name, inputs):
        outputs = self.classifier.predict_proba([inputs])[0]
        return outputs


####################################################


class GradientBoostingRegressor(Sklearn):
    def __init__(self, context, classifier_name):
        from sklearn.ensemble import GradientBoostingRegressor

        self.classifier = GradientBoostingRegressor(
            loss=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["loss"],
            learning_rate=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["learning_rate"],
            n_estimators=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["n_estimators"],
            subsample=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["subsample"],
            min_samples_split=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                "min_samples_split"],
            min_samples_leaf=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                "min_samples_leaf"],
            max_depth=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["max_depth"],
            max_features=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["max_features"],
        )

    ####################################################

    def predict(self, context, classifier_name, inputs):
        outputs = [x for x in self.classifier.predict([inputs])]
        return outputs

    ####################################################

    def core_learning(self, context, classifier_name, **kwargs):
        classes_len = len(context["classifiers"][classifier_name]["classes_names"])
        total_len = len(context["patterns"].patterns[classifier_name]["learning"][0])
        len_inputs = total_len - classes_len
        inputs = context["patterns"].patterns[classifier_name]["learning"][:, range(len_inputs)]
        outputs = [x[0] for x in
                   context["patterns"].patterns[classifier_name]["learning"][:, range(len_inputs, total_len)]]
        self.classifier.fit(inputs, outputs)


####################################################


class RadiusNeighborsClassifier(Sklearn):
    def __init__(self, context, classifier_name):
        from sklearn.neighbors import RadiusNeighborsClassifier

        self.classifier = RadiusNeighborsClassifier(
            radius=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["radius"],
            weights=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["weights"],
            algorithm=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["algorithm"],
            leaf_size=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["leaf_size"],
            p=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["p"],
            metric=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["metric"])

    ####################################################

    def predict(self, context, classifier_name, inputs):
        outputs = [x[0] for x in self.classifier.predict([inputs])]
        return outputs


####################################################


class KNeighborsClassifier(Sklearn):
    def __init__(self, context, classifier_name):
        from sklearn.neighbors import KNeighborsClassifier

        self.classifier = KNeighborsClassifier(
            n_neighbors=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["n_neighbors"],
            weights=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["weights"],
            algorithm=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["algorithm"],
            leaf_size=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["leaf_size"],
            p=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["p"],
            metric=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["metric"])

    ####################################################

    def predict(self, context, classifier_name, inputs):
        return [x for x in self.classifier.predict([inputs])[0]]


####################################################


class KNeighborsRegressor(Sklearn):
    def __init__(self, context, classifier_name):
        from sklearn.neighbors import KNeighborsRegressor

        self.classifier = KNeighborsRegressor(
            n_neighbors=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["n_neighbors"],
            weights=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["weights"],
            algorithm=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["algorithm"],
            leaf_size=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["leaf_size"],
            p=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["p"],
            metric=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["metric"])

    ####################################################

    def predict(self, context, classifier_name, inputs):
        return [x[0] for x in self.classifier.predict([inputs])]


####################################################


class Gaussian(Sklearn):
    def __init__(self, context, classifier_name):
        from sklearn import gaussian_process

        self.classifier = gaussian_process.GaussianProcess(
            regr=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["regr"],
            corr=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["corr"],
            normalize=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["normalize"],
            beta0=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["beta0"],
            theta0=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["theta0"],
            thetaL=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["thetaL"],
            thetaU=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["thetaU"])

    ####################################################

    def predict(self, context, classifier_name, inputs):
        return [x for x in self.classifier.predict([inputs])[0]]


####################################################


class SVR(Sklearn):
    ####################################################
    def __init__(self, context, classifier_name):
        from sklearn.svm import SVR

        self.classifier = \
            SVR(
                C=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["C"],
                cache_size=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                    "cache_size"],
                coef0=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["coef0"],
                degree=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["degree"],
                gamma=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["gamma"],
                kernel=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["kernel"],
                max_iter=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["max_iter"],
                probability=True,
                shrinking=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["shrinking"],
                tol=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["tol"],
                verbose=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["verbose"]
            )

    ####################################################

    def predict(self, context, classifier_name, inputs):
        return [x for x in self.classifier.predict([inputs])]
        ####################################################

    def core_learning(self, context, classifier_name, **kwargs):
        classes_len = len(context["classifiers"][classifier_name]["classes_names"])
        total_len = len(context["patterns"].patterns[classifier_name]["learning"][0])
        len_inputs = total_len - classes_len
        inputs = context["patterns"].patterns[classifier_name]["learning"][:, range(len_inputs)]
        outputs = [x[0] for x in
                   context["patterns"].patterns[classifier_name]["learning"][:, range(len_inputs, total_len)]]

        self.classifier.fit(inputs, outputs)

        ####################################################


class NuSVC(Sklearn):
    ####################################################
    def __init__(self, context, classifier_name):
        from sklearn.svm import NuSVC

        self.classifier = \
            NuSVC(
                nu=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["nu"],
                cache_size=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                    "cache_size"],
                coef0=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["coef0"],
                degree=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["degree"],
                gamma=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["gamma"],
                kernel=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["kernel"],
                max_iter=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["max_iter"],
                probability=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                    "probability"],
                shrinking=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["shrinking"],
                tol=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["tol"],
                verbose=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["verbose"]
            )

    def core_learning(self, context, classifier_name, **kwargs):
        classes_len = len(context["classifiers"][classifier_name]["classes_names"])
        total_len = len(context["patterns"].patterns[classifier_name]["learning"][0])
        len_inputs = total_len - classes_len
        inputs = context["patterns"].patterns[classifier_name]["learning"][:, range(len_inputs)]
        outputs = [x[1] for x in
                   context["patterns"].patterns[classifier_name]["learning"][:, range(len_inputs, total_len)]]

        self.classifier.fit(inputs, outputs)

    ####################################################

    def predict(self, context, classifier_name, inputs):
        return [x for x in self.classifier.predict_proba([inputs])[0]]


####################################################


class SVM(Sklearn):
    ####################################################
    def __init__(self, context, classifier_name):
        from sklearn.svm import SVC

        self.classifier = \
            SVC(
                C=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["C"],
                cache_size=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                    "cache_size"],
                class_weight=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                    "class_weight"],
                coef0=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["coef0"],
                degree=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["degree"],
                gamma=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["gamma"],
                kernel=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["kernel"],
                max_iter=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["max_iter"],
                probability=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                    "probability"],
                shrinking=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["shrinking"],
                tol=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["tol"],
                verbose=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["verbose"]
            )

    def core_learning(self, context, classifier_name, **kwargs):
        classes_len = len(context["classifiers"][classifier_name]["classes_names"])
        total_len = len(context["patterns"].patterns[classifier_name]["learning"][0])
        len_inputs = total_len - classes_len
        inputs = context["patterns"].patterns[classifier_name]["learning"][:, range(len_inputs)]
        outputs = [x[1] for x in
                   context["patterns"].patterns[classifier_name]["learning"][:, range(len_inputs, total_len)]]

        self.classifier.fit(inputs, outputs)

    ####################################################

    def predict(self, context, classifier_name, inputs):
        return [x for x in self.classifier.predict_proba([inputs])[0]]


####################################################


class GaussianNB(Sklearn):
    def __init__(self, context, classifier_name):
        from sklearn import naive_bayes

        self.classifier = naive_bayes.GaussianNB()

    ####################################################

    def core_learning(self, context, classifier_name, **kwargs):
        classes_len = len(context["classifiers"][classifier_name]["classes_names"])
        total_len = len(context["patterns"].patterns[classifier_name]["learning"][0])
        len_inputs = total_len - classes_len
        inputs = context["patterns"].patterns[classifier_name]["learning"][:, range(len_inputs)]
        outputs = [x[1] for x in
                   context["patterns"].patterns[classifier_name]["learning"][:, range(len_inputs, total_len)]]

        self.classifier.fit(inputs, outputs)

    ####################################################

    def predict(self, context, classifier_name, inputs):
        outputs = [x for x in self.classifier.predict_proba([inputs])[0]]
        return outputs


####################################################


class BernoulliNB(Sklearn):
    def __init__(self, context, classifier_name):
        from sklearn import naive_bayes

        self.classifier = naive_bayes.BernoulliNB()

    ####################################################

    def core_learning(self, context, classifier_name, **kwargs):
        classes_len = len(context["classifiers"][classifier_name]["classes_names"])
        total_len = len(context["patterns"].patterns[classifier_name]["learning"][0])
        len_inputs = total_len - classes_len
        inputs = context["patterns"].patterns[classifier_name]["learning"][:, range(len_inputs)]
        outputs = [x[1] for x in
                   context["patterns"].patterns[classifier_name]["learning"][:, range(len_inputs, total_len)]]

        self.classifier.fit(inputs, outputs)

    ####################################################

    def predict(self, context, classifier_name, inputs):
        outputs = [x for x in self.classifier.predict_proba([inputs])[0]]
        return outputs


####################################################


class MultinomialNB(Sklearn):
    def __init__(self, context, classifier_name):
        from sklearn import naive_bayes

        self.classifier = naive_bayes.MultinomialNB()

    ####################################################

    def core_learning(self, context, classifier_name, **kwargs):
        classes_len = len(context["classifiers"][classifier_name]["classes_names"])
        total_len = len(context["patterns"].patterns[classifier_name]["learning"][0])
        len_inputs = total_len - classes_len
        inputs = context["patterns"].patterns[classifier_name]["learning"][:, range(len_inputs)]
        outputs = [x[1] for x in
                   context["patterns"].patterns[classifier_name]["learning"][:, range(len_inputs, total_len)]]

        self.classifier.fit(inputs, outputs)

    ####################################################

    def predict(self, context, classifier_name, inputs):
        return [x for x in self.classifier.predict_proba([inputs])[0]]


####################################################


class DTClassifier(Sklearn):
    """
    A decision tree classifier.
    """
    ####################################################
    def __init__(self, context, classifier_name):
        from sklearn import tree

        self.classifier = \
            tree.DecisionTreeClassifier(
                criterion=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["criterion"],
                splitter="best",
                max_depth=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["max_depth"],
                min_samples_split=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                    "min_samples_split"],
                min_samples_leaf=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                    "min_samples_leaf"],
                max_features=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                    "max_features"],
                random_state=None,
                compute_importances=None)

    def predict(self, context, classifier_name, inputs):
        return [y[1] for x in self.classifier.predict_proba([inputs]) for y in x]


#######################################################################


class ETClassifier(Sklearn):
    """An extremely randomized tree classifier.

    Extra-trees differ from classic decision trees in the way they are built.
    When looking for the best split to separate the samples of a node into two
    groups, random splits are drawn for each of the `max_features` randomly
    selected features and the best split among those is chosen. When
    `max_features` is set 1, this amounts to building a totally random
    decision tree.

    Warning: Extra-trees should only be used within ensemble methods.
    """

    def __init__(self, context, classifier_name):
        from sklearn import tree

        self.classifier = \
            tree.ExtraTreeClassifier(
                context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["criterion"],
                context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["max_depth"],
                context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["min_samples_split"],
                context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["min_samples_leaf"],
                context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["min_density"],
                context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["max_features"],
            )

    def core_learning(self, context, classifier_name, **kwargs):
        """
        The difference in GaussianNB is that it expects another output format for training
        """
        inputs = context["patterns"].patterns[classifier_name]["learning"][:, 0]
        new_inputs = np.ndarray(shape=(len(inputs), len(inputs[0])), dtype=np.float16, order='C')
        for i in range(len(inputs)):
            for j in range(len(inputs[i])):
                new_inputs[i][j] = inputs[i][j]
        outputs = [np.nonzero(x)[0][0] for x in context["patterns"].patterns[classifier_name]["learning"][:, 1]]
        # print(outputs)
        self.classifier.fit(new_inputs, outputs)
        ####################################################

    def predict(self, context, classifier_name, inputs):
        # print([y for x in self.classifier.predict_proba([inputs]) for y in x])
        # print(self.classifier.classes_)
        # print(self.classifier.n_features_)
        return [y for x in self.classifier.predict_proba([inputs]) for y in x]


#######################################################################


class DTR(Sklearn):
    def __init__(self, context, classifier_name):
        from sklearn import tree

        self.classifier = \
            tree.DecisionTreeRegressor(
                criterion=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["criterion"],
                max_depth=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["max_depth"],
                min_samples_split=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                    "min_samples_split"],
                min_samples_leaf=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                    "min_samples_leaf"],
                max_features=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["max_features"]
            )

    def core_learning(self, context, classifier_name, **kwargs):
        classes_len = len(context["classifiers"][classifier_name]["classes_names"])
        total_len = len(context["patterns"].patterns[classifier_name]["learning"][0])
        len_inputs = total_len - classes_len
        inputs = context["patterns"].patterns[classifier_name]["learning"][:, range(len_inputs)]
        outputs = [x[0] for x in
                   context["patterns"].patterns[classifier_name]["learning"][:, range(len_inputs, total_len)]]

        self.classifier.fit(inputs, outputs)

    def predict(self, context, classifier_name, inputs):
        output = [x for x in self.classifier.predict([inputs])]
        return output


#######################################################################


class ETRegressor(Sklearn):
    def __init__(self, context, classifier_name):
        from sklearn import tree

        self.classifier = \
            tree.ExtraTreeRegressor(
                context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["criterion"],
                context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["max_depth"],
                context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["min_samples_split"],
                context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["min_samples_leaf"],
                context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["min_density"],
                context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["max_features"],
            )

    def predict(self, context, classifier_name, inputs):
        return [x for x in self.classifier.predict([inputs])[0]]


#######################################################################


class LM_LinearRegression(Sklearn):
    def __init__(self, context, classifier_name):
        from sklearn import linear_model

        self.classifier = \
            linear_model.LinearRegression()

    ####################################################

    def predict(self, context, classifier_name, inputs):
        return [x for x in self.classifier.predict([inputs])[0]]


#######################################################################


class ARDRegression(Sklearn):
    def __init__(self, context, classifier_name):
        from sklearn.linear_model import ARDRegression

        self.classifier = \
            ARDRegression(
                n_iter=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["n_iter"],
                tol=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["tol"],
                alpha_1=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["alpha_1"],
                alpha_2=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["alpha_2"],
                lambda_1=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["lambda_1"],
                lambda_2=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["lambda_2"],
            )

    ####################################################

    def predict(self, context, classifier_name, inputs):
        return [x for x in self.classifier.predict([inputs])[0]]


#######################################################################


class BayesianRidge(Sklearn):
    def __init__(self, context, classifier_name):
        from sklearn.linear_model import BayesianRidge

        self.classifier = BayesianRidge(
            n_iter=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["n_iter"],
            tol=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["tol"],
            alpha_1=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["alpha1"],
            alpha_2=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["alpha2"],
            lambda_1=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["lambda1"],
            lambda_2=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["lambda2"],
            fit_intercept=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["fit_intercept"],
            normalize=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["normalize"],
        )

    ####################################################

    def core_learning(self, context, classifier_name, **kwargs):
        classes_len = len(context["classifiers"][classifier_name]["classes_names"])
        total_len = len(context["patterns"].patterns[classifier_name]["learning"][0])
        len_inputs = total_len - classes_len
        inputs = context["patterns"].patterns[classifier_name]["learning"][:, range(len_inputs)]
        outputs = [x[0] for x in
                   context["patterns"].patterns[classifier_name]["learning"][:, range(len_inputs, total_len)]]
        self.classifier.fit(inputs, outputs)

    ####################################################

    def predict(self, context, classifier_name, inputs):
        output = [x for x in self.classifier.predict([inputs])]
        return output


#######################################################################


class Ridge(Sklearn):
    def __init__(self, context, classifier_name):
        from sklearn.linear_model import Ridge

        self.classifier = \
            Ridge(alpha=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["alpha"],
                  copy_X=True,
                  solver=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["solver"],
                  tol=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["tol"]
                  )

    ####################################################

    def predict(self, context, classifier_name, inputs):
        return [x for x in self.classifier.predict([inputs])[0]]


#######################################################################


class Lasso(Sklearn):
    def __init__(self, context, classifier_name):
        from sklearn.linear_model import Lasso

        self.classifier = \
            Lasso(
                alpha=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["alpha"],
                copy_X=True,
                tol=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["tol"]
            )

    ####################################################
    def predict(self, context, classifier_name, inputs):
        return [x for x in self.classifier.predict([inputs])[0]]


#######################################################################


class ElasticNet(Sklearn):
    def __init__(self, context, classifier_name):
        from sklearn.linear_model import ElasticNet

        self.classifier = \
            ElasticNet(
                alpha=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["alpha"],
                l1_ratio=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["l1_ratio"],
                tol=context["classifiers"][classifier_name]["learning_algorithm"]["parameters"]["tol"]
            )

    ####################################################

    def predict(self, context, classifier_name, inputs):
        return [x for x in self.classifier.predict([inputs])[0]]


#######################################################################


class NN(Classifier):
    """
    Neural Network class.
    """
    ####################################################
    def __init__(self, context, classifier_name):
        """
        Construct a Neural network defined by context param or load it from a data file, giving:
            -Neurons as a set of number of hidden neurons
        """
        #Otherwise initialize with the manual values
        if type(context["classifiers"][classifier_name]["configuration"]["neurons"]) is not list:
            raise ValueError("Neurons must be a list in NN. Define the number of neurons for each layer")

        # +1 for bias node
        patterns = context["patterns"].patterns[classifier_name][context["patterns_texts"][0]]
        len_classes = len(context["classifiers"][classifier_name]["classes_names"])
        len_inputs = len(patterns[0]) - len_classes

        self.ni = len_inputs + 1
        self.no = len_classes
        if len(context["classifiers"][classifier_name]["configuration"]["neurons"]):
            self.layers = [self.ni]
            for element in context["classifiers"][classifier_name]["configuration"]["neurons"]:
                self.layers.append(element)
            self.layers.append(self.no)
        else:
            self.layers = [self.ni, self.no]

        # activations for nodes
        self.a = []
        for number_layer, elements in enumerate(self.layers):
            self.a.append(np.ones(elements, dtype="f"))

        self.w = []
        #One layer minus of weights than activations
        low_interval_initialization = -np.sqrt(6. / (self.ni + self.no))
        high_interval_initialization = np.sqrt(6. / (self.ni + self.no))

        for number_layer, elements in zip(range(len(self.layers) - 1), self.layers):
            self.w.append(np.random.uniform(low_interval_initialization,
                                            high_interval_initialization,
                                            (elements, self.layers[number_layer + 1])))

        if context["classifiers"][classifier_name]["learning_algorithm"]["kind"] == "backpropagate":
            # last change in weights for momentum
            self.c = []
            for number_layer, elements in zip(range(len(self.layers) - 1), self.layers):
                self.c.append(np.zeros((elements, self.layers[number_layer + 1]), dtype="f"))

    ####################################################

    def predict(self, context, classifier_name, inputs):
        """
        Return the activation output of the NN class.
        """
        if len(inputs) != self.ni - 1:
            raise NameError('wrong number of inputs')
            # input activations
        self.a[0][0:self.ni - 1] = inputs

        trans_function = context["classifiers"][classifier_name]["classifier_kind"]["transfer_function"]
        # hidden and output activations
        for number_layer in range(len(self.layers) - 1):
            res = np.dot(np.transpose(self.w[number_layer]), self.a[number_layer])
            elements = self.layers[number_layer + 1]
            self.a[number_layer + 1][0:elements] = getattr(self, trans_function)(res)[0:elements]

        return self.a[-1]

    ####################################################

    def backpropagate(self, context, classifier_name,
                      targets,
                      n,
                      m,
                      penalty_term=0.0, ensemble_evaluation=0.0, alpha=0.0):
        """
        Adjust weights matrix to the targets with momentum and Learning rate.
        """
        if len(targets) != self.no:
            raise NameError('wrong number of target values')

        # calculate error terms for output
        if context["execution_kind"] == "learning":
            # error_o = np.square(targets - self.a[-1])
            error_o = targets - self.a[-1]
        else:
            if context["execution_kind"] == "NClearning":
                error_o = targets - self.a[-1] - penalty_term * (ensemble_evaluation - self.a[-1])
            elif context["execution_kind"] == "RNClearning":
                error_o = targets - self.a[-1] - (ensemble_evaluation - self.a[-1]) + alpha
            else:
                raise ValueError("Error in learning process. Execution_kind is not well defined")

        errors = error_o
        transfer_function = context["classifiers"][classifier_name]["classifier_kind"]["transfer_function"]

        # calculate error terms for hidden
        for number_layer, elements in zip(reversed(range(len(self.layers) - 1)), reversed(self.layers)):
            delta = getattr(self, "d" + transfer_function)(self.a[number_layer + 1]) * errors

            change = delta * np.reshape(self.a[number_layer], (self.a[number_layer].shape[0], 1))

            self.w[number_layer] = self.w[number_layer] + n * change + m * self.c[number_layer]
            self.c[number_layer] = change
            if number_layer > 0:
                errors = np.dot(self.w[number_layer], delta)

        # calculate error
        return sum(0.5 * error_o ** 2)

    ####################################################

    def learning(self, context, classifier_name, **kwargs):
        """
        Make an epoch of the NN classifier
        """
        #Due to the Negative correlation learning
        if len(kwargs.keys()) < 2:
            kwargs = AutoVivification()
            kwargs["ensemble_error"] = []

        if type(kwargs["ensemble_error"]) != np.ndarray:
            kwargs["ensemble_error"] = np.array(context["patterns"].patterns[classifier_name]["learning"][:, 1])

        error = 0.0
        for i, pattern in enumerate(context["patterns"].patterns[classifier_name]["learning"]):
            inputs = pattern[:self.ni - 1]
            self.predict(context, classifier_name, inputs)
            targets = pattern[self.ni - 1:]

            error += self.learning_functions_scheduler(context, classifier_name, targets, kwargs["ensemble_error"][i])

        return error / float(len(context["patterns"].patterns[classifier_name]["learning"]))

    ####################################################

    def learning_functions_scheduler(self, context, classifier_name, targets, ensemble_evaluation):
        if context["classifiers"][classifier_name]["learning_algorithm"]["kind"] == "backpropagate":
            return self.backpropagate(context, classifier_name, targets,
                                      context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                                          "learning_rate"],
                                      context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                                          "momentum"],
                                      context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                                          "penalty_term"],
                                      # context["classifiers"][classifier_name]["learning_algorithm"]["parameters"][
                                      #     "alpha"],
                                      ensemble_evaluation)


#######################################################################
class Hybrid(Classifier):
    """
    An Hybrid classifier is comprised of a set of layers arranged sequentially.
    The first layer receive the inputs from the data
    The output will be given from the last layer.
    Each layer has to be a classifier defined individually.
    """

    def __init__(self, context, classifier_name):
        self.classifier = None

    ####################################

    def predict(self, context, classifier_name, inputs):
        outputs = None
        for i, layer in enumerate(context["classifiers"][classifier_name]["classifier_kind"]["Hybrid"]):
            if i > 0:
                outputs = context["classifiers"][layer]["instance"].predict(outputs)
            else:
                outputs = context["classifiers"][layer]["instance"].predict(inputs)
        return outputs

    ####################################

    def core_learning(self, context, classifier_name, **kwargs):
        for i, classifier_name in enumerate(context["classifiers"][classifier_name]["classifier_kind"]["Hybrid"]):
            context["classifiers"][classifier_name]["instance"].core_learning(context, classifier_name, **kwargs)


#######################################################################


class Grossberg(Classifier):
    def __init__(self, context, classifier_name):
        # create Grossberg output layer
        inputs = context["classifiers"][classifier_name]["configuration"]["neurons"][0]
        som_dimensions = list(context["classifiers"][classifier_name]["configuration"]["neurons"])
        outputs = len(context["classifiers"][classifier_name]["classes_names"])

        grossberg_activation_func = \
            context["classifiers"][classifier_name]["classifier_kind"]["transfer_function"]["grossberg"]

        #Insert the number of outputs at the beginning, each output neuron receives a signal from every SOM element
        som_dimensions.insert(0, outputs)

        self.w = np.random.uniform(-1, 1, tuple(som_dimensions))

    def predict(self, context, classifier_name, inputs):
        return [0.0] * len(context["classifiers"][classifier_name]["classes_names"])

    def learning(self, context, classifier_name):
        return 0.0


#######################################################################


class Classifier_from_file(Classifier):
    """
    A classifier defined by the outputs on a file
    """

    def __init__(self):
        """
        Init the structure for the outputs
        """
        self.output = AutoVivification()

    def load_config_file(self, context, classifier_name):
        if os.path.isfile(context["classifiers"][classifier_name]["config_file"]):
            f = open(context["classifiers"][classifier_name]["config_file"], "r")
            all_elements = f.readlines()
            for i in range(len(all_elements)):
                res = list(
                    map(float, all_elements[i].replace("\t", context["patterns_separator"]).replace("\n", "").split()))
                inputs = res[:len(res) - len(context["classifiers"][classifier_name]["classes_names"])]
                if context["classifiers"][classifier_name]["patterns"]["range"] == "[-1,1]":
                    outputs = [-1.0 if x == 0 else x for x in res[len(res) - len(
                        context["classifiers"][classifier_name]["classes_names"]):]]
                elif context["classifiers"][classifier_name]["patterns"]["range"] == "[0,1]":
                    outputs = [0.0 if x == -1.0 else x for x in res[len(res) - len(
                        context["classifiers"][classifier_name]["classes_names"]):]]
                self.output[str(inputs)] = outputs
                ####################################################

    def predict(self, inputs):
        return self.output[str(inputs)]

#######################################################################


#######################################################################
#######################################################################
#######################################################################

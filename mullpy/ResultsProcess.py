# This Python file uses the following encoding: utf-8
# !/usr/local/bin/python3.4
####################################################
# <Copyright (C) 2012, 2013, 2014 Yeray Alvarez Romero>
# This file is part of MULLPY.
####################################################
import itertools
import re
from mullpy.auxiliar import AutoVivification
import numpy as np
import glob


def read_file(name):
    f = open(name, "r")
    df = {}
    for linea in f.readlines():
        linea = linea.split()
        elem1 = linea[0][:-1]
        elem2 = float(linea[1])
        df[elem1] = elem2

    f.close()
    return df


def LuisFunc():
    # TODO: Ponerle a esto un mnombre apropiado
    folder_name = "/home/carmenpaz/Projects/SIAD/results/mci_sano_raw_k_fold_features_combination/"
    Data = {}
    # for k in ["DTClassifier", "ETClassifier", "DTRegressor", "ETRegressor"]:
    for k in ["DTClassifier", "DTRegressor"]:
        for min_density in ["0.1", "0.5", "1"]:
            df = read_file(folder_name + "validation_" + k + min_density + ".txt")
            for x in df:
                if x in Data:
                    Data[x]["count"] += 1
                    Data[x]["value"] += df[x]
                else:
                    Data[x] = {}
                    Data[x]["count"] = 1
                    Data[x]["value"] = df[x]

    f = open(folder_name + "Results.txt", "w")
    for comb in reversed([x for x in sorted(Data.keys(), key=lambda y: Data[y]["value"] / float(Data[y]["count"]))]):
        f.write(comb + ":\t")
        f.write("%d\t" % Data[comb]["count"])
        # print(Data[comb]["value"])
        valor = Data[comb]["value"] / float(Data[comb]["count"])
        f.write("%.3f\n" % valor)
    f.close()


def combined_features_to_file(file_name, structure):
    f = open(file_name, "w")
    for number in structure:
        f.write(str(number) + ": ")
        for feature in structure[number]:
            f.write(str(feature) + " ")
        f.write("\n")
    f.close()


def features_ranking(file_name, ranking):
    structure = structure_combined_features()

    f = open(file_name)
    lines = f.readlines()

    for line in lines:
        res = re.search(r'[0-9]+', line[:line.find("_")])
        for feature in structure[int(line[res.start():res.end()])]:
            ranking[feature].append(float(line[line.rfind("\t") + 1:]))

    for feature in ranking:
        ranking[feature] = np.mean(ranking[feature])

    for feature in sorted(ranking.keys(), key=lambda y: ranking[y], reverse=True):
        print(feature, ranking[feature])


def structure_combined_features():
    from auxiliar import AutoVivification

    structure = AutoVivification()
    i = 0
    for amount in range(2, 5 + 1):
        temporal = list(itertools.combinations(["AGE", "EDUC", "LIMMTOTAL", "FAQ", "MMSE", "GDS", "LDELTOTAL"], amount))
        for t in temporal:
            structure[i] = list(t)
            i += 1
    return structure


def features_from_combinations_to_keep(lista):
    structure = structure_combined_features()
    to_keep = []

    for combination in structure:
        if (len(set(lista) - set(structure[combination]))) == len(lista):
            if combination not in to_keep:
                to_keep.append(combination)
    return to_keep


def select_best_configuration_each_combination(in_file, out_file):
    # Sólo muestra la mejor configuración para cada combinación
    f = open(in_file)
    f2 = open(out_file, "w")
    lines = f.readlines()
    resultados = AutoVivification()
    for line in lines:
        resultados[line[:line.find(":")]] = line[line.find("\t") + 1:]

    temp = []
    for classifier_name in reversed([x for x in sorted(resultados.keys(), key=lambda y: resultados[y])]):
        res = re.search(r'[0-9]+', classifier_name[:classifier_name.find("_")])
        nombre = classifier_name[res.start():res.end()]
        if nombre not in temp:
            f2.write(classifier_name + ":\t")
            f2.write("%.4f\n" % (float(resultados[classifier_name])))
            temp.append(nombre)
    f.close()
    f2.close()


def concatenate_files(files, out_file):
    files = glob.glob(files)
    outfile = \
        open(out_file, 'wb')
    for file in files:
        with open(file, "rb") as infile:
            outfile.write(infile.read())


def select_classifier_names(in_file, out_file, lista):
    to_keep = features_from_combinations_to_keep(lista)
    f = open(in_file)
    f_out = open(out_file, "w")
    lines = f.readlines()
    for line in lines:
        classifier_name = line[:line.find(":")]
        res = re.search(r'[0-9]+', classifier_name[:classifier_name.find("_")])
        nombre = int(classifier_name[res.start():res.end()])
        if nombre in to_keep:
            for i in range(5):
                name = classifier_name[:res.start()] + str(i) + "_" + classifier_name[res.start():]
                f_out.write(name + "\n")


def select_classifiers_by_CV_accuracy(file, required_accuracy):
    classifiers_selected = []
    for line in open(file).readlines():
        accuracy = re.sub("\t", "", re.sub(" ", "", line[line.find(":") + 1:]))
        if float(accuracy) >= required_accuracy:
            # If classifier accuracy is better than indicated, the classifier is selected.
            classifiers_selected.append(line[:line.find(":")])
    return classifiers_selected


def separate_classifiers_by_fold(classifiers, lista, f_out_name):
    to_keep = features_from_combinations_to_keep(lista)
    f_out_list = []
    for i in range(5):
        f_out = open(f_out_name + str(i) + ".txt", "w")
        f_out_list.append(f_out)
    for classifier in classifiers:
        res = re.search(r'[0-9]+', classifier[:classifier.find("_")])
        nombre = int(classifier[res.start():res.end()])
        if nombre in to_keep:
            for i in range(5):
                name = classifier[:res.start()] + str(i) + "_" + classifier[res.start():]
                f_out_list[i].write(name + "\n")

# Given a text file with the following format:
#   ensemble_name_1: \t accuracy_1 \t accuracy_n \t diversity_1 \t diversity_n
#   ensemble_name_2: \t accuracy_1 \t accuracy_n \t diversity_1 \t diversity_n
#   ...
# , returns a dict with the sums of accuracy and diversity measures for each classifier.
####################################################################


def summary_stats_from_file(input_file):
    """
    :param input_file: The name of the file that has the information in the next manner:
    1) A set of measures names separated by tabulator
    2) A undefined number of lines with:
        classifier name: \t measure_value_0 \t measure_value_1 \t etc.
    :return: Summary structure which contains classifiers statistics. Each classifiers has a structure
    of measures names and its values associated
    """
    summary = {}
    measures_names = []
    for i, line in enumerate(open(input_file).readlines()):
        if i > 0:
            name = line[:line.find(":")]
            for measure_name, value in zip(measures_names, line[line.find("\t") + 1:].split("\t")):
                summary[name][measure_name] = value
        else:
            measures_names.append([x for x in line.split("\t") if len(x)])
    return summary


####################################################################


def ensembles_summary(input_file):
    f_in = open(input_file)
    in_lines = f_in.readlines()
    summary = {}
    for line in in_lines:
        name = line[:line.find(":")]
        remaining_line = line[line.find("\t") + 1:]
        sum = 0
        while remaining_line.find("\t") != -1:
            sum += float(remaining_line[:remaining_line.find("\t")])
            remaining_line = remaining_line[remaining_line.find("\t") + 1:]
            # We have to add the last element (accuracy or diversity value)
        sum += float(remaining_line)
        summary[name] = sum

    return summary

####################################################################


def ensembles_summary_cv_alternative(folder_path, member_range, fold_range):
    pass

####################################################################


def ensembles_summary_cv(folder_path, member_range, fold_range):
    #files_format = "%s/test_ensemble_*_*.txt" % folder_path
    #files_names = glob.glob(files_format)
    summary_total = []
    for fold in fold_range:
        for member in member_range:
            raw_file_name = "%s/test_ensemble_%d_%d.txt" % (folder_path, fold, member)
            summary_total.append(ensembles_summary(raw_file_name))

    # CV measures
    summary_total_dict = {}
    for summary in summary_total:
        for classifier_comb in summary.keys():
            summary_total_dict[classifier_comb] = summary[classifier_comb]
    cv_summary = {}
    for classifier_name in summary_total_dict.keys():
        remaining_classifier_name = classifier_name
        name = ""
        while remaining_classifier_name.find("+") != -1:
            name += remaining_classifier_name[:remaining_classifier_name.find("_")-1] +\
                      remaining_classifier_name[remaining_classifier_name.find("_")+1:
                      remaining_classifier_name.find("+")+1]
            remaining_classifier_name = remaining_classifier_name[remaining_classifier_name.find("+")+1:]
        # We add the last classifier of the specific combination to the ensemble name
        name += remaining_classifier_name[:remaining_classifier_name.find("_")-1] +\
                      remaining_classifier_name[remaining_classifier_name.find("_")+1:]
        ensemble_members = name.split("+")
        if name in cv_summary:
            cv_summary[name] += summary_total_dict[classifier_name]
        else:
            control = 0
            for element in cv_summary:
                if sorted(element.split("+")) == sorted(ensemble_members):
                    cv_summary[element] += summary_total_dict[classifier_name]
                    control = 1
            # Creation of the entry
            if control == 0:
                cv_summary[name] = summary_total_dict[classifier_name]
    # Calculating mean and exporting results into a file
    output_file = open(f_out, "w")
    for ensemble in reversed([x for x in sorted(cv_summary.keys(), key=lambda y: cv_summary[y])]):
        cv_summary[ensemble] /= len(fold_range)
        output_file.write("%s:\t%f\n" % (ensemble, cv_summary[ensemble]))


#####################################################################################


def create_file_given_classifiers_names(folder, output_file, classifier_list):
    import auxiliar
    f_out = auxiliar.create_output_file(folder, output_file, "append", "y")
    if not f_out:
        raise ValueError("File %s could not be created" % folder+output_file)
    for classifier_name in classifier_list:
        f_out.write(classifier_name + "\n")
    f_out.close()

#####################################################################################


def get_next_element(iterator_generated):
    try:
        return next(iterator_generated)
    except StopIteration:
        return None

#####################################################################################


def generate_ensembles_combination_from_classifier_file(folder, input_file, user_given_amounts):
    classifiers_file = open(folder + input_file)
    classifiers_selected = [name[:-1] for name in classifiers_file.readlines()]
    import mullpy

    for amount in user_given_amounts:
        classifier_list = mullpy.Process.automatic_ensemble_generation(classifiers_selected, [amount])
        for classifier_generator in classifier_list:
            # next_element = get_next_element(classifier_generator)
            for next_element in classifier_generator:
                if next_element is not None:
                    yield list(next_element)

#####################################################################################


def from_cross_validation_test_to_classifiers_selected(folder, features_to_delete, common_names, accuracy):
    classifiers_to_ensemble = []
    for name in common_names:
        f_cross_val = "%s_CrossValidation_Test.txt" % name
        f_cross_val_best_config = "%s_CrossVal_Best.txt" % name
        f_selected = "%s_Selected.txt" % name
        select_best_configuration_each_combination(folder + f_cross_val, folder + f_cross_val_best_config)

        select_classifier_names(folder + f_cross_val_best_config, folder + f_selected, features_to_delete)
        classifiers_to_ensemble.extend(
            select_classifiers_by_CV_accuracy("{0}{1}_CrossVal_Best.txt".format(folder, name), accuracy))

        f_out_name = folder + "classifiers_to_ensemble_fold"
        separate_classifiers_by_fold(classifiers_to_ensemble, features_to_delete, f_out_name)

#####################################################################################

if __name__ == '__main__':
    folder = "/home/yeray/Dropbox/SIAD/results/deployment/"
    # print(features_from_combinations_to_keep(["LIMMTOTAL", "LDELTOTAL", "GDS"]))
    # combined_features(2, 5, 7)
    # structure_combined_features()
    ranking = {"AGE": [], "EDUC": [], "LIMMTOTAL": [], "FAQ": [], "MMSE": [], "GDS": [], "LDELTOTAL": []}
    common_names = ["DT", "NB", "SVM"]
    features_to_delete = ["LIMMTOTAL", "LDELTOTAL", "GDS"]
    accuracy = 0.75
    from_cross_validation_test_to_classifiers_selected(folder, features_to_delete, common_names, accuracy)

    # ensembles_summary_cv("/home/aaron/Dropbox/SIAD/results/mci_sano_raw_k_fold_features_combination", range(2, 3),
    #                      range(2))
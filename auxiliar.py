# This Python file uses the following encoding: utf-8
# !/usr/local/bin/python3.3
####################################################
# <Copyright (C) 2012, 2013, 2014 Yeray Alvarez Romero>
# This file is part of MULLPY.
####################################################
import os
import errno
####################################################


def create_output_file(folder, output_file, operation, user_default_input):
    """
    :param folder:
    :param output_file:
    :param operation: "overwrite" or "append"
    :param user_default_input: "y" or "n". Leave it blank if you want to ask the user by console
    :return:
    """
    path_exists(folder)
    if not check_file(folder, output_file):
        return open(folder+output_file, "w")
    else:
        if operate_over_file(folder, output_file, operation, user_default_input):
            if operation == "overwrite":
                return open(folder+output_file, "w")
            elif operation == "append":
                return open(folder+output_file, "a")
            else:
                raise ValueError("Operation not recognized")
        else:
            return False
####################################################


def check_file(folder, output_file):
    if os.path.isfile(folder+output_file):
        return True
    else:
        return False
####################################################


def operate_over_file(folder, output_file, operation, user_default_input=None):
    if user_default_input:
        reply = user_default_input
    else:
        reply = input('¿Do you want to %s the file %s, y/n?' % (operation, folder+output_file))
    if reply == "y":
        return True
    else:
        return False
####################################################


def path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

####################################################


def summary_stats_from_file(folder, input_file):
    """
    :param folder: The name of the folder
    :param input_file: The name of the file that has the information in the next manner:
    1) A set of measures names separated by tabulator
    2) A undefined number of lines with:
        classifier name: \t measure_value_0 \t measure_value_1 \t etc.
    :return: Summary structure which contains classifiers statistics. Each classifiers has a structure
    of measures names and its values associated
    """
    summary = AutoVivification()
    if not check_file(folder, input_file):
        return summary
    for i, line in enumerate(open(folder+input_file).readlines()):
        if i > 0:
            name = line[:line.find(":")]
            values_list = []
            #Each measures values has to be separated by \t
            for x in sub(r" *\n", "", sub(r"  +", "", sub(r"\n", "", line[line.find(":")+1:]))).split("\t"):
                if x != "":
                    if len(x.split(',')) > 1:
                        values_list.append([float(y) for y in x.split(',')])
                    else:
                        values_list.append(float(x))

            #Assign each sublist or each value to the summary structure
            for measure_name, value in zip(measures_names, values_list):
                summary[name][measure_name] = value
        else:
            from re import sub
            measures_names = [x for x in sub(r" *\n", "", sub(r"  +", "", sub(r"\n", "", line))).split("\t") if len(x)]

    return summary
####################################################


def check_equal_classifier_patterns(context, classifier_name, classifier_name_2, pattern_kind):
    if classifier_name_2 != classifier_name and classifier_name_2 in context["classifiers"].keys():
        if pattern_kind in context["classifiers"][classifier_name_2]["patterns"] and \
            context["classifiers"][classifier_name]["patterns"][pattern_kind] == \
            context["classifiers"][classifier_name_2]["patterns"][pattern_kind] and \
                context["patterns"].patterns[classifier_name_2][pattern_kind] is not None and \
            context["classifiers"][classifier_name]["data_transformation"] == \
            context["classifiers"][classifier_name_2]["data_transformation"] and \
                "features_names" in context["classifiers"][classifier_name_2]:

            if "features_names" in context["classifiers"][classifier_name] and \
                context["classifiers"][classifier_name]["features_names"] == \
                    context["classifiers"][classifier_name_2]["features_names"]:
                return 1
            elif "features_names" not in context["classifiers"][classifier_name]:
                return 1
    return 0

####################################################


class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""

    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


####################################################
###############
# CSV to ARFF #
###############
#https://github.com/christinequintana/CSV-to-ARFF
import csv


class Csv2arff(object):

    content = []
    name = ''

    def __init__(self):
        self.csvInput()
        self.arffOutput()
        print('\nFinished.')

    #import CSV
    def csvInput(self):

        user = input('Enter the CSV file name: ')

        #remove .csv
        if user.endswith('.csv') == True:
            self.name = user.replace('.csv', '')

        print('Opening CSV file.')
        try:
            with open(user, 'rt') as csvfile:
               lines = csv.reader(csvfile, delimiter=',')
               for row in lines:
                   self.content.append(row)
            csvfile.close()

        #just in case user tries to open a file that doesn't exist
        except IOError:
            print('File not found.\n')
            self.csvInput()

    #export ARFF
    def arffOutput(self):
        print('Converting to ARFF file.\n')
        title = str(self.name) + '.arff'
        new_file = open(title, 'w')

        ##
        #following portions formats and writes to the new ARFF file
        ##

        #write relation
        new_file.write('@relation ' + str(self.name)+ '\n\n')

        #get attribute type input
        for i in range(len(self.content[0])-1):
            # attribute_type = input('Is the type of ' + str(self.content[0][i]) + ' numeric or nominal? ')
            # new_file.write('@attribute ' + str(self.content[0][i]) + ' ' + str(attribute_type) + '\n')
            new_file.write('@attribute ' + str(self.content[0][i]) + ' numeric\n')

        #create list for class attribute
        last = len(self.content[0])
        class_items = []
        for i in range(len(self.content)):
            name = self.content[i][last-1]
            if name not in class_items:
                class_items.append(self.content[i][last-1])
            else:
                pass
        del class_items[0]

        string = '{' + ','.join(sorted(class_items)) + '}'
        new_file.write('@attribute ' + str(self.content[0][last-1]) + ' ' + str(string) + '\n')

        #write data
        new_file.write('\n@data\n')

        del self.content[0]
        for row in self.content:
            new_file.write(','.join(row) + '\n')

        #close file
        new_file.close()
####################################################


def csv2pat(input_file, classes_len):
    import pandas as pd

    input_df = pd.read_csv(input_file)
    output_file = open("%s.pat" % input_file[:input_file.find('.csv')], "w")

    features = list(input_df.columns.values)
    # First, we build the beginning of the pat file, specifying feature names.
    for feature in features[:len(features)-classes_len]:
        output_file.write("@FEATURE %s\n" % feature)

    # Second, we add the content of the csv file, row by row
    for i in range(len(input_df)):
        # Building each row to be put in the output file.
        row = []
        for feature in features:
            row.append(input_df[feature][i])
        output_file.write("%s\n" % ",".join([str(x) for x in row]))

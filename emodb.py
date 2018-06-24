import os
import os.path
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioAnalysis as aA
import numpy as np

g_base_dir_delimiter = '/'
g_base_data_path = 'data'
g_base_model_path = 'data{}model'.format(g_base_dir_delimiter)

g_svm_model_path = g_base_model_path + "{}svm{}svmModel".format(g_base_dir_delimiter,g_base_dir_delimiter)
g_regression_model_path = g_base_model_path + "{}regression{}regressionModel".format(g_base_dir_delimiter,g_base_dir_delimiter)

def gen_classifier_model():
    base_path = g_base_data_path +'{}emodb_svm_train'.format(g_base_dir_delimiter)
    tmpSubdirectories = os.listdir(base_path)
    subdirectories =[]
    for subDir in  tmpSubdirectories:
        if os.path.isdir(base_path + g_base_dir_delimiter + subDir) :
            subdirectories.append(base_path + g_base_dir_delimiter + subDir)

    print(subdirectories)
    aT.featureAndTrain(subdirectories, 1.0, 1.0, aT.shortTermWindow,
                       aT.shortTermStep, "svm", g_svm_model_path, False)

def test_classifier_model():
    isSignificant = 0.8  # try different values.

    base_path = g_base_data_path +'{}emodb_svm_test'.format(g_base_dir_delimiter)
    sub_paths= os.listdir(base_path)

    for folder in sub_paths:

        base_folder = "{}{}{}".format(base_path, g_base_dir_delimiter,folder)

        if not os.path.isdir(base_folder):
            continue

        print ""
        print ""
        print "[" + folder + "]"
        print ""
        fileEntry = os.listdir("{}{}{}".format(base_path, g_base_dir_delimiter,folder))


        for file in fileEntry:
            print file
            if file.endswith("wav"):
                file_path = "{}{}{}".format(base_folder ,g_base_dir_delimiter,  file)
                Result, P, classNames = aT.fileClassification(file_path, g_svm_model_path, "svm")
                winner = np.argmax(P)

                if P[winner] > isSignificant:
                    print("File: " + file_path + " is in category: " + classNames[
                        winner] + ", with probability: " + str(P[winner]))
                else:
                    print("Can't classify sound: " + str(P))


def gen_regression():

    base_path = g_base_data_path +'{}emodb_regression_train'.format(g_base_dir_delimiter)
    aT.featureAndTrainRegression(base_path, 1, 1, aT.shortTermWindow,
                                 aT.shortTermStep, "svm",g_regression_model_path, False)



def test_regression():
    base_path = g_base_data_path + "{}emodb_regression_test".format(g_base_dir_delimiter)

    dirs = os.listdir(base_path)

    for dir  in dirs:
        each_dir = "{}{}{}".format(base_path, g_base_dir_delimiter, dir)
        if os.path.isdir(each_dir):
            aA.regressionFolderWrapper(each_dir, "svm", g_regression_model_path)

            files = os.listdir(each_dir)
            print ""
            print ""
            print "[" + dir + "]"
            print ""
            for file in files:
                if file.endswith("wav"):
                    print file
                    file_path = "{}{}{}".format(each_dir, g_base_dir_delimiter, file)
                    aA.regressionFileWrapper(file_path, "svm", g_regression_model_path)


            print ""
            print ""

if __name__ == '__main__':
    #gen_classifier_model()
    #test_classifier_model()
    #gen_regression()
    test_regression()

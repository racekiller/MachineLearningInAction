from numpy import *
import numpy as np
import operator
from os import listdir


def createDataSet():
    group = array(
        [
            [1.0, 1.1],
            [1.0, 1.0],
            [0, 0],
            [0, 0.1]
        ])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # Distance calculation
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # Voting with lowest k distances
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # Sort dictionary
    sortedClassCount = sorted(iter(classCount.items()),
    key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    # This function takes a filename string and outputs two things: a matrix
    # of training examples and a vector of class labels.
    fr = open(filename)
    # Get number of line in the dataset
    numberOfLines = len(fr.readlines())
    # Create Numpy matrix to return
    # adding matrix with zeroes
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        # Lets get the labels in string format as a vector
        classLabelVector.append(listFromLine[-1])
        index += 1
    # Here we get the integer version of each label using np.unique function
    classLabel, classLabelArrayInt = np.unique(classLabelVector,
        return_inverse=True)
    # When getting the unique value for each lable it starts in zer0, it wont
    # work when calculating values for colors, size because is getting
    # multiplied by zero therefore it gets deleted and it wont show in the
    # plot, so we need to add 1 to the result
    classLabelArrayInt = classLabelArrayInt + 1
    # Here we convert the vector classLabelVector to an Array
    classLabelVectorArray = np.asarray(classLabelVector)
    return returnMat, classLabelVectorArray, classLabelArrayInt


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels, datingLabelsInt = file2matrix(
        '../SourceCode2/Ch02/datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.1
    for i in range(numTestVecs):
        print ("Loop: %d" % i)
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
            datingLabelsInt[numTestVecs:m], 3)
        print (("The classifier cam back with: %d, the real answer is: %d") \
                % (classifierResult, datingLabelsInt[i]))
        if (classifierResult != datingLabelsInt[i]): errorCount += 1.0
    print (("the total error rate is: %f") % (errorCount / float(numTestVecs)))


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input(
        "percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels, datingLabelsInt = file2matrix(
        '../SourceCode2/Ch02/datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArray = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArray - minVals) / ranges, normMat,
                        datingLabelsInt, 3)
    print ("You will probable like this person: ",
            resultList[classifierResult - 1])


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        linesStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(linesStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    fileTrainingPath = '../SourceCode2/Ch02/digits/trainingDigits'
    fileTestPath = '../SourceCode2/Ch02/digits/testDigits'
    trainingFileList = listdir(fileTrainingPath)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector(fileTrainingPath + '/%s' % fileNameStr)
    testFileList = listdir(fileTestPath)
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(fileTestPath + '/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print ('the classifier came back with: %d, the real answer is: %d'
                % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print('\nthe total number or errors is: %d' % errorCount)
    print('\nthe total error rate is: %f' % (errorCount / float(mTest)))

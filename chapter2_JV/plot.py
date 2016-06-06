from numpy import *
import KNN
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
datingDataMat,datingLabels,datingLabelsInt = KNN.file2matrix(
    '../SourceCode2/Ch02/datingTestSet.txt')
ax.scatter(datingDataMat[:,0], datingDataMat[:,1], s=15.0*datingLabelsInt
    ,marker='^', c=datingLabelsInt)
# ax.axis([-2, 60000, -0.2, 16])
plt.xlabel('Percentage of Time Spent Playing Video Games')
plt.ylabel('Liters of Ice Cream Consumed Per Week')
plt.legend()
plt.show()

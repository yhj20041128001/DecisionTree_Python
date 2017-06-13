# -*- coding: UTF-8 -*-  
'''
Created on 2017��6��12��
Yang Hong Jun
@author: Administrator
'''

from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import tree
from sklearn import preprocessing
from sklearn.externals.six import  StringIO


# Read in the csv file and put features into list of dict and list of class label

allElectronicsData = open(r'C:\Python35\workspace\DeepLearningBasicsMachineLearning\DecisionTree\AllElectronics.csv','rb')
reader = csv.reader(allElectronicsData)
headers = reader.next()

print(headers)

featureList = []
labelList = []

for row in reader:  # 每一个行
    labelList.append(row[len(row)-1])  #每一行最后一个值
    rowDict = {}
    for i in range(1,len(row)-1):

        rowDict[headers[i]] = row[i]

    featureList.append(rowDict)
    
print(featureList)
# print(len(row))
# print(row[5])
# print(row)
# print(rowDict)
# print(labelList)
# print(headers[len(row)-1])

# Vetorize.features
vec = DictVectorizer()
dummyX  = vec.fit_transform(featureList).toarray()

print("dummyX" + str(dummyX))
print(vec.get_feature_names())


print("labelList: " +str(labelList))

# # vectorize class labels
lb=  preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY: " +str(dummyY) )

#Using decision tree for classification
#clf = tree.DecisionTreeClassifier()
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX,dummyY)
print("clf: " + str(clf))


#Visualize model

with open("decisontree.dot",'w') as f:
   f = tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=f)

oneRowX = dummyX[0,:]
print("oneRowX: " + str(oneRowX))

newRowX = oneRowX
newRowX[0]= 1
newRowX[2]= 0
print("newRoxX: "  + str(newRowX))

predictedY = clf.predict(newRowX)
print("predictedY:" + str(predictedY))

#[ 0.  0.  1.  0.  1.  1.  0.  0.  1.  0.]
#[ 1.  0.  0.  0.  1.  1.  0.  0.  1.  0.]













#
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
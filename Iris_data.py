from sklearn import datasets
iris = datasets.load_iris()

'''
print(len(iris.data))
print(len(iris.target))

print(iris.data[0])
#Sepal length,Sepal width,Petal length,Petal width
print(set(iris.target))
#0-setosa,1-virginica,2-versicolor
'''

return iris

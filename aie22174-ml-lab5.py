import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import openpyxl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

#Q3
def classify_data(X_values, Y_values):
    class_True = []
    class_False = []
    
    for i in range(len(X_values)):
        if X_values[i] + Y_values[i] > 12:
            class_True.append((X_values[i], Y_values[i]))
        else:
            class_False.append((X_values[i], Y_values[i]))
    
    return class_True, class_False

def plot_data(class_True, class_False):
    class_True_X, class_True_Y = zip(*class_True)
    class_False_X, class_False_Y = zip(*class_False)
    
    plt.scatter(class_True_X, class_True_Y, color='blue', label='Class 0')
    plt.scatter(class_False_X, class_False_Y, color='red', label='Class 1')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Training Data')
    plt.legend()
    plt.show()


data = np.random.randint(1, 11, size=(20, 2))
X_values = data[:, 0] 
Y_values = data[:, 1]

# Classify data
class_True, class_False = classify_data(X_values, Y_values)

# Plot data
plot_data(class_True, class_False)



#Q4 and Q5
def generate_training_data():
    np.random.seed(0)
    X_train = np.random.rand(300, 2) * 10
    y_train = (X_train[:, 0] + X_train[:, 1] > 10).astype(int)
    return X_train, y_train

def classify_test_points(X_train, y_train,k):
    x_values = np.arange(0, 10.1, 0.1)
    y_values = np.arange(0, 10.1, 0.1)
    xx, yy = np.meshgrid(x_values, y_values)
    test_data = np.column_stack((xx.ravel(), yy.ravel()))

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    predicted_labels = knn.predict(test_data)
    
    return xx, yy, predicted_labels

def plot_results(xx, yy, predicted_labels, X_train, y_train,k):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

    plt.figure(figsize=(10, 8))
    plt.scatter(xx.ravel(), yy.ravel(), c=predicted_labels, cmap=cmap_light)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    t="kNN Classification k= "+str(k)
    plt.title(t)
    plt.xlabel("X")
    plt.ylabel("Y")

    # Adding colorbar
    cb = plt.colorbar()
    loc = np.arange(0, max(y_train), max(y_train)/float(len(np.unique(y_train))))
    cb.set_ticks(loc)
    cb.set_ticklabels(np.unique(y_train))

    plt.show()

# Generate training data
X_train, y_train = generate_training_data()
for k in [1,3,5,7]:
    xx, yy, predicted_labels = classify_test_points(X_train, y_train,k)
    plot_results(xx, yy, predicted_labels, X_train, y_train,k)




def generate_training_data():
    np.random.seed(0)
    X_train = np.random.rand(300, 2) * 10
    y_train = (X_train[:, 0] + X_train[:, 1] > 10).astype(int)
    return X_train, y_train

def classify_test_points(X_train, y_train):
    x_values = np.arange(0, 10.1, 0.1)
    y_values = np.arange(0, 10.1, 0.1)
    xx, yy = np.meshgrid(x_values, y_values)
    test_data = np.column_stack((xx.ravel(), yy.ravel()))

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    predicted_labels = knn.predict(test_data)
    
    return xx, yy, predicted_labels

def plot_results(xx, yy, predicted_labels, X_train, y_train):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

    plt.figure(figsize=(10, 8))
    plt.scatter(xx.ravel(), yy.ravel(), c=predicted_labels, cmap=cmap_light)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("kNN Classification (k=3)")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Adding colorbar
    cb = plt.colorbar()
    loc = np.arange(0, max(y_train), max(y_train)/float(len(np.unique(y_train))))
    cb.set_ticks(loc)
    cb.set_ticklabels(np.unique(y_train))

    plt.show()


#Q1
df=pd.read_excel("C:\\Users\\Rahul\\OneDrive\\Desktop\\sem 4\\\ML\\Assignment\data1.xlsx")

mean0fgrade = df['radius_mean'].mean()
df['results'] = np.where(df['radius_mean'] > mean0fgrade, 1, 0)
result = df['results']

grad=df['radius_mean']

X_train,X_test,Y_train,Y_test = train_test_split(grad,result, test_size=0.01, shuffle=True)


X_train1=np.reshape(X_train,(-1,1))
Y_train1=np.reshape(Y_train,(-1,1))

X_test1=np.reshape(X_test,(-1,1))
Y_test1=np.reshape(Y_test,(-1,1))

knn=KNeighborsClassifier(3)

accuracies = []
for k in range(1, 12):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train1, Y_train)
    accuracy = knn.score(X_test1, Y_test)
    accuracies.append(accuracy)

train_predictions = knn.predict(X_train1)
test_predictions = knn.predict(X_test1)

train_conf_matrix = confusion_matrix(Y_train, train_predictions)
print("Confusion Matrix for Training Data:")
print(train_conf_matrix)

test_conf_matrix = confusion_matrix(Y_test, test_predictions)
print("Confusion Matrix for Test Data:")
print(test_conf_matrix)

train_precision = precision_score(Y_train, train_predictions, average=None)
print("Precision for Training Data:", train_precision)
test_precision = precision_score(Y_test, test_predictions, average=None)
print("Precision for Test Data:", test_precision)

train_recall = recall_score(Y_train, train_predictions, average=None)
print("Recall for Training Data:", train_recall)
test_recall = recall_score(Y_test, test_predictions, average=None)
print("Recall for Test Data:", test_recall)

train_f1 = f1_score(Y_train, train_predictions, average=None)
print("F1-score for Training Data:", train_f1)
test_f1 = f1_score(Y_test, test_predictions, average=None)
print("F1-score for Test Data:", test_f1)


#Q1
df = pd.read_excel("C:\\Users\\Rahul\\OneDrive\\Desktop\\sem 4\\ML\\Assignment\\data1.xlsx", sheet_name='Purchase data')
df=pd.DataFrame(df)
print(df)
#convert datafram to array
Array=df.to_numpy()
A=Array[:,1:4]
C=Array[:,4:5]
#typecast the elements to float
A=np.float64(A)
#return the pseudoinverse
pinv=np.linalg.pinv(A)
print("pseudo inverse:",pinv)

#calculating the cost of each product
print("Costs of the products:",np.matmul(pinv,C))
pinv2=np.matmul(pinv,C)

#comparing real prices with the model for calculating prices
print("real prices:",C)
predicted=np.matmul(A,pinv2)
print("model of calculating prices:",predicted)

#funtion to calculate MSE
def MSE(C,predicted):
    sum=0
    for i in range(len(C)):
        sub=0
        sub=C[i]-predicted[i]
        sub=sub*sub
        sum+=sub
    return sum/len(C)
print("Mean squared error:",MSE(C,predicted))

#funtion to calculate RMSE
def RMSE(C,predicted):
    sum=0
    for i in range(len(C)):
        sub=0
        sub=C[i]-predicted[i]
        sub=sub*sub
        sum+=sub
    return math.sqrt(sum/len(C))
print("RMSE:",RMSE(C,predicted))

#funtion to calculate MAPE
def MAPE(C,predicted):
    sum=0
    for i in range(len(C)):
        sub=0
        sub=C[i]-predicted[i]
        sub=abs((sub)/C[i])
        sum+=sub
    return sum/len(C)
print("MAPE:",MAPE(C,predicted))

#funtion to calculate R2 score
import math
import numpy as np
def R2(C,predicted):
    rss=0
    for i in range(len(C)):
        err=0
        err=C[i]-predicted[i]
        rss += err*err
    tss=0
    mean=np.mean(predicted)
    for i in range(len(C)):
        err=0
        err=C[i]-mean
        tss += err*err   
    return 1-(rss/tss)
print("R2",R2(C,predicted))
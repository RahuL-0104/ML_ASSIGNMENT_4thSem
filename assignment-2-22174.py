#EUCLIDEAN DISTANCE
from math import sqrt

def Create_Vector(Dimension_of_Vector):               # we ask user for the dimensions.
    vector = []
    for i in range(Dimension_of_Vector):              # Iterate through each dimension.
        vector.append(float(input(f"Enter the {i}th element of the vector: ")))# we append the elements.
    return vector

def Manhattan(vector_A, vector_B):                    # creating a function called manhattan.
    manhattan_distance = 0
    for i in range(len(vector_A)):                    # we find len(A) and then iterate through every elemet.
        manhattan_distance += abs(vector_B[i] - vector_A[i])  # we find manhattan distance using the formula.
    return (manhattan_distance)                       # we return the balue stores in the variable manhattan_distance.

def Euclidean_Distance(Vector_C, Vector_D):           # Function called euclidean distance>
    square_of_distance = [(Vector_D[i] - Vector_C[i]) ** 2 for i in range(len(Vector_C))] # direct formuale for euclidean distance.
    euclidean_distance = sqrt(sum(square_of_distance))# we finally use sqrt to get the square root value of the square of the distnace.
    return (euclidean_distance)                       # we return value stored in euclidean distance.

X_vector = Create_Vector(int(input("Enter the dimension of X vector: "))) # Asking the user to enter the dimension of the vector
Y_vector = Create_Vector(int(input("Enter the dimension of Y vector: ")))

print("Manhattan Distance is:", Manhattan(X_vector, Y_vector)) # we can pass any valude here iresspective of input taken in the function class.
print("Euclidean Distance is:", Euclidean_Distance(X_vector, Y_vector))

#MANHATTAN DISTANCE 

def calculate_Manhattan_distance(vectorA, vectorB): # Funtion to find the Manhattan distance.
    distance = 0
    for i in range(len(vectorB)):                   # we iterate through len(vector 2) to find number of elements in the string. 
        distance += abs(vectorB[i] - vectorA[i])    # Equation to find the Manhattan distance.
    return (distance)                               # we return the manhattan distance.

def find_nearest_neighbors(X, Y, Target, K):        # Function to find K nearest neighbors.
    distances_between_them = {i: calculate_Manhattan_distance(X[i], Y[i]) for i in range(len(X))} # Calculate the Manhattan distance.
    nearest_indices = sorted(distances_between_them, key=distances_between_them.get)[:K] # Sorting the Distance between them.
    nearest_labels = [Target[i] for i in nearest_indices] # Get corresponding target labels of K nearest neighbors.
    return (nearest_labels) 

X = [[2], [6], [3], [5], [9]]
Y = [[7], [3], [8], [1], [4]]
Target = ['bad', 'good', 'good', 'bad', 'bad']

K = int(input("Enter the value of K: "))           # Entering the K value.
nearest_neighbors = find_nearest_neighbors(X, Y, Target, K) # finding nearest k neighbour values.
print(f"Nearest neighbors for K = {K}: {nearest_neighbors}") # using the f string to format accordingly.

#LABEL ENCODING

def label_encoding(data):                         # seperating data into label encoding.

    unique_categories = set(data) #Finding the unique categories
    encoding_dict = {} #Dictionary to map categories to numeric labels
    for i, category in enumerate(unique_categories):
        encoding_dict[category] = i

    numerical_data = [] #Convert categorical data to numeric using
    for category in data:
        numerical_data.append(encoding_dict[category])

    return numerical_data

categorical_data = ["dog", "cat", "dog", "bird", "cat", "bird", "dog"] #Input
numerical_data = label_encoding(categorical_data) #Using the funtion to convert the data
print("The variables to be converted ",categorical_data)
print("After converting the variables to label encoding data: ") #Result
print(numerical_data)

#QUESTION 4

def one_hot_encoding(data): #Function find one hot encoding of set of data

    unique_categories = set(data) #Getting the unique categories for set of data
    encoding_dict = {}
    for category in unique_categories: #Condition to iterate through categories

        one_hot_encoding = []
        for unique_category in unique_categories: #Iterating to find the unique character

            if category == unique_category: #Condition to make sure category is equal to unique category
                one_hot_encoding.append(1)
            else:
                one_hot_encoding.append(0)

        encoding_dict[category] = one_hot_encoding #Storing the one hot encoding for the category in the dictionary


    one_hot_encoded_data = [] #Convert categorical data to one hot encoded numeric data using the encoding dictionary

    for category in data:
        one_hot_encoded_data.append(encoding_dict[category])

    return one_hot_encoded_data

categorical_data = ["dog", "cat", "dog", "bird", "cat", "bird", "dog"] #Input
numerical_data = one_hot_encoding(categorical_data) #Using the funtion to convert the data
print("Tha variable to be coverted ",categorical_data)
print("After converting the variables to one hot encoding data: ") #Result
for item in numerical_data:
    print(item)
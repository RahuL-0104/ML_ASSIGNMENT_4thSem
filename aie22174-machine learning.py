#question1
def find_vowels(string1):                               # question1 find count of vowels.
    vowels=["a","e","i","o","u","A","E","I","O","U"]                        # defined a list consisting of vowels.
    count_of_vowels=0                                   # initialising vowels count=0 because till now we didnt find any vowel.
    count_of_consonants=0                               # initialising consonants count=0 because till now we didnt find any consonant.
    for i in string1:                                   # we are iterating thorugh every element in string1 whichwe get through user input.
     if i in vowels:                                    # iterating throught vowels and checking if any element from string 1 is in list vowels.
      count_of_vowels=count_of_vowels+1                 # if yes vowel count increases by 1.
     else:  
        count_of_consonants= count_of_consonants+1      # else count of consonant increases by 1.

    return (count_of_vowels,count_of_consonants)        # returns the 2 parameters we want.



#question2
def common_list(list1,list2):                           # question 2 to find common elemenst in 2 lists.
  count_of_common_elements=0                            # count of elementsis 0 because till now no element has matches.
  for i in list1:                                       # iterate in list 1.
    if (i in list2):                                    
      count_of_common_elements+=1                       # and if element from list1 is in list2 count adds by 1.
  return (count_of_common_elements)                     # return the value.



#question 3
def find_transpose(input_matrix):                        # question 3 to find transpose of the matrix.  
  matrix2=[]                                             # we make a second matrix which is a copy of input matrix.
  for i in range(len(input_matrix)):                     # this is to get the total number of rows. 
      row=[]                                             # this is to row of matrix2
    for j in range(len(input_matrix[0])):                # this is to get the columns.
        row.append(input_matrix[j][i])                   # store values in the row of matrix2
    matrix2.append(row)                                  # adding the rows to the matri
  return matrix2



#question 4
def matrix_multiplication(Matrix_a,Matrix_b):
  result=[[0 for column in range(len(Matrix_b[0]))] for row in range (len (Matrix_a))]
  for i in range (len(Matrix_a)):
    for j in range(len(Matrix_b[0])):
      for k in range(len(Matrix_b)):
        result[i][j]+=Matrix_a[i][k]*Matrix_b[k][j]
  return (result)      

def main():                                              # creating a main function.
 print("which operating do you want to select")                                 
 operation=int(input("""
               1.find no of vowels for a given string
               2.find common elements in the list
               3.find transpose of the matrix 
               4.find multiplication of matrix
               """))
 if operation==1:                                       # if operatin is equal to 1.
    input_string=input("Enter a string:")               # we take input string to send it to the function later.
    vowels,consonants=find_vowels(input_string)         # we call the function and staroe the values in 2 variables.
    print("no of vowels are:",vowels,"no of consonants are:",consonants)# we print count of vowels and consonants.

 elif operation==2:                                     # if operation is equal to 2.
  input_list_1=input("enter list1 by spaces")           # we take 2 lists as input from user.
  input_list_2=input("enter list2 by spaces")
  list_1=[int(i) for i in input_list_1.split()]         # the user splits and gives the values so we use split keyword to seperate each and every value.
  list_2=[int(j) for j in input_list_2.split()]          
  count=common_list(list_1,list_2)                      #we call the function common_list.
  print("no of common elements are:",count)
 
 elif operation ==3:                               # nested if statement to check condition 3.
  rows_of_the_matrix = int(input("Enter the number of rows: "))  # Input the rows and colums of the matrix and type script into integer value.
  columns_of_the_matrix = int(input("Enter the number of columns: "))
  matrix = []                                       # creating an empty matrix to store values later.
  print("Enter the elements row by row:")
  for i in range(rows_of_the_matrix):               # iterating through every element in rows.
    row_of_empty_matrix= []                         # creating rows to stoe input values.
    for j in range(columns_of_the_matrix):          # iterating in colums.
        element = int(input(f"Enter element at position ({i+1}, {j+1}): "))# after stroing one element i+1 makes the iteration to go to next element.
        row_of_empty_matrix.append(element)         # we append the input elements to row.
    matrix.append(row_of_empty_matrix)              # we append the input row to empty matrix we have created in line 59.
  transpose_of_the_matrix=find_transpose(matrix)    # we are callling the function.
  print(transpose_of_the_matrix)  
  
 elif operation==4:
  rows1 = int(input("Enter the number of rows: "))  # Input the rows and colums of the matrix and type script into integer value.
  columns1 = int(input("Enter the number of columns: "))
  Matrix_a = []                                     # creating an empty matrix to store values later.
  print("Enter the elements row by row:")
  for i in range(rows1):                            # iterating through every element in rows.
      row1= []                                      # creating rows to store input values.
      for j in range(columns1):                     # iterating in colums.
          element = int(input(f"Enter element at position ({i+1}, {j+1}): "))# after stroing one element i+1 makes the iteration to go to next element.
          row1.append(element)                      # we append the input elements to row.
      Matrix_a.append(row1)                         # we append the input row to empty matrix we have created in line 59.
  rows2=int(input("enter the no of rows:"))         # take row2 as input
  columns__2=int(input("enter the number of colums:"))#take column as input.
  Matrix_b=[]                                       # create an empty matrib_b.
  print("enter the elements by rows:")              # we ask user for number of rows.
  for l in range(rows2):                            # iterate through row2.
      row2=[]                                       # we define row2 and store elements in it later.
      for j in range(columns__2):                   # iterate through column__2.
          element=int(input(f"enter the element at position ({i+1},{j+1}):"))# we input elements into every position of the matrix.
          row2.append(element)                      # we append elemets into row2 which we defined earlier.
      Matrix_b.append(row2)                         # append rows to matrix created earlier.
  mult_of_matrices=matrix_multiplication(Matrix_a,Matrix_b)#we call the function matrix_multiplication.
 print(mult_of_matrices)                            #print the variable.

main()                                              #we call main function

import numpy as np 
#element = 2*np.arange(4).reshape((2, 2))
element = np.array([[0, 2], 
                    [4, 8]])
# print(element)
new_element = element[len(element)-1]
print(new_element)

test_elements = [0, 4]
mask = np.isin(element, test_elements)
indices = np.where(np.isin(element,test_elements))[0]

print(indices)

A = np.array([[0,1,2,3,4],[9,8,7,6,5],[10,11,12,13,14]])
ind = np.unravel_index(np.argmax(A, axis=None), A.shape)
print(ind)

# f= open("guru99.txt","w+")


# f.write(str(A[0]))
# for i in range(0, 5):
#     f.write(",")
#     f.write(str(A[i]))

# f.close()

# S = sum(A)
# print(S)

# for i in range(len(A)-1):
#     print(A[i], end = ",")
# print(A[len(A)-1])
B = np.array(range(2,11))
print(B)

C = [0] *20
print(C)

dict = {"a": [1,0], "b": [2,2], "c": [9,3], "d": [8,4]}

# data = dict.values()
an_array = np.array([dict[k] for k in dict.keys()])
print(an_array)
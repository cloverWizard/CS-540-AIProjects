import numpy as np
from math import log2
import copy

'''
This script is using all 9 features (2,3,...,10) to create a tree, which serves as a template.
Todo: you need to modify this by using the several specified features to create your own tree 
Todo: you need to do the pruning yourself
Todo: you need to get all the output including the test results.
Todo: you also need to generate the tree of such the format in the writeup: 'if (x3 <= 6) return 2 .......'
'''

with open('breast-cancer-wisconsin.data', 'r') as f:
    a = [l.strip('\n').split(',') for l in f if '?' not in l]


a = np.array(a).astype(int)   # training data

f = open("output.txt", "w+")
np.savetxt("output.txt", a, fmt="%.4f")


f1 = open("p2q1.txt", "w+")
f2 = open("p2q2.txt", "w+")
f3 = open("p2q3.txt", "w+")
f4 = open("p2q4.txt", "w+")

ben = sum(a[:,-1] == 2)
mag = sum(a[:,-1] == 4)

# q1
# print(str(ben)+ "\t" + str(mag))
f1.write(str(ben))
f1.write(",")
f1.write(str(mag))
f1.close()

def entropy(data):
    count = len(data)
    p0 = sum(b[-1] == 2 for b in data) / count
    if p0 == 0 or p0 == 1: return 0
    p1 = 1 - p0
    return -p0 * log2(p0) - p1 * log2(p1)

# q2
f2.write(str(round(entropy(a), 4)))
f2.close()

def infogain(data, fea, threshold):  # x_fea <= threshold;  fea = 2,3,4,..., 10; threshold = 1,..., 9
    count = len(data)
    d1 = data[data[:, fea - 1] <= threshold]
    d2 = data[data[:, fea - 1] > threshold]
    if len(d1) == 0 or len(d2) == 0: return 0
    return entropy(data) - (len(d1) / count * entropy(d1) + len(d2) / count * entropy(d2))

max_ig_2 = -1
index = 0
for i in range(len(a)):
    ig = infogain(a, 2, a[i][1])
    # print(ig)
    if ig > max_ig_2:
        max_ig_2 = ig
        index = i
# print(index)

# q3
threshold = a[index][1]
above_benign = 0
below_benign = 0
above_malignant = 0
below_malignant = 0
for i in range(len(a)):
    if a[i, 1] > threshold and a[i, -1] == 2:
        above_benign += 1
    if a[i, 1] <= threshold and a[i, -1] == 2:
        below_benign += 1
    if a[i, 1] > threshold and a[i, -1] == 4:
        above_malignant += 1
    if a[i, 1] <= threshold and a[i, -1] == 4:
        below_malignant += 1
f3.write(str(above_benign))
f3.write(",")
f3.write(str(below_benign))
f3.write(",")
f3.write(str(above_malignant))
f3.write(",")
f3.write(str(below_malignant))
f3.close()

# q4
f4.write(str(max_ig_2))
f4.close()

indices = [4, 5, 7, 8, 2]

def find_best_split(data):
    c = len(data)
    c0 = sum(b[-1] == 2 for b in data) # # of 2s in training set
    if c0 == c: return (2, None)
    if c0 == 0: return (4, None)
    ig = [[infogain(data, f, t) for t in range(1, 10)] for f in [4, 5, 7, 8, 2]]
    # f.write(str(type(ig)) + "\n")
    ig = np.array(ig)
    # ig dimensions 9 * 9
    max_ig = max(max(i) for i in ig)
    if max_ig == 0:
        if c0 >= c - c0:
            return (2, None)
        else:
            return (4, None)
    ind = np.unravel_index(np.argmax(ig, axis=None), ig.shape)
    fea, threshold = indices[ind[0]], ind[1] + 1
    return (fea, threshold)

fea, threshold = find_best_split(a)
# print(fea, threshold)

def split(data, node):
    fea, threshold = node.fea, node.threshold
    d1 = data[data[:,fea-1] <= threshold]
    d2 = data[data[:, fea-1] > threshold]
    return (d1,d2)

class Node:
    def __init__(self, fea, threshold):
        self.fea = fea
        self.threshold = threshold
        self.left = None
        self.right = None



ig = [[infogain(a, fea, t) for t in range(1,10)] for fea in [2,4,5,7,8]]
ig = np.array(ig)
ind = np.unravel_index(np.argmax(ig, axis=None), ig.shape)
root = Node(indices[ind[0]], ind[1] + 1)


def create_tree(data, node, depth):
    if depth < 4:
        depth += 1
        depth_left = depth
        depth_right = depth
        d1,d2 = split(data, node)
        f1, t1 = find_best_split(d1)
        f2, t2 = find_best_split(d2)
        if t1 == None: node.left = f1
        else:
            node.left = Node(f1,t1)
            depth_left = create_tree(d1, node.left, depth)
        if t2 == None: node.right = f2
        else:
            node.right = Node(f2,t2)
            depth_right = create_tree(d2, node.right, depth)
        return max(depth_left, depth_right)
    else: 
        d1,d2 = split(data, node)
        c_left = len(d1)
        c_left_1 = sum(b[-1] == 2 for b in d1)
        if c_left_1 > c_left - c_left_1: node.left = 2
        else: node.left = 4

        c_right = len(d2)
        c_right_1 = sum(b[-1] == 2 for b in d2)
        if c_right_1 > c_right - c_right_1: node.right = 2
        else: node.right = 4

        return depth

depth_tree = create_tree(a, root, 0) + 1
# print(depth_tree)

s1 = [root] # new way to create string!
s2 = []
depth = -1
while s1:
    s2 = copy.deepcopy(s1)
    s1 = []
    for n in s2:
        if n != 2 and n != 4:
            print("feature, theshold ")
            print(n.fea, n.threshold)
            if n.left != None: s1 += [n.left]
            else: print("no left child")
            if n.right != None: s1 += [n.right]
            else: print("no right child")
        else:
            print(n)
    depth += 1
    print("depth = ", depth)

f.close()


with open('test.txt', 'r') as f:
    b = [l.strip('\n').split(',') for l in f if '?' not in l]

b = np.array(b).astype(int)   # training data

f7 = open("p2q7.txt", "w+")

for d in b:
    # f7.write(str(d[5]))
    if (d[4] <= 2):   
        if (d[1] <= 6):
            if (d[7] <= 3):
                f7.write(str(2))
            else: f7.write(str(4))
        else: f7.write(str(4))
    else:
        if (d[3] <= 2):
            if (d[6] <= 4):
                f7.write(str(2))
            else: f7.write(str(4))
        else: f7.write(str(4))
    f7.write
  
f7.close()

f8 = open("p2q8.txt", "w+")

def printTree(root): 
    
    f8.write("if (x")
    f8.write(str(root.fea))
    f8.write(" <= ")
    f8.write(str(root.threshold))
    f8.write(") ")

    if root.left != 2 and root.left != 4: 
        f8.write("\n")
        printTree(root.left) 
    else:
        f8.write("return ")
        f8.write(str(root.left))
        f8.write("\n")

    f8.write("else")
    f8.write(" ")
    if root.right != 2 and root.right != 4: 
        f8.write("\n")
        printTree(root.right) 
    else:
        f8.write("return ")
        f8.write(str(root.right))
        f8.write("\n")
printTree(root)
        
f9 = open("p2q9.txt", "w+")

def DFS(root, data):
    if data[root.fea-1] <= root.threshold:
        if root.left == 2:
            return 2
        if root.left == 4:
            return 4
        else:
            return DFS(root.left,data)
    else:
        if root.right == 2:
            return 2
        if root.right == 4:
            return 4
        else:
            return DFS(root.right,data)
for i in b:
    f9.write(str(DFS(root, i)))
    f9.write(",")
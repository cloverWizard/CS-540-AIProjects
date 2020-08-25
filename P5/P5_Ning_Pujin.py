import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
# %matplotlib inline
import copy
import math
import heapq

'''
The below script is based on a 55 * 57 maze. 
Todo:
	1. Plot the maze and solution in the required format.
	2. Implement DFS algorithm. (I've given you the BFS below)
	3. Implement A* with Euclidean distance. (I've given you the one with Manhattan distance)

'''



width, height = 57, 58
X, Y = 14, 2

ori_img = mpimg.imread('maze.png')
img = ori_img[:,:,0]

class Cell:
    def __init__(self, i, j):
        self.i = i
        self.j = j
        self.succ = ''
        self.action = ''  # which action the parent takes to get this cell
cells = [[Cell(i,j) for j in range(width)] for i in range(height)]

f2 = open("p5q2.txt", "w+")
for i in range(height):
    succ = []
    for j in range(width):
        s = ''
        c1, c2 = i * 16 + 8, j * 16 + 8
        if img[c1-8, c2] == 1: s += 'U'
        if img[c1+8, c2] == 1: s += 'D'
        if img[c1, c2-8] == 1: s += 'L'
        if img[c1, c2+8] == 1: s += 'R'
        cells[i][j].succ = s
        succ.append(s)
        f2.write(s)
        if j < width-1: f2.write(",")
    if i < height-1: f2.write("\n")
    

maze = np.empty((2*height+1, 2*width+1),dtype='object')
for i in range(height+1):
    for j in range(width+1):
            maze[2*i][2*j] = '+'
            if i < height and j < width:
                c1, c2 = i * 16 + 8, j * 16 + 8
                if img[c1-8, c2] == 0: maze[2*i][2*j+1] = "--"
                if img[c1+8, c2] == 0: maze[2*i+2][2*j+1] = "--"
                if img[c1, c2-8] == 0: maze[2*i+1][2*j] = "|"
                if img[c1, c2+8] == 0: maze[2*i+1][2*j+2] = "|"

f4 = open("p5q4.txt", "w+")
with open("p5q1.txt", "w+") as f1:
    # f1.write(str(maze))
    for i in range(2*height+1):
        for j in range(2*width+1):
            if i % 2 == 1 and j % 2 == 1: 
                f1.write("  ")
                # f4.write("  ")
            else:
                if maze[i][j] == None: 
                    if i % 2 == 0: 
                        f1.write("  ")
                        # f4.write("  ")
                    else: 
                        f1.write(" ")
                        # f4.write(" ")
                else: 
                    f1.write(maze[i][j])
                    # f4.write(maze[i][j])
        if i < 2*height:
            f1.write("\n") 
        # f4.write("\n") 

# 2    


cells[0][28].succ = cells[0][28].succ.replace('U', '')
cells[57][28].succ = cells[57][28].succ.replace('D', '')

# bfs
visited = set()
s1 = {(0,28)}
s2 = set()
while (57,28) not in visited:
    for a in s1:
        visited.add(a)
        i, j = a[0], a[1]
        succ = cells[i][j].succ
        if 'U' in succ and (i-1,j) not in (s1 | s2 | visited): 
            s2.add((i-1,j))
            cells[i-1][j].action = 'U'
        if 'D' in succ and (i+1,j) not in (s1 | s2 | visited): 
            s2.add((i+1,j))
            cells[i+1][j].action = 'D'
        if 'L' in succ and (i,j-1) not in (s1 | s2 | visited): 
            s2.add((i,j-1))
            cells[i][j-1].action = 'L'
        if 'R' in succ and (i,j+1) not in (s1 | s2 | visited): 
            s2.add((i,j+1))
            cells[i][j+1].action = 'R'     
    s1 = s2
    s2 = set()
with open("p5q5.txt", "w+") as f5:
    for i in range(height):
        for j in range(width):
            if (i,j) in visited: f5.write(str(1))
            else: f5.write(str(0))
            if j < width-1:f5.write(",")
        if i < height-1: f5.write("\n")


cur = (57,28)
s = ''
seq = []
while cur != (0,28):
    seq.append(cur)
    i, j = cur[0], cur[1]
    t = cells[i][j].action
    s += t
    if t == 'U': cur = (i+1, j)
    if t == 'D': cur = (i-1, j)
    if t == 'L': cur = (i, j+1)
    if t == 'R': cur = (i, j-1)
action = s[::-1]
seq = seq[::-1]
with open("output.txt", "w+") as f:
    f.write(str(seq))

# dfs
visited_d, stack = set(), [(0,28)]   
while (57,28) not in visited_d and stack:
    # print(str(stack))
    vertex = stack.pop()
    if vertex not in visited_d:
        visited_d.add(vertex)
        i, j = vertex[0], vertex[1]
        succ = cells[i][j].succ
        # print(str(succ))
        if 'U' in succ and (i-1,j) not in visited_d and (i-1,j) not in stack: 
            stack.append((i-1,j))
        if 'D' in succ and (i+1,j) not in visited_d and (i+1,j) not in stack: 
            stack.append((i+1,j))
        if 'L' in succ and (i,j-1) not in visited_d and (i,j-1) not in stack: 
            stack.append((i,j-1))
        if 'R' in succ and (i,j+1) not in visited_d and (i,j+1) not in stack: 
            stack.append((i,j+1))    

with open("p5q6.txt", "w+") as f6:
    for i in range(height):
        for j in range(width):
            if (i,j) in visited_d: f6.write(str(1))
            else: f6.write(str(0))
            if j < width-1:f6.write(",")
        if i < height-1: f6.write("\n")


with open("p5q3.txt", "w+") as f3:
    f3.write(action)
    # print(action)
maze_solution = copy.deepcopy(maze)
maze_solution[0*2,28*2+1] = "##"
maze_solution[0*2+1,28*2+1] = "##"
maze_solution[57*2+2, 28*2+1] = "##"

x_i, x_j = 0, 28
for direction in action:
    i, j = 2*x_i+1, 2*x_j+1
    if direction == 'U':
        x_i -= 1
        maze_solution[i-1][j] = "##"
        maze_solution[i-2][j] = "##"
    if direction == 'D':
        x_i += 1
        maze_solution[i+1][j] = "##"
        maze_solution[i+2][j] = "##"
    if direction == 'L':
        x_j -= 1
        maze_solution[i][j-1] = "#"
        maze_solution[i][j-2] = "##"
    if direction == 'R':
        x_j += 1
        maze_solution[i][j+1] = "#"
        maze_solution[i][j+2] = "##"    

for i in range(2*height+1):
        for j in range(2*width+1):
            if i % 2 == 1 and j % 2 == 1 and maze_solution[i][j] == None: 
                f4.write("  ")
            else:
                if maze_solution[i][j] == None: 
                    if i % 2 == 0: 
                        f4.write("  ")
                    else:  
                        f4.write(" ")
                else: 
                    f4.write(maze_solution[i][j])
        if i < 2*height:
            f4.write("\n") 
f4.close()
# 3 


## Part2
man = {(i,j): abs(i-57) + abs(j-28) for j in range(width) for i in range(height)}
euc = {(i,j): math.sqrt((i-57)**2 + (j-28)**2 ) for j in range(width) for i in range(height)}

# manhattan   use man
g = {(i,j): float('inf') for j in range(width) for i in range(height)}
g[(0,28)] = 0

queue = [(0,28)]
visited = set()

while queue and (57,28) not in visited:
    queue.sort(key=lambda x: g[x] + man[x])
    point = queue.pop(0)
    if point not in visited:
        visited.add(point)
        # print(str(point))
        i, j = point[0], point[1]
        succ = cells[i][j].succ
        if 'U' in succ and (i-1,j) not in visited:
            if (i-1,j) not in queue: queue += [(i-1,j)]
            g[(i-1,j)] = min(g[(i-1,j)], g[(i,j)]+1)
        if 'D' in succ and (i+1,j) not in visited:
            if (i+1,j) not in queue: queue += [(i+1,j)]
            g[(i+1,j)] = min(g[(i+1,j)], g[(i,j)]+1)
        if 'L' in succ and (i,j-1) not in visited:
            if (i,j-1) not in queue: queue += [(i,j-1)]
            g[(i,j-1)] = min(g[(i,j-1)], g[(i,j)]+1)
        if 'R' in succ and (i,j+1) not in visited:
            if (i,j+1) not in queue: queue += [(i,j+1)]
            g[(i,j+1)] = min(g[(i,j+1)], g[(i,j)]+1)     

with open("p5q7.txt", "w+") as f7:
    for i in range(height):
        for j in range(width):
            f7.write(str(man[i,j]))
            if j < width-1: f7.write(",")
        if i < height-1: f7.write("\n")

with open("p5q8.txt", "w+") as f8:
    for i in range(height):
        for j in range(width):
            if (i,j) in visited: f8.write(str(1))
            else: f8.write(str(0))
            if j < width-1:f8.write(",")
        if i < height-1: f8.write("\n")

g = {(i,j): float('inf') for j in range(width) for i in range(height)}
g[(0,28)] = 0

queue = [(0,28)]
visited = set()
while queue and (57,28) not in visited:
    queue.sort(key=lambda x: g[x] + euc[x])
    point = queue.pop(0)
    if point not in visited:
        visited.add(point)
        # print(str(point))
        i, j = point[0], point[1]
        succ = cells[i][j].succ
        if 'U' in succ and (i-1,j) not in visited:
            if (i-1,j) not in queue: queue += [(i-1,j)]
            g[(i-1,j)] = min(g[(i-1,j)], g[(i,j)]+1)
        if 'D' in succ and (i+1,j) not in visited:
            if (i+1,j) not in queue: queue += [(i+1,j)]
            g[(i+1,j)] = min(g[(i+1,j)], g[(i,j)]+1)
        if 'L' in succ and (i,j-1) not in visited:
            if (i,j-1) not in queue: queue += [(i,j-1)]
            g[(i,j-1)] = min(g[(i,j-1)], g[(i,j)]+1)
        if 'R' in succ and (i,j+1) not in visited:
            if (i,j+1) not in queue: queue += [(i,j+1)]
            g[(i,j+1)] = min(g[(i,j+1)], g[(i,j)]+1)
with open("p5q9.txt", "w+") as f9:
    for i in range(height):
        for j in range(width):
            if (i,j) in visited: f9.write(str(1))
            else: f9.write(str(0))
            if j < width-1:f9.write(",")
        if i < height-1: f9.write("\n")

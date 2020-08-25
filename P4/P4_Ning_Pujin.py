import numpy as np
import math
from sklearn.cluster import AgglomerativeClustering

'''
Todo: 
1. Part 1 in P4.
2. Euclidean distance (currently are all manhattan in my code below)
3. Complete linkage distance
4. Total distortion
5. Output all required information in correct format

PS: Currently, I choose 
	n = num of all distinct countries, and
	m = 3 (latitude, longitude, total deaths until Jun27, 
		  i.e., 1st, 2nd, last number for each country as parameters).
	Also, for countries that have several rows, I average the latitude, longitude and sum up the deaths.

	You may need to change some of that based on your part 1 results.

'''

np.set_printoptions(suppress = True)

# For 'South Korea', and "Bonaire Sint Eustatius and Saba" (line 145 and 257), I removed the ',' in name manually
with open('time_series_covid19_deaths_global.csv') as f:
    data = list(f)[1:]

d_dict = {}
days = 0 # # of days in the record
for d in data:
    l = d.strip('\n').split(',')
    c = l[1]  # country
    days = len(l) - 4
    if c == "\"Korea":
        l[1] += l[2]
        del l[2]
    if l[2] == "Netherlands":
        l[0] += l[1]
        del l[1]
    if c in d_dict:
        for i in range(4,len(l)):
            d_dict[c][i-4] += float(l[i])
    else:
        d_dict[c] = [float(l[i]) for i in range(4,len(l))]
        # d_dict[c] = []
    # if c in d_dict:
    #     d_dict[c][0].append(float(l[2]))
    #     d_dict[c][1].append(float(l[3]))
    #     d_dict[c][2].append(float(l[-1]))
    # else:
    #     d_dict[c] = [[float(l[2])], [float(l[3])], [float(l[-1])]]

f = open("output.txt", "w+")
f.write(str(d_dict))
# print(sum(v[0]) for v in d_dict.items())
# d_dict = {k:np.array([sum(v[0])/len(v[0]), sum(v[1])/len(v[1]), sum(v[2])]) for k,v in d_dict.items()}
f.close()

index = {}
for k in d_dict.keys():
    if d_dict[k][-1] < 10:
        continue
    index[k] = [0,0,0,0]
    for i in range(len(d_dict[k])):
        if d_dict[k][i] <= d_dict[k][-1]/8:
            index[k][0] = i 
        if d_dict[k][i] <= d_dict[k][-1]/4:
            index[k][1] = i 
        if d_dict[k][i] <= d_dict[k][-1]/2:
            index[k][2] = i 
        index[k][3] = len(d_dict[k])

d_dict = {}
for k in index.keys():
    d_dict[k] = np.array([index[k][1]-index[k][0], index[k][2]-index[k][1], index[k][3]-index[k][2]])

with open("output.txt", "w+") as f:
    f.write(str(d_dict))

with open("p4q4.txt", "w+") as f4:
    for k in d_dict.keys():
        f4.write(str(d_dict[k][0]))
        f4.write(",")
        f4.write(str(d_dict[k][1]))
        f4.write(",")
        f4.write(str(d_dict[k][2]))
        f4.write("\n")

US = [0] * days
Canada = [0] * days

for d in data:
    l = d.strip('\n').split(',')
    c = l[1]  # country
    records = l[4:]
    # print(len(records))
    if c == "US":
        US = [US[i] + float(records[i]) for i in range(len(records))]
        # print(US)
    if c == "Canada":
        Canada = [Canada[i] + float(records[i]) for i in range(len(records))]
        # print(Canada)

with open('p4q1.txt', 'w+') as f1:
    for item in US:
        f1.write("%s," % item)
    f1.write("\n")
    for item in Canada:
        f1.write("%s," % item)

with open('p4q2.txt', 'w+') as f2:
    for i in range(days-1):
        f2.write("%s," % (US[i+1]-US[i]))
    f2.write("\n")
    for i in range(days-1):
        f2.write("%s," % (Canada[i+1]-Canada[i]))

countries = sorted([c for c in d_dict.keys()])


def manhattan(a,b):
    return sum(abs(a[i]-b[i]) for i in range(len(a)))

def euclidean(a,b):
    return math.sqrt(sum((a[i]-b[i]) ** 2 for i in range(len(a))))

def distance(a,b):
    return sum((a[i]-b[i]) ** 2 for i in range(len(a)))
# print(countries[0],countries[1])
# print(euclidean(d_dict[countries[0]],d_dict[countries[1]]))
 # single linkage distance
def sld(cluster1, cluster2): 
    res = float('inf')
    # c1, c2 each is a country in the corresponding cluster
    for c1 in cluster1:
        for c2 in cluster2:
            dist = euclidean(d_dict[c1], d_dict[c2])
            if dist < res:
                res = dist
    return res

def cld(cluster1, cluster2): 
    res = 0.0
    # c1, c2 each is a country in the corresponding cluster
    for c1 in cluster1:
        for c2 in cluster2:
            dist = euclidean(d_dict[c1], d_dict[c2])
            if dist > res:
                res = dist
    return res

k = 5
# print(countries)
fc = open("clusters", "w+")
# hierarchical clustering (sld, 'manhattan')
n = len(d_dict)
clusters = [{d} for d in d_dict.keys()]
for _ in range(n-k):
    dist = float('inf')
    best_pair = (None, None)
    for i in range(len(clusters)-1):
        for j in range(i+1, len(clusters)):
            if cld(clusters[i], clusters[j]) <= dist:
                dist = cld(clusters[i], clusters[j])
                best_pair = (i,j)
    new_clu = clusters[best_pair[0]] | clusters[best_pair[1]]
    clusters = [clusters[i] for i in range(len(clusters)) if i not in best_pair]
    clusters.append(new_clu)
fc.write(str(clusters))
fc.close()
X = np.array([d_dict[k] for k in d_dict.keys()])
clustering_s = AgglomerativeClustering(linkage='single', n_clusters=5).fit(X)
clustering_c = AgglomerativeClustering(linkage='complete', n_clusters=5).fit(X)

with open('p4q5.txt', 'w+') as f5:
    # print(len(clustering_s.labels_))
    for ind in clustering_s.labels_:
        f5.write(str(ind))
        f5.write(",")
    # for c in d_dict.keys():
    #     for i in range(5):
    #         if c in clusters[i]:
    #             f5.write(str(i))
    #             f5.write(",")
with open('p4q6.txt', 'w+') as f6:
    # print(len(clustering_c.labels_))
    # for ind in clustering_c.labels_:
    #     f6.write(str(ind))
    #     f6.write(",")
    for c in d_dict.keys():
        for i in range(5):
            if c in clusters[i]:
                f6.write(str(i))
                f6.write(",")

## k-means (eucidean)
import copy
def center(cluster):
    return np.average([d_dict[c] for c in cluster], axis=0)

init_num = np.random.choice(len(countries) - 1, k)
clusters = [{countries[i]} for i in init_num]
while True:
    new_clusters = [set() for _ in range(k)]
    centers = [center(cluster) for cluster in clusters]
    for c in countries:
        clu_ind = np.argmin([euclidean(d_dict[c], centers[i]) for i in range(k)])
        new_clusters[clu_ind].add(c)
    if all(new_clusters[i] == clusters[i] for i in range(k)):
        break
    else:
        clusters = copy.deepcopy(new_clusters)

clustering = {}
with open("p4q7.txt", "w+") as f7:
    for c in d_dict.keys():
        for i in range(5):
            if c in clusters[i]:
                clustering[c] = i
                f7.write(str(i))
                f7.write(",")
# print(clustering)
with open("p4q8.txt", "w+") as f8:
    for cluster in clusters:
        centers = center(cluster)
        f8.write(str(round(centers[0],4)))
        f8.write(",")
        f8.write(str(round(centers[1],4)))
        f8.write(",")
        f8.write(str(round(centers[2],4)))
        f8.write("\n")

distortion = 0
with open("p4q9.txt", "w+") as f9:
    for cluster in clusters:
        print(cluster)
        for country in cluster:
            cen = center(cluster)
            cen = [round(c,4) for c in cen]
            print(distance(d_dict[country],cen))
            distortion += distance(d_dict[country],cen)
            print(str(distortion))  
    f9.write(str(distortion))

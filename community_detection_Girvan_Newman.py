import sys
from pyspark import SparkConf, SparkContext, SQLContext
from collections import defaultdict
import time
import itertools
import random


start = time.time()
sc = SparkContext()


threshold= int(sys.argv[1])
input_path=sys.argv[2]
between_out=sys.argv[3]
comm_out=sys.argv[4]


def user_pairs(x):
    business_id = x[0]
    user_ids = x[1]
    res = []
    pairs = list(itertools.combinations(user_ids, 2))
    for i in pairs:
        i = sorted(i)
        res.append(((i[0], i[1]), business_id))
    return  res


def bfs(root,adjacency_dict):
    t=dict()
    t[root]=(0,list())

    visited= {}
    tobevisited=[]
    tobevisited.append(root)

    while len(tobevisited) >0:
        parent=tobevisited.pop(0)
        if parent not in visited:
            visited[parent]=1
        for children in adjacency_dict[parent]:
            if children not in visited.keys():
                visited[children]=1
                t[children]=(t[parent][0]+1,[parent])
                tobevisited.append(children)
            elif t[parent][0]+1 == t[children][0]:
                t[children][1].append(parent)
    ans={}
    for k,v in sorted(t.items(),key=lambda  x: -x[1][0]):
        ans[k]=v
    return ans

def edge_weight(tdict,nodes):
    weight_dict={}
    for node in nodes:
        weight_dict[node]=1
    levels={}
    shortest_path={}
    for child,parent in tdict.items():
        if parent[0] not in levels:
            levels[parent[0]]=[]
        levels[parent[0]].append((child,parent[1]))
    for l in range(0,len(levels.keys())):
        for (child,parent) in levels[l]:
            if len(parent)>0:
                a=0
                for p in parent:
                    a+=shortest_path[p]
                shortest_path[child]=a
            else:
                shortest_path[child]=1
    ans={}
    for k,v in tdict.items():
        if len(v[1])>0:
            temp=0
            for parent in v[1]:
                temp+=shortest_path[parent]
            denominator=temp
            for parent in v[1]:
                if k>parent:
                    t=(parent,k)
                elif k<parent:
                    t=(k,parent)
                weight=float(float(weight_dict[k])*int(shortest_path[parent])/denominator)
                ans[t]=weight
                weight_dict[parent]=float(weight+weight_dict[parent])

    return ans



def betweenness(nodes,adjacency_dict):
    result={}
    for node in nodes:
        bfs_dict=bfs(node,adjacency_dict)
        traverse=edge_weight(bfs_dict,nodes)

        for k,v in traverse.items():
            if k in result.keys():
                result[k]=float(result[k]+v)
            else:
                result[k]=v

    r=dict(map(lambda x:(x[0],float(x[1]/2)),result.items()))
    r2=sorted(r.items(),key=lambda x:(-x[1],x[0][0]))
    return r2


def modularity(nodes,adjacency_list,user_set):
    community = list()
    tobevisited = list()
    visited = set()
    individual_community = set()
    n = len(nodes)

    random_r = nodes[random.randint(0, len(nodes) - 1)]
    individual_community.add(random_r)
    tobevisited.append(random_r)

    while len(visited) != n:
        while len(tobevisited) > 0:
            parent = tobevisited.pop(0)
            visited.add(parent)
            individual_community.add(parent) ##
            for children in adjacency_list[parent]:
                if children not in visited:
                    individual_community.add(children)
                    visited.add(children)
                    tobevisited.append(children)
        community.append(sorted(individual_community))
        individual_community=set()
        if n>len(visited):
            tobevisited.append(set(nodes).difference(visited).pop())
    v= set()
    cnt=0
    for start,end in adjacency_list.items():
        for e in end:
            k=tuple(sorted([start,e]))

            if  k not in v:
                v.add(k)
                cnt+=1
    m=cnt
    t_sum=0
    for cluster in community:
        combo=itertools.combinations(list(cluster),2)
        for pair in combo:
            if pair[0]>pair[1]:
                ans=(pair[1],pair[0])
            else:
                ans=(pair[0],pair[1])
            k_i=len(adjacency_list[pair[0]])
            k_j=len(adjacency_list[pair[1]])
            if ans in user_set:
                a=1
            else:
                a=0
            t_sum+=float(a-(k_i*k_j/(2*m)))
    temp=float(t_sum/(2*m))
    final=(community,temp)
    return final




input_first = sc.textFile(input_path).map(lambda x: x.split(",")).map(lambda x:(x[0], x[1]))

header=input_first.first()

data=input_first.filter(lambda x: x!=header)

## Edges and nodes construction
businesswise_users= data.map(lambda x: (x[1], x[0])).groupByKey().mapValues(lambda x: sorted(list(x)))
edges_users= businesswise_users.flatMap(lambda x: user_pairs(x)).groupByKey().mapValues(lambda x: list(set(x))).filter(lambda x: len(x[1])>=threshold).map(lambda x: x[0])
edges_users_collect=edges_users.collect()

##
nodes= edges_users.flatMap(lambda x: list(x)).distinct()
nodes_collect=nodes.collect()
no_nodes=len(nodes.collect())

adjacency_dict=edges_users.union(edges_users.map(lambda x:(x[1],x[0]))).groupByKey().mapValues(lambda x: list(set(x))).collectAsMap()


vertex_weight_dict=dict.fromkeys(nodes_collect,1)

edges_users_set=set(edges_users_collect)

betweenness_list=betweenness(nodes_collect,adjacency_dict)

with open(between_out,'w+') as outpath:
    for item in betweenness_list:
        outpath.writelines(str(item)[1:-1]+"\n")
    outpath.close()

max_modularity=float("-inf")

if len(betweenness_list)>0:
    edge_highest = betweenness_list[0][0]
    if adjacency_dict[edge_highest[0]] is not None:
        try:
            adjacency_dict[edge_highest[0]].remove(edge_highest[1])
        except ValueError:
            pass

    if adjacency_dict[edge_highest[1]] is not None:
        try:
            adjacency_dict[edge_highest[1]].remove(edge_highest[0])
        except ValueError:
            pass

    temp=modularity(nodes_collect,adjacency_dict,edges_users_set)
    b_community=temp[0]
    mm=temp[1]
    betweenans=betweenness(nodes_collect, adjacency_dict)

while True:
    edge_highest = betweenans[0][0]
    if adjacency_dict[edge_highest[0]] is not None:
        try:
            adjacency_dict[edge_highest[0]].remove(edge_highest[1])
        except ValueError:
                pass

    if adjacency_dict[edge_highest[1]] is not None:
        try:
            adjacency_dict[edge_highest[1]].remove(edge_highest[0])
        except ValueError:
            pass
    temp2 = modularity(nodes_collect, adjacency_dict,edges_users_set)
    community = temp2[0]
    cm = temp2[1]
    betweenans=betweenness(nodes_collect,adjacency_dict)
    if cm<mm:
        break
    else:
        b_community=community
        mm=cm

sorted_communities=sorted(b_community,key=lambda x:(len(x),x[0],x[1]))


with open(comm_out,'w+') as outpath:
    for item in sorted_communities:
        outpath.writelines(str(item)[1:-1]+"\n")
    outpath.close()


import pyspark
import sys
import os
import time
import itertools
from graphframes import *
from collections import defaultdict

from pyspark.sql import SparkSession,SQLContext


os.environ["PYSPARK_SUBMIT_ARGS"] = '--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 pyspark-shell'

conf = pyspark.SparkConf()
conf.setMaster('local[16]')
conf.setAppName('assignment_4')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)
sc.setLogLevel("ERROR")



filter_threshold=int(sys.argv[1])
input_path = sys.argv[2]
output_path = sys.argv[3]

def get_user_pairs(x):
    business_id = x[0]
    user_ids = x[1]
    res = []

    pairs = list(itertools.combinations(user_ids, 2))
    for i in pairs:
        i = sorted(i)
        res.append(((i[1],i[0]),business_id))
        res.append(((i[0], i[1]), business_id))

    return res



start = int(time.time())

input_first = sc.textFile(input_path).map(lambda x: x.split(",")).map(lambda x:(x[0], x[1]))

header=input_first.first()

data=input_first.filter(lambda x: x!=header)

user_business=data.groupByKey().mapValues(lambda x: sorted(list(x))).collectAsMap()




users=data.map(lambda x: x[0]).distinct()
users_collections = users.collect()

businesswise_users= data.map(lambda x: (x[1], x[0])).groupByKey().mapValues(lambda x: sorted(list(x)))
edges_users= businesswise_users.flatMap(lambda x: get_user_pairs(x)).groupByKey().mapValues(lambda x: list(set(x))).filter(lambda x: len(x[1])>=filter_threshold).map(lambda x: x[0])
nodes=edges_users.flatMap(lambda x:list(x)).map(lambda x: tuple([x])).distinct()
edges_users_collection = edges_users.collect()

edge_df=edges_users.toDF(["src","dst"])
edge_df.show()
vertex_df=nodes.toDF(['id'])
vertex_df.show()


g=GraphFrame(vertex_df, edge_df)
result=g.labelPropagation(maxIter=5)
result.show()


communityRdd=result.rdd.map(tuple)
communities = communityRdd.map(lambda x: (x[1],x[0])).groupByKey().map(lambda x: sorted(list(x[1]))).sortBy(lambda x: (len(x), x[0]))

final = communities.collect()


with open(output_path,"w+") as f:
    f.write('\n'.join(["'"+"', '".join(x) for x in final]))
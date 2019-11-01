import os
import time
from pyspark import SparkContext
from pyspark.sql import SQLContext
import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

sc = SparkContext()
sqlc = SQLContext(sc)

def process(from_path,save_path):
    # path='training_set'
    if not os.path.exists(from_path):
         print('数据路径不存在')
    path_list=os.listdir(from_path)
    path_list.sort()
    i=0
    start=time.time()
    for f in path_list:
        i=i+1
        file_path=os.path.join(from_path,f)
        name=f.split('.')[0]+'.csv'
        save_path=os.path.join(save_path,name)
        data=pd.read_table(file_path,header=None,skiprows=1,sep=',',names=['userid','rate','time'])
        data['itemid'] = i
        data.to_csv(save_path,index=False)
    end=time.time()
    print('转换数据用时',end-start)
    return i


#读取数据
def get_data(from_path,save_path):
    if not os.listdir(save_path):
        i=process(from_path)
    print('读取数据')
    data_all=pd.DataFrame(columns=('userid','itemid','rate','time'))
    # path_list = os.listdir(save_path)
    # path_list.sort()
    #n=0
    for f in os.listdir(save_path):
        file_path = os.path.join(save_path, f)
        data= pd.read_csv(file_path)
        data = data.reindex(columns=['userid', 'itemid', 'rate', 'time'])
        data_all = data_all.append(data)
        # n=n+1
        # print('导入数据',n)
    sdata = sqlc.createDataFrame(data_all)
    print('数据的条数：',sdata.count())
    print('数据不重复的条数：',sdata.distinct().count())
    sdata=sdata.dropDuplicates() #删除重复项
    return sdata

def model_als(sdata):
    start=time.time()
    training,test = sdata.randomSplit([0.8, 0.2])
    alsExplicit = ALS(maxIter=5, regParam=0.01, userCol="userid", itemCol="itemid", ratingCol="rate") #regParam 是ALS的正则化参数
    modelExplicit = alsExplicit.fit(training)
    predictionsExplicit = modelExplicit.transform(test)
    predictionsExplicit.show()
    evaluator = RegressionEvaluator().setMetricName("rmse").setLabelCol("rate").setPredictionCol("prediction")
    rmse=evaluator.evaluate(predictionsExplicit)
    end=time.time()
    print('耗时为：',end-start)
    return rmse

if __name__=='__main__':
    from_path='training_set'
    save_path='samples'
    sdata=get_data(from_path,save_path)
    rmse=model_als(sdata)
    print('rmse为',rmse)

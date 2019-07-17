


from pyspark.sql import SparkSession


spark = SparkSession.builder \
    .master("local[20]") \
    .appName("COM6012 Spark Intro") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

print("Answer for Q1.1")

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import Binarizer
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.classification import LogisticRegression
import numpy as np
import time
init = time.time()
rawdata = spark.read.csv('../Data/HIGGS.csv.gz', sep=',', header='false').cache()
rawdata = rawdata.repartition(10).cache()
rawdata_sampled = rawdata.sample(False, .25, 50).cache() # setting seed value as 50
rawdata_sampled = rawdata_sampled.repartition(10).cache()

schemaNames = rawdata_sampled.schema.names
ncolumns = len(rawdata_sampled.columns)
from pyspark.sql.types import DoubleType
for i in range(ncolumns):
    rawdata_sampled = rawdata_sampled.withColumn(schemaNames[i], rawdata_sampled[schemaNames[i]].cast(DoubleType()))
rawdata_sampled = rawdata_sampled.withColumnRenamed('_c0', 'label')


assembler = VectorAssembler(inputCols = schemaNames[1:ncolumns], outputCol = 'features') 
sampled_raw_plus_vector = assembler.transform(rawdata_sampled).cache()

data = sampled_raw_plus_vector.select('features','label')

trainingData, testData = data.randomSplit([0.7, 0.3], 50)

print("\n")
print("Starting Decision Tree Classifier for 25% of dataset...")
print("\n")

evaluator = MulticlassClassificationEvaluator\
      (labelCol="label", predictionCol="prediction", metricName="accuracy")
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
pipeline = Pipeline(stages=[dt])
paramGrid = ParamGridBuilder() \
    .addGrid(dt.maxDepth, [5, 10, 15]) \
    .addGrid(dt.maxBins, [30, 31, 32]) \
    .addGrid(dt.impurity, ["gini", "entropy"])\
    .build()
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                         numFolds = 2)
cvModel = crossval.fit(trainingData)
bestModel = cvModel.bestModel
bestnewModel = bestModel.stages[0]
bestParams = bestnewModel.extractParamMap()
print("\n")
print("The best parameters for Decision Tree Classifier are...")
print("\n")
for x in bestParams:
    print (x.name,bestParams[x])
print("\n")
print("Printing the selected training parameters... ")
print("\n")
maxDepth_dtc = bestnewModel._java_obj.getMaxDepth()
print("Best maxDepth = ", maxDepth_dtc)
maxBins_dtc = bestnewModel._java_obj.getMaxBins()
print("Best maxBins = ", maxBins_dtc)
impurity_dtc = bestnewModel._java_obj.getImpurity()
print("Best impurity = ", impurity_dtc)


prediction = cvModel.transform(testData)
accuracy = evaluator.evaluate(prediction)
print("\n")
print("Accuracy for DecisionTreeClassifier = %g " % accuracy)
evaluate_area_reg = BinaryClassificationEvaluator(rawPredictionCol = "rawPrediction")
reg_area = evaluate_area_reg.evaluate(prediction)
print("\n")
print("Area under the curve for DecisionTreeClassifier = %g " % reg_area)

print("\n")
print("Starting Decision Tree Regressor 25% of dataset...")
print("\n")

evaluator_reg = RegressionEvaluator\
      (labelCol="label", predictionCol="prediction", metricName="rmse")
dtr = DecisionTreeRegressor(labelCol="label", featuresCol="features")
pipeline_dtr = Pipeline(stages=[dtr])
paramGrid_reg = ParamGridBuilder() \
    .addGrid(dtr.maxDepth, [5, 10, 30]) \
    .addGrid(dtr.maxBins, [20, 35, 40]) \
    .build()
crossval_reg = CrossValidator(estimator=pipeline_dtr,
                          estimatorParamMaps=paramGrid_reg,
                          evaluator=evaluator_reg,
                          numFolds=2)
cvModel_reg = crossval_reg.fit(trainingData)
prediction_reg = cvModel_reg.transform(testData)
binarizer = Binarizer(threshold=0.5, inputCol="prediction", outputCol="binarized_prediction")
binarizedDataFrame = binarizer.transform(prediction_reg)
binarized = binarizedDataFrame.drop('prediction')
bdf = binarized.withColumnRenamed('binarized_prediction', 'prediction')
bestModel_reg = cvModel_reg.bestModel
bestnewModel_reg = bestModel_reg.stages[0]
bestParams_reg = bestnewModel_reg.extractParamMap()
print("\n")
print("The best parameters for Decision Tree Regressor are...")
print("\n")
for x in bestParams_reg:
    print (x.name,bestParams_reg[x])
print("\n")
print("Printing the selected training parameters... ")
print("\n")
maxDepth_dtr = bestnewModel_reg._java_obj.getMaxDepth()
print("Best maxDepth = ", maxDepth_dtr)
maxBins_dtr = bestnewModel_reg._java_obj.getMaxBins()
print("Best maxBins = ", maxBins_dtr)
evaluator_reg = MulticlassClassificationEvaluator\
      (labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy_reg = evaluator_reg.evaluate(bdf)
print("\n")
print("Accuracy for DecisionTreeRegressor = %g " % accuracy_reg)
evaluate_area_bdf = BinaryClassificationEvaluator(rawPredictionCol = "prediction")
reg_area = evaluate_area_bdf.evaluate(prediction_reg)
print("\n")
print("Area under the curve for DecisionTreeRegressor = %g " % reg_area)



print("\n")
print("Starting Logistic Regression 25% of dataset...")
print("\n")


evaluator_lr = MulticlassClassificationEvaluator\
      (labelCol="label", predictionCol="prediction", metricName="accuracy")
lr = LogisticRegression(featuresCol='features', labelCol='label')
pipeline_lr = Pipeline(stages=[lr])
paramGrid_lr = ParamGridBuilder() \
    .addGrid(lr.maxIter, [5, 10, 15]) \
    .addGrid(lr.regParam, [0.1, 0.2, 0.3]) \
    .build()
crossval_lr = CrossValidator(estimator=pipeline_lr,
                          estimatorParamMaps=paramGrid_lr,
                          evaluator=evaluator_lr,
                          numFolds=2)
cvModel_lr = crossval_lr.fit(trainingData)
prediction_lr = cvModel_lr.transform(testData)
bestModel_lr = cvModel_lr.bestModel
bestnewModel_lr = bestModel_lr.stages[0]
bestParams_lr = bestnewModel_lr.extractParamMap()
print("\n")
print("The best parameters for Logistic Regression are...")
print("\n")
for x in bestParams_lr:
    print (x.name,bestParams_lr[x])
print("\n")
print("Printing the selected training parameters... ")
print("\n")
maxIter_lr = bestnewModel_lr._java_obj.getMaxIter()
print("Best maxIter = ", maxIter_lr)
regParam_lr = bestnewModel_lr._java_obj.getRegParam()
print("Best regParam = ", regParam_lr)


accuracy_lr = evaluator.evaluate(prediction_lr)
print("\n")
print("Accuracy for LogisticRegression = %g " % accuracy_lr)
evaluate_area_lr = BinaryClassificationEvaluator(rawPredictionCol = "rawPrediction")
lr_area = evaluate_area_reg.evaluate(prediction_lr)
print("\n")
print("Area under the curve for LogisticRegression = %g " % lr_area)
print("\n")

########################################################################################
########################################################################################
########################################################################################
########################################################################################

print("Answer for Q1.2")

schemaNames = rawdata.schema.names
ncolumns = len(rawdata.columns)
from pyspark.sql.types import DoubleType
for i in range(ncolumns):
    rawdata = rawdata.withColumn(schemaNames[i], rawdata[schemaNames[i]].cast(DoubleType()))
rawdata = rawdata.withColumnRenamed('_c0', 'label')
assembler = VectorAssembler(inputCols = schemaNames[1:ncolumns], outputCol = 'features') 
alldata_plus_vector = assembler.transform(rawdata).cache()
data_all = alldata_plus_vector.select('features','label')
trainingData, testData = data_all.randomSplit([0.7, 0.3], 50)
print("\n")
print("For the whole dataset, the DecisionTreeClassifier is starting...")
evaluator = MulticlassClassificationEvaluator\
      (labelCol="label", predictionCol="prediction", metricName="accuracy")
print("\n")
print("Fetching the best values of parameters from 25% dataset and using them...")
ctime = time.time()
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth = maxDepth_dtc, maxBins = maxBins_dtc, impurity = impurity_dtc)
model_dt = dt.fit(trainingData)
predictions_dt = model_dt.transform(testData)
c = time.time() - ctime
accuracy_dt = evaluator.evaluate(predictions_dt)
print("\n")
print("Accuracy for DecisionTreeClassifier on the whole dataset = %g " % accuracy_dt)
evaluate_area_dt = BinaryClassificationEvaluator(rawPredictionCol = "rawPrediction")
area_dt = evaluate_area_dt.evaluate(predictions_dt)
print("\n")
print("Area under the curve for DecisionTreeClassifier on the whole dataset = %g " % area_dt)
print("\n")
print("Time taken (in seconds) to train the DecisionTreeClassifier algorithm = ", c)
print("\n")

print("For the whole dataset, the DecisionTreeRegressor is starting...")
evaluator_reg = RegressionEvaluator\
      (labelCol="label", predictionCol="prediction", metricName="rmse")
print("\n")
print("Fetching the best values of parameters from 25% dataset and using them...")
rtime = time.time()
dtr = DecisionTreeRegressor(labelCol="label", featuresCol="features", maxDepth = maxDepth_dtr, maxBins = maxBins_dtr)
model_dtr = dtr.fit(trainingData)
predictions_dtr = model_dtr.transform(testData)
binarizer = Binarizer(threshold=0.5, inputCol="prediction", outputCol="binarized_prediction")
binarizedDataFrame = binarizer.transform(predictions_dtr)
binarized = binarizedDataFrame.drop('prediction')
bdf_dtr = binarized.withColumnRenamed('binarized_prediction', 'prediction')
r = time.time() - rtime
evaluator_reg = MulticlassClassificationEvaluator\
      (labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy_reg = evaluator_reg.evaluate(bdf_dtr)
print("\n")
print("Accuracy for DecisionTreeRegressor on the whole dataset = %g " % accuracy_reg)
evaluate_area_dtr = BinaryClassificationEvaluator(rawPredictionCol = "prediction")
area_dtr = evaluate_area_dtr.evaluate(predictions_dtr)
print("\n")
print("Area under the curve for DecisionTreeRegressor on the whole dataset = %g " % area_dtr)
print("\n")
print("Time taken (in seconds) to train the DecisionTreeRegressor algorithm = ", r)
print("\n")

print("For the whole dataset, the LogisticRegression is starting...")
evaluator_lr = MulticlassClassificationEvaluator\
      (labelCol="label", predictionCol="prediction", metricName="accuracy")
print("\n")
print("Fetching the best values of parameters from 25% dataset and using them...")
ltime = time.time()
lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter = maxIter_lr, regParam = regParam_lr)
model_lr = lr.fit(trainingData)
predictions_lr = model_lr.transform(testData)
l = time.time() - ltime
accuracy_log = evaluator.evaluate(predictions_lr)
print("\n")
print("Accuracy for LogisticRegression = %g " % accuracy_log)
evaluate_area_log = BinaryClassificationEvaluator(rawPredictionCol = "rawPrediction")
lr_area = evaluate_area_reg.evaluate(prediction_lr)
print("\n")
print("Area under the curve for LogisticRegression = %g " % lr_area)
print("\n")
print("Time taken (in seconds) to train the LogisticRegression algorithm = ", l)
print("\n")

########################################################################################
########################################################################################
########################################################################################
########################################################################################

print("Answer for Q1.3")
imp_dtc = model_dt.featureImportances
imp_feat_dtc = np.zeros(ncolumns-1)
imp_feat_dtc[imp_dtc.indices] = imp_dtc.values
idx_dtc = (-imp_feat_dtc).argsort()[:3]
print("\n")
print("Top 3 features for DecisionTreeClassifier are... ")
print("\n")
for x in idx_dtc:
    print (schemaNames[x+1])

imp_dtr = model_dtr.featureImportances
imp_feat_dtr = np.zeros(ncolumns-1)
imp_feat_dtr[imp_dtr.indices] = imp_dtr.values
idx_dtr = (-imp_feat_dtr).argsort()[:3]
print("\n")
print("Top 3 features for DecisionTreeRegressor are... ")
print("\n")
for x in idx_dtr:
    print (schemaNames[x+1])
	
imp_lr = model_lr.coefficients.values
sorted_index = np.argsort(imp_lr)
top_3 = sorted_index[:3]
print("\n")
print("Top 3 features for LogisticRegression are... ")
print("\n")
for x in top_3:
    print (schemaNames[x+1])

final = time.time() - init
print("The code took (in seconds) ", final)
spark.stop()

##################################################
# Churn Prediction using PySpark 
##################################################

# Can you develop a machine learning model that can predict customers leaving the company?
# The aim is to predict whether a bank's customers will leave the bank or not.

# 1. Installation
# 2. Exploratory Data Analysis
# 3. SQL Queries
# 4. Data Preprocessing & Feature Engineering
# 5. Modelling

##################################################
# Installation
##################################################

# https://spark.apache.org/downloads.html
# download under username/spark.

# pyarrow: A module that provides ease of working between different data structures.

# pip install pyspark
# pip install findspark
# pip install PyArrow


import warnings
import findspark
import pandas as pd
import seaborn as sns
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.ml.feature import Bucketizer


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

findspark.init("/Users/onder-tanrikulu/spark/spark-3.1.1-bin-hadoop2.7")

spark = SparkSession.builder \
    .master("local") \
    .appName("pyspark_intro") \
    .getOrCreate()

# PoC

sc = spark.sparkContext

# http://onder-tanrikulu-pc1.lan:4040/jobs/
# sc.stop()


##################################################
# Exploratory Data Analysis
##################################################

############################
# Understanding the difference between Pandas df and Spark df.
############################

spark_df = spark.read.csv("churn.csv", header=True, inferSchema=True)
type(spark_df)

df = sns.load_dataset("diamonds")
type(df)

df.head()
spark_df.head()

df.dtypes
spark_df.dtypes

df.ndim
# spark_df.ndim


# Reading a json file
# df = spark.read.json(json_file_path)
#
# # Reading a text file
# df = spark.read.text(text_file_path)

# # Reading a parquet file
# df = spark.read.load(parquet_file_path) # or
# df = spark.read.parquet(parquet_file_path)

############################
# Exploratory Data Analysis
############################

# Number of observations and variables
print("Shape: ", (spark_df.count(), len(spark_df.columns)))

# Variable types
spark_df.printSchema()
spark_df.dtypes

# Variable selection
spark_df.Age

# Selecting and displaying a single variable
spark_df.select(spark_df.Age).show()

# Head
spark_df.show(3, truncate=True)
spark_df.take(5)
spark_df.head()

# Variable names in lower case
spark_df = spark_df.toDF(*[c.lower() for c in spark_df.columns])
spark_df.show(5)


# Summary statistics
spark_df.describe().show()

# Summary statistics for certain variables
spark_df.describe(["age", "churn"]).show()

# Statistics of categorical variables
spark_df.groupby("churn").count().show()

# Unique classes
spark_df.select("churn").distinct().show()

# select(): Show some variables
spark_df.select("age", "names").show(5)

# filter(): Observation selection / filtering
spark_df.filter(spark_df.age > 40).show()
spark_df.filter(spark_df.age > 40).count()

# groupby 
spark_df.groupby("churn").count().show()
spark_df.groupby("churn").agg({"age": "mean"}).show()


# Selection and summary statistics of all numeric variables
num_cols = [col[0] for col in spark_df.dtypes if col[1] != 'string']

spark_df.select(num_cols).describe().toPandas().transpose()

# Selection and summary of all categorical variables
cat_cols = [col[0] for col in spark_df.dtypes if col[1] == 'string']

for col in cat_cols:
    spark_df.select(col).distinct().show()


for col in [col.lower() for col in num_cols]:
    spark_df.groupby("churn").agg({col: "mean"}).show()


##################################################
# SQL Queries
##################################################

spark_df.createOrReplaceTempView("tbl_df")

spark.sql("show databases").show()

spark.sql("show tables").show()

spark.sql("select age from tbl_df limit 5").show()

spark.sql("select churn, avg(age) from tbl_df group by Churn").show()

##################################################
# Data Preprocessing & Feature Engineering
##################################################

############################
# Missing Values
############################

from pyspark.sql.functions import when, count, col

spark_df.select([count(when(col(c).isNull(), c)).alias(c) for c in spark_df.columns]).toPandas().T

# delete rows with missing value
spark_df.dropna().show()


# Replacing the missing values with a certain value
spark_df.fillna(50).show()

# Filling the missing values based on some variables
spark_df.na.fill({'age': 50, 'names': 'unknown'}).show()


############################
# Feature Interaction
############################


spark_df = spark_df.withColumn('age_total_purchase', spark_df.age / spark_df.total_purchase)
spark_df.show(5)

############################
# Bucketization / Bining / Num to Cat
############################

spark_df.select('age').describe().toPandas().transpose()

bucketizer = Bucketizer(splits=[0, 35, 45, 65], inputCol="age", outputCol="age_cat")

spark_df = bucketizer.setHandleInvalid("keep").transform(spark_df)

spark_df.show(20)

spark_df = spark_df.withColumn('age_cat', spark_df.age_cat + 1)


spark_df.groupby("age_cat").count().show()

spark_df.groupby("age_cat").agg({'churn': "mean"}).show()

spark_df = spark_df.withColumn("age_cat", spark_df["age_cat"].cast("integer"))

spark_df.groupby("age_cat").agg({'churn': "mean"}).show()

############################
# Create features with 'when' (segment)
############################

spark_df = spark_df.withColumn('segment', when(spark_df['years'] < 5, "segment_b").otherwise("segment_a"))

############################
# Create features with 'when'(age_cat_2)
############################


spark_df.withColumn('age_cat_2',
                    when(spark_df['age'] < 36, "young").
                    when((35 < spark_df['age']) & (spark_df['age'] < 46), "mature").
                    otherwise("senior")).show()


############################
# Label Encoding
############################

spark_df.show(5)
indexer = StringIndexer(inputCol="segment", outputCol="segment_label")
indexer.fit(spark_df).transform(spark_df).show(5)

temp_sdf = indexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("segment_label", temp_sdf["segment_label"].cast("integer"))
spark_df.show(5)

spark_df = spark_df.drop('segment')


############################
# One Hot Encoding
############################

encoder = OneHotEncoder(inputCols=["age_cat"], outputCols=["age_cat_ohe"])

spark_df = encoder.fit(spark_df).transform(spark_df)
spark_df.show(5)


############################
# Defining the TARGET
############################

stringIndexer = StringIndexer(inputCol='churn', outputCol='label')
temp_sdf = stringIndexer.fit(spark_df).transform(spark_df)

spark_df = temp_sdf.withColumn("label", temp_sdf["label"].cast("integer"))
spark_df.show(5)


############################
# Defining the features
############################

cols = ['age', 'total_purchase', 'account_manager', 'years',
        'num_sites', 'age_total_purchase', 'segment_label', 'age_cat_ohe']


# Vectorize independent variables.
va = VectorAssembler(inputCols=cols, outputCol="features")
va_df = va.transform(spark_df)
va_df.show()

# Final sdf
final_df = va_df.select("features", "label")
final_df.show(5)

# StandardScaler
# scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
# final_df = scaler.fit(final_df).transform(final_df)

# Split the dataset into test and train sets.
train_df, test_df = final_df.randomSplit([0.7, 0.3], seed=17)
train_df.show(10)
test_df.show(10)

print("Training Dataset Count: " + str(train_df.count()))
print("Test Dataset Count: " + str(test_df.count()))


##################################################
# Modelling
##################################################

############################
# Logistic Regression
############################

log_model = LogisticRegression(featuresCol='features', labelCol='label').fit(train_df)
y_pred = log_model.transform(test_df)
y_pred.show()

y_pred.select("label", "prediction").show()

# accuracy
y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count()

#

evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName='areaUnderROC')
evaluatorMulti = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

acc = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "accuracy"})
precision = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "precisionByLabel"})
recall = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "recallByLabel"})
f1 = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "f1"})
roc_auc = evaluator.evaluate(y_pred)

print("accuracy: %f, precision: %f, recall: %f, f1: %f, roc_auc: %f" % (acc, precision, recall, f1, roc_auc))


############################
# Gradient Boosted Tree Classifier
############################

gbm = GBTClassifier(maxIter=100, featuresCol="features", labelCol="label")
gbm_model = gbm.fit(train_df)
y_pred = gbm_model.transform(test_df)
y_pred.show(5)

y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count()


############################
# Model Tuning
############################

evaluator = BinaryClassificationEvaluator()

gbm_params = (ParamGridBuilder()
              .addGrid(gbm.maxDepth, [2, 4, 6])
              .addGrid(gbm.maxBins, [20, 30])
              .addGrid(gbm.maxIter, [10, 20])
              .build())

cv = CrossValidator(estimator=gbm,
                    estimatorParamMaps=gbm_params,
                    evaluator=evaluator,
                    numFolds=5)

cv_model = cv.fit(train_df)


y_pred = cv_model.transform(test_df)
ac = y_pred.select("label", "prediction")
ac.filter(ac.label == ac.prediction).count() / ac.count()


############################
# New Prediction
############################


names = pd.Series(["Ali Ahmetoğlu", "Taner Gün", "Berkay", "Polat Konak", "Kamil Atasoy"])
age = pd.Series([18, 43, 34, 50, 40])
total_purchase = pd.Series([5000, 10000, 6000, 30000, 100000])
account_manager = pd.Series([1, 0, 0, 1, 1])
years = pd.Series([20, 10, 3, 8, 30])
num_sites = pd.Series([2, 8, 8, 6, 50])
age_total_purchase = age / total_purchase
segment_label = pd.Series([1, 1, 0, 1, 1])
age_cat_ohe = pd.Series([1, 1, 0, 2, 1])


new_cust = pd.DataFrame({
    'names': names,
    'age': age,
    'total_purchase': total_purchase,
    'account_manager': account_manager,
    'years': years,
    'num_sites': num_sites,
    "age_total_purchase": age_total_purchase,
    "segment_label": segment_label,
    "age_cat_ohe": age_cat_ohe})


new_sdf = spark.createDataFrame(new_cust)
new_customers = va.transform(new_sdf)
new_customers.show(3)

results = cv_model.transform(new_customers)
results.select("names", "prediction").show()


##################################################
# User Defined Functions (UDFs)
##################################################

def age_converter(age):
    if age < 35:
        return 1
    elif age < 45:
        return 2
    elif age <= 65:
        return 3


from pyspark.sql.types import IntegerType, StringType, FloatType
from pyspark.sql.functions import udf

func_udf = udf(age_converter, IntegerType())

spark_df = spark_df.withColumn('age_cat2', func_udf(spark_df['age']))

spark_df.show(5)

spark_df.groupby("age_cat2").count().show()

cat_cols.append('age_cat2')


def age_converter(age):
    if age < 35:
        return "young"
    elif age < 45:
        return "mature"
    elif age <= 65:
        return "senior"


func_udf = udf(age_converter, StringType())
spark_df = spark_df.withColumn('age_cat3', func_udf(spark_df['age']))
spark_df.show(5)
spark_df.groupby("age_cat3").count().show()
cat_cols.append('age_cat3')



def segment(years):
    if years < 5:
        return "segment_b"
    else:
        return "segment_a"



func_udf = udf(segment, StringType())
spark_df = spark_df.withColumn('segment', func_udf(spark_df['years']))
spark_df.show(5)
spark_df.groupby("segment").count().show()


##################################################
# Pandas UDFs
##################################################
# Creates a pandas user defined function (a.k.a. vectorized user defined function).


spark_df.withColumn('age_square', spark_df.age ** 2).show()

from pyspark.sql.functions import pandas_udf

@pandas_udf(FloatType())
def pandas_square(col):
    return col ** 2

spark_df.select(pandas_square(spark_df.age)).show()

spark_df = spark_df.withColumn('age_square', pandas_square(spark_df.age))

@pandas_udf(FloatType())
def pandas_log(col):
    import numpy as np
    return np.log(col)

spark_df.withColumn('age_log', pandas_log(spark_df.age)).show()




































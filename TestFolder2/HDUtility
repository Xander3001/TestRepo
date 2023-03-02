
from pyspark.sql.types import *
from pyspark.sql.functions import *

def exploreTree(blobFiles,path):
  for ent in dbutils.fs.ls(path):
    if (ent.path[-1:]) == "/":
      #print(f"exploring: {ent.path}")
      exploreTree(blobFiles,ent.path)
    else:
      #print(ent.path)
      blobFiles.append(ent.path)
  return blobFiles

def listFiles(path):
  blobFiles = []
  exploreTree(blobFiles,path)
  
  blobdf = spark.createDataFrame(blobFiles, StringType())
  blobdf = (blobdf
       .withColumnRenamed("value","filePath")
       .withColumn("fileType",regexp_extract(col("filePath"),"\.(.*)",1))
       .withColumn("entity",split("filePath","/")[4])
       .withColumn("entitylvl2",split("filePath","/")[5])
       .withColumn("entitylvl3",split("filePath","/")[6])
       .withColumn("entitylvl4",split("filePath","/")[7])
       .withColumn("fileName",regexp_extract(split("filePath","/")[size(split("filePath","/"))-1],"(.*)\.",1))
       )
  blobdf = blobdf.filter("fileType != ''")

  return blobdf

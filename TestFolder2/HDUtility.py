# This code defines two functions that are used to explore a given path and list all the files within it.
# The exploreTree function recursively explores the specified path and appends any files it finds to a list.
# The listFiles function uses the exploreTree function to create a list of all files within the specified path and returns a Spark dataframe with metadata for each file.


from pyspark.sql.types import *
from pyspark.sql.functions import *

def exploreTree(blobFiles,path):
  """
  Recursively explores the specified path and appends any files it finds to a list.

  Parameters:
  blobFiles (list): list of file paths
  path (str): path to explore

  Returns:
  list: updated list of file paths
  """
  for ent in dbutils.fs.ls(path):
    if (ent.path[-1:]) == "/":
      #print(f"exploring: {ent.path}")
      exploreTree(blobFiles,ent.path)
    else:
      #print(ent.path)
      blobFiles.append(ent.path)
  return blobFiles

def listFiles(path):
  """
  Uses the exploreTree function to create a list of all files within the specified path and returns
  a Spark dataframe with metadata for each file.

  Parameters:
  path (str): path to explore for files

  Returns:
  DataFrame: Spark dataframe with metadata for each file.
  """
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
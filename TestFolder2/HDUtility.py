# The overall code takes in a file path and produces a dataframe with information about the files
# in the path, including the file type, entity, and file name.

from pyspark.sql.types import *
from pyspark.sql.functions import *


def exploreTree(blobFiles, path):
  """Recursively explores a directory tree and adds all file paths to a list"""
  for ent in dbutils.fs.ls(path):
    if (ent.path[-1:]) == "/":
      exploreTree(blobFiles, ent.path)
    else:
      blobFiles.append(ent.path)
  return blobFiles


def listFiles(path):
  """Takes in a file path and returns a dataframe with information about the files in the path"""
  blobFiles = []
  exploreTree(blobFiles, path)

  blobdf = spark.createDataFrame(blobFiles, StringType())
  blobdf = (blobdf
            .withColumnRenamed("value", "filePath")
            .withColumn("fileType", regexp_extract(col("filePath"), "\.(.*)", 1))
            .withColumn("entity", split("filePath", "/")[4])
            .withColumn("entitylvl2", split("filePath", "/")[5])
            .withColumn("entitylvl3", split("filePath", "/")[6])
            .withColumn("entitylvl4", split("filePath", "/")[7])
            .withColumn("fileName", regexp_extract(split("filePath", "/")[size(split("filePath", "/")) - 1], "(.*)\.", 1))
            )
  blobdf = blobdf.filter("fileType != ''")

  return blobdf
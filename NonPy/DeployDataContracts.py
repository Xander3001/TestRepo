# Databricks notebook source
"""
This notebook deploys changes to data contract configurations by loading all JSON files from the contract 
configuration directory, including the `Contracts`, `Transforms`, `EnvironmentConfig`, `global.jsonc` and `EntitySchemas` 
folders. It then merges this new data with the existing data from the ContractStore.TBL_DPLY_JSON table, overwriting 
any data that exists for the current branch, and saving the merged data to the ContractStore.TBL_DPLY_JSON table.
"""

# Import necessary libraries
import json
import requests
from datetime import datetime
from pyspark.sql.functions import col, input_file_name, lit

# Load branch information and credentials
with tracer.start_as_current_span("branch-info") as span:
    ctx = json.loads(
        dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson()
    )

    # Extract relevant information from context
    notebook_path = ctx["extraContext"]["notebook_path"]
    repo_path = "/".join(notebook_path.split("/")[:4])
    api_url = ctx["extraContext"]["api_url"]
    api_token = ctx["extraContext"]["api_token"]

    # Query the workspace API for the current branch
    repo_dir_data = requests.get(
        f"{api_url}/api/2.0/workspace/get-status",
        headers={"Authorization": f"Bearer {api_token}"},
        json={"path": repo_path},
    ).json()
    repo_id = repo_dir_data["object_id"]
    repo_data = requests.get(
        f"{api_url}/api/2.0/repos/{repo_id}",
        headers={"Authorization": f"Bearer {api_token}"},
    ).json()

    # Extract branch information and log it as an event
    branch = repo_data.get("branch")
    span.add_event(
        f"Deploy Data Contracts is operating on Branch: {branch}",
        attributes={
            "repo_path": repo_data.get("path"),
            "repo_url": repo_data.get("url"),
            "branch": branch,
        },
    )

# Load existing contracts from database
with tracer.start_as_current_span("load-existing") as span:
    # Load connection details from secrets
    contract_store_sql_server = dbutils.secrets.get(scope="hydr8v3-scope", key="sql-server-name")
    hydr8v3_db = dbutils.secrets.get(scope="hydr8v3-scope", key="hydr8v3-db-name")
    dbx_sp_client_id = dbutils.secrets.get(scope="hydr8v3-scope", key="dbx-sp-client-id")
    dbx_sp_client_secret = dbutils.secrets.get(scope="hydr8v3-scope", key="dbx-sp-client-secret")
    sql_server_url = f"jdbc:sqlserver://{contract_store_sql_server}.database.windows.net:1433;database={hydr8v3_db};AADSecurePrincipalId={dbx_sp_client_id};AADSecurePrincipalSecret={dbx_sp_client_secret};encrypt=true;trustServerCertificate=false;hostNameInCertificate=*.database.windows.net;loginTimeout=30;authentication=ActiveDirectoryServicePrincipal"
    contracts_table = "ContractStore.TBL_DPLY_JSON"
    driver = "com.microsoft.sqlserver.jdbc.SQLServerDriver"

    # Load existing contracts from database
    db_contracts_df = (
        spark.read.format("jdbc")
        .option("driver", driver)
        .option("url", sql_server_url)
        .option("dbtable", contracts_table)
        .load()
    )

    # Log event for successfully loading contracts from the database
    span.add_event(
        f"Loaded contracts from the DB",
        attributes={
            "loadedRecords": db_contracts_df.count(),
            "dbServer": contract_store_sql_server,
            "db": hydr8v3_db,
        },
    )

# Load new contracts from contract configuration directory
with tracer.start_as_current_span("load-new") as span:
    json_contracts_df = (
        spark.read.option("recursiveFileLookup", "true")
        .text(
            "file:/Workspace/Repos/hydr8v3/hydr8-data-contracts/config/Contracts/*",
            wholetext=True,
        )
        .withColumn("fileType", lit("contract"))
    )
    json_transforms_df = (
        spark.read.option("recursiveFileLookup", "true")
        .text(
            "file:/Workspace/Repos/hydr8v3/hydr8-data-contracts/config/Transforms/*",
            wholetext=True,
        )
        .withColumn("fileType", lit("transform"))
    )
    json_env_config_df = (
        spark.read.option("recursiveFileLookup", "true")
        .text(
            "file:/Workspace/Repos/hydr8v3/hydr8-data-contracts/config/EnvironmentConfig/*",
            wholetext=True,
        )
        .withColumn("fileType", lit("envConfig"))
    )
    json_global_config_df = (
        spark.read.option("recursiveFileLookup", "true")
        .text(
            "file:/Workspace/Repos/hydr8v3/hydr8-data-contracts/config/global.jsonc",
            wholetext=True,
        )
        .withColumn("fileType", lit("globalConfig"))
    )
    json_contract_entity_schema = (
        spark.read.option("recursiveFileLookup", "true")
        .text(
            "file:/Workspace/Repos/hydr8v3/hydr8-data-contracts/config/EntitySchemas/*",
            wholetext=True,
        )
        .withColumn("fileType", lit("entitySchema"))
    )

    # Combine all loaded JSON files into a single dataframe and format it for updating the database
    collated_df = (
        json_contracts_df.unionByName(json_transforms_df)
        .unionByName(json_env_config_df)
        .unionByName(json_global_config_df)
        .unionByName(json_contract_entity_schema)
    )
    conformed_new_json_df = (
        collated_df.withColumnRenamed("value", "jsonString")
        .withColumn("filePath", input_file_name())
        .withColumn("branch", lit(branch))
        .withColumn("dateLoaded", lit(datetime.now()))
    )

    # Log event for successfully loading new contracts from the contract configuration directory
    span.add_event(
        "Loaded new contracts from JSON",
        attributes={"loadedFiles": json_contracts_df.count()},
    )

# Overwrite database with updated dataframe
with_old_contracts_for_branch_removed_df = db_contracts_df.filter(
    col("branch") != lit(branch)
)
with_new_contracts_added = with_old_contracts_for_branch_removed_df.unionByName(
    conformed_new_json_df
)

with tracer.start_as_current_span("overwrite-with-updated") as span:
    with_new_contracts_added.write.format("jdbc").option("driver", driver).option(
        "url", sql_server_url
    ).option("dbtable", contracts_table).mode("overwrite").option(
        "truncate", "true"
    ).save()
    
    # Log event for successfully updating the database
    span.add_event(
        "Loaded new records from JSON",
        attributes={"recordsWritten": with_new_contracts_added.count()},
    )
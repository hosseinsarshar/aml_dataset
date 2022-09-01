import os
import uuid
import argparse
from datetime import datetime
from pyspark.sql import SparkSession
from databricks.feature_store import FeatureStoreClient
from databricks.feature_store.client import FeatureStoreClient


def populate_environ():
    parser = argparse.ArgumentParser(description='Process arguments passed to script')

    # Arguments that are related to your application
    parser.add_argument("--ground-truth-tbl-name", type=str, dest='ground_truth_tbl_name', default='ground_truth', help="ground-truth-tbl-name")
    
    (args, extra_args) = parser.parse_known_args()
    return args, extra_args

if __name__ == '__main__':
    spark = SparkSession.builder.getOrCreate()
    args, extra_args = populate_environ()

    spark.sql('create schema if not exists ft_experiments')

    spark.sql('use database ft_experiments')

    spark.sql('DROP TABLE IF EXISTS customers')

    spark.sql("""
    CREATE TABLE if not exists customers (
     id INT,
     age INT,
     salary INT
     )""")

    import random
    from random import choice

    dic_customer = {}
    for id in range(0, 1000):
      dic_customer[id] = {
        "age": random.randint(20, 100),
        'salary': choice([x * 10000 for x in range(1, 20)])
      }
    
    dic_customer
    li_records = []
    for customer_id in dic_customer:
      li_records.append(f"({customer_id}, {dic_customer[customer_id]['age']}, {dic_customer[customer_id]['salary']})")

    spark.sql(f"""
      insert into customers values 
      {','.join(li_records)}
     """)

    spark.sql('DROP TABLE IF EXISTS network')

    spark.sql("""
      create table if not exists network (
        network_id INT,
        failure boolean
      )
     """)

    import random
    from random import choice

    dic_network = {}
    for id in range(0, 1000):
      dic_network[id] = {
        'failure': choice([True, False])
      }

    li_records = []
    for net_id in dic_network:
      li_records.append(f"({net_id}, {dic_network[net_id]['failure']})")

    spark.sql(f"""
      insert into table network values 
      {','.join(li_records)}
     """)

    spark.sql(f'DROP TABLE IF EXISTS {args.ground_truth_tbl_name}')

    spark.sql(f"""
      create table {args.ground_truth_tbl_name} (
        customer int,
        network int,
        label boolean
      )
     """)

    li_ground_truth = []
    network_min = min(dic_network.keys())
    network_max = max(dic_network.keys())

    customer_min = min(dic_customer.keys())
    customer_max = max(dic_customer.keys())

    for id in range(0, 1000):
      customer = random.randint(customer_min, customer_max)
      network = random.randint(network_min, network_max)
      label = choice([True, False])
    
      li_ground_truth.append(f"({customer}, {network}, {label})")
    
    spark.sql(f"""
      insert into table {args.ground_truth_tbl_name} values 
      {','.join(li_ground_truth)}
     """)

    fs = FeatureStoreClient()

    fs.register_table(
        delta_table='ft_experiments.customers',
        primary_keys='id',
        description='Customer features'
    )

    fs.register_table(
        delta_table='ft_experiments.network',
        primary_keys='network_id',
        description='Network features'
    )


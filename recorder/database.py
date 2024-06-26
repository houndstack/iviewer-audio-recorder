import os
import json
import requests
import pandas as pd
import pymysql
from IPython.display import display, HTML

from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, MetaData


image_id = "test"
db_name = f'{image_id}.db'
DATABASE_DIR = "./databases/"
db_url = f"sqlite:///{os.path.join(DATABASE_DIR, db_name)}"
engine = create_engine(db_url)
metadata = MetaData()
metadata.reflect(bind=engine)

tables = metadata.tables
print(tables)
# Print table names
for table in tables:
    df = pd.read_sql_table(table, engine)
    df.size
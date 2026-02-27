
import unittest
import sys
import os
import pandas as pd
from unittest.mock import MagicMock

# Mock AWS Glue Libraries BEFORE import
sys.modules["awsglue"] = MagicMock()
sys.modules["awsglue.utils"] = MagicMock()
sys.modules["awsglue.context"] = MagicMock()
sys.modules["awsglue.job"] = MagicMock()
sys.modules["awsglue.transforms"] = MagicMock()

# Mock tiktoken/textstat if not present locally
try:
    import tiktoken
    import textstat
except ImportError:
    sys.modules["tiktoken"] = MagicMock()
    sys.modules["textstat"] = MagicMock()
    
    mock_enc = MagicMock()
    mock_enc.encode.return_value = [1, 2, 3] # Dummy tokens
    sys.modules["tiktoken"].get_encoding.return_value = mock_enc
    sys.modules["textstat"].flesch_reading_ease.return_value = 50.0

# Initialize Spark for Tests
try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType
except ImportError:
    print("PySpark not found, skipping tests")
    sys.exit(0)

class TestMetricCalculations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder \
            .master("local[1]") \
            .appName("UnitTest") \
            .config("spark.sql.session.timeZone", "UTC") \
            .getOrCreate()
            
    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
        
    def test_rejection_logic(self):
        # Create a sample DF
        data = [
            ("valid", "This is a normal sentence. It has enough length.", "s3://file1"),
            ("short", "Too short", "s3://file2"),
            ("html", "<html><div>" * 20 + "spam", "s3://file3"),
        ]
        schema = "id string, text string, file_path string"
        df = self.spark.createDataFrame(data, schema)
        
        # --- REPLICATE LOGIC FROM SCRIPT (Tier 1) ---
        df = df.withColumn("byte_length", F.length(F.encode("text", "utf-8")))
        df = df.withColumn("char_length", F.length("text"))
        
        # HTML Density
        df = df.withColumn("no_tags_len", F.length(F.regexp_replace("text", r"<[^>]+>", "")))
        df = df.withColumn("html_tag_density", (F.col("char_length") - F.col("no_tags_len")) / F.col("char_length"))
        
        # Tier 1 Rejection
        c1 = (F.col("byte_length") < 50)
        c4 = (F.col("html_tag_density") > 0.05)
        
        df = df.withColumn("t1_rejected", c1 | c4)
        
        # Assertions
        res = {r['id']: r for r in df.collect()}
        
        self.assertFalse(res['valid']['t1_rejected'])
        self.assertTrue(res['short']['t1_rejected']) # Length
        self.assertTrue(res['html']['t1_rejected']) # HTML

if __name__ == "__main__":
    unittest.main()

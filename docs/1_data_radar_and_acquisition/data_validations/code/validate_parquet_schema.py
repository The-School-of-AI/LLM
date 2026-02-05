"""
S3 Parquet Schema Validation Tool
Validates random records from parquet files against defined schema
"""

import boto3
import pandas as pd
import pyarrow.parquet as pq
import random
from datetime import datetime
from typing import Dict, List, Any
import io
import hashlib
import json

# AWS Configuration
AWS_ACCESS_KEY = "******"
AWS_SECRET_KEY = "******"
BUCKET_NAME = "t1-dataacquisition-datasets"
PREFIX = "datasets_prod/sangraha/hin/"

# Expected Schema Definition
SCHEMA_DEFINITION = {
    "id": {"type": "string", "description": "Unique record identifier"},
    "hash": {"type": "string", "description": "SHA-256 hash of text content (for deduplication)"},
    "dataset": {"type": "string", "description": "Source dataset name"},
    "domain": {"type": "string", "description": "Content domain (web, literature, education)"},
    "source": {"type": "string/null", "description": "Source identifier (for Dolma)"},
    "text": {"type": "string", "description": "Main text content"},
    "language": {"type": "string", "description": "Full language name"},
    "metadata": {"type": "dict", "description": "Additional dataset-specific fields"},
    "added": {"type": "string/null", "description": "ISO timestamp when added"},
    "created": {"type": "string/null", "description": "ISO timestamp of creation"},
    "version": {"type": "string/null", "description": "Dataset version"}
}


class ParquetSchemaValidator:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY
        )
        self.validation_results = []
        
    def list_parquet_files(self) -> List[str]:
        """List all parquet files in the S3 path"""
        print(f"Listing parquet files in s3://{BUCKET_NAME}/{PREFIX}")
        
        paginator = self.s3_client.get_paginator('list_objects_v2')
        parquet_files = []
        
        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=PREFIX):
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].endswith('.parquet'):
                        parquet_files.append(obj['Key'])
        
        print(f"Found {len(parquet_files)} parquet files")
        return parquet_files
    
    def read_parquet_from_s3(self, key: str) -> pd.DataFrame:
        """Read parquet file from S3"""
        print(f"Reading: s3://{BUCKET_NAME}/{key}")
        
        obj = self.s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        parquet_data = obj['Body'].read()
        
        # Read parquet from bytes
        return pd.read_parquet(io.BytesIO(parquet_data))
    
    def validate_field(self, field_name: str, value: Any, expected_type: str) -> Dict[str, Any]:
        """Validate a single field against expected type"""
        result = {
            "field": field_name,
            "expected_type": expected_type,
            "actual_type": type(value).__name__,
            "value_preview": str(value)[:100] if value is not None else "None",
            "is_valid": False,
            "issues": []
        }
        
        # Handle None/null values
        if value is None or pd.isna(value):
            if "null" in expected_type:
                result["is_valid"] = True
            else:
                result["issues"].append(f"Value is null but type '{expected_type}' doesn't allow null")
            return result
        
        # Type validation
        if expected_type == "string" or expected_type == "string/null":
            if isinstance(value, str):
                result["is_valid"] = True
            else:
                result["issues"].append(f"Expected string, got {type(value).__name__}")
        
        elif expected_type == "dict":
            if isinstance(value, dict):
                result["is_valid"] = True
            elif isinstance(value, str):
                try:
                    json.loads(value)
                    result["is_valid"] = True
                    result["issues"].append("Dict stored as JSON string")
                except:
                    result["issues"].append("Expected dict, got unparseable string")
            else:
                result["issues"].append(f"Expected dict, got {type(value).__name__}")
        
        return result
    
    def validate_record(self, record: Dict[str, Any], file_name: str, record_index: int) -> Dict[str, Any]:
        """Validate a single record against schema"""
        validation = {
            "file": file_name,
            "record_index": record_index,
            "timestamp": datetime.now().isoformat(),
            "fields_validation": [],
            "missing_fields": [],
            "extra_fields": [],
            "overall_valid": True
        }
        
        # Check for missing required fields
        expected_fields = set(SCHEMA_DEFINITION.keys())
        actual_fields = set(record.keys())
        
        missing = expected_fields - actual_fields
        extra = actual_fields - expected_fields
        
        validation["missing_fields"] = list(missing)
        validation["extra_fields"] = list(extra)
        
        if missing:
            validation["overall_valid"] = False
        
        # Validate each field
        for field_name, field_spec in SCHEMA_DEFINITION.items():
            if field_name in record:
                field_result = self.validate_field(
                    field_name,
                    record[field_name],
                    field_spec["type"]
                )
                validation["fields_validation"].append(field_result)
                
                if not field_result["is_valid"]:
                    validation["overall_valid"] = False
        
        validation["record_data"] = {k: str(v)[:200] if v is not None else None 
                                     for k, v in record.items()}
        
        return validation
    
    def sample_and_validate(self, num_files: int = 7, total_records: int = 25):
        """Sample random records from random files and validate"""
        # Get all parquet files
        all_files = self.list_parquet_files()
        
        if not all_files:
            print("No parquet files found!")
            return
        
        # Sample random files
        num_files = min(num_files, len(all_files))
        selected_files = random.sample(all_files, num_files)
        
        print(f"\nSelected {num_files} random files for validation")
        print(f"Target: ~{total_records} total records\n")
        
        records_per_file = max(1, total_records // num_files)
        
        for file_key in selected_files:
            try:
                # Read parquet file
                df = self.read_parquet_from_s3(file_key)
                
                if len(df) == 0:
                    print(f"  Warning: File is empty, skipping")
                    continue
                
                # Sample random records
                sample_size = min(records_per_file, len(df))
                sample_indices = random.sample(range(len(df)), sample_size)
                
                print(f"  Sampling {sample_size} records from {len(df)} total records")
                
                for idx in sample_indices:
                    record = df.iloc[idx].to_dict()
                    validation = self.validate_record(record, file_key, idx)
                    self.validation_results.append(validation)
                
                print(f"  ✓ Validated {sample_size} records\n")
                
            except Exception as e:
                print(f"  ✗ Error processing file: {str(e)}\n")
                self.validation_results.append({
                    "file": file_key,
                    "error": str(e),
                    "overall_valid": False
                })
    
    def generate_markdown_report(self, output_file: str = "validation_result.md"):
        """Generate detailed markdown report"""
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("# Parquet Schema Validation Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**S3 Path:** s3://{BUCKET_NAME}/{PREFIX}\n\n")
            f.write(f"**Total Records Validated:** {len(self.validation_results)}\n\n")
            
            # Summary Statistics
            valid_count = sum(1 for r in self.validation_results if r.get("overall_valid", False))
            invalid_count = len(self.validation_results) - valid_count
            
            f.write("## Validation Summary\n\n")
            f.write(f"- ✅ **Valid Records:** {valid_count}\n")
            f.write(f"- ❌ **Invalid Records:** {invalid_count}\n")
            f.write(f"- 📊 **Success Rate:** {(valid_count/len(self.validation_results)*100):.1f}%\n\n")
            
            # Schema Definition
            f.write("## Expected Schema\n\n")
            f.write("| Field | Type | Description |\n")
            f.write("|-------|------|-------------|\n")
            for field, spec in SCHEMA_DEFINITION.items():
                f.write(f"| {field} | {spec['type']} | {spec['description']} |\n")
            f.write("\n")
            
            # Detailed Results
            f.write("## Detailed Validation Results\n\n")
            
            for i, result in enumerate(self.validation_results, 1):
                if "error" in result:
                    f.write(f"### Record {i} - ERROR\n\n")
                    f.write(f"**File:** `{result['file']}`\n\n")
                    f.write(f"**Error:** {result['error']}\n\n")
                    f.write("---\n\n")
                    continue
                
                status = "✅ VALID" if result["overall_valid"] else "❌ INVALID"
                f.write(f"### Record {i} - {status}\n\n")
                f.write(f"**File:** `{result['file']}`\n\n")
                f.write(f"**Record Index:** {result['record_index']}\n\n")
                
                # Issues Summary
                if result["missing_fields"]:
                    f.write(f"**❌ Missing Fields:** {', '.join(result['missing_fields'])}\n\n")
                
                if result["extra_fields"]:
                    f.write(f"**⚠️ Extra Fields:** {', '.join(result['extra_fields'])}\n\n")
                
                # Field Validation Details
                f.write("#### Field Validation\n\n")
                f.write("| Field | Expected Type | Actual Type | Valid | Issues |\n")
                f.write("|-------|---------------|-------------|-------|--------|\n")
                
                for fv in result.get("fields_validation", []):
                    valid_icon = "✅" if fv["is_valid"] else "❌"
                    issues = ", ".join(fv["issues"]) if fv["issues"] else "-"
                    f.write(f"| {fv['field']} | {fv['expected_type']} | {fv['actual_type']} | {valid_icon} | {issues} |\n")
                
                f.write("\n")
                
                # Record Data Preview
                f.write("#### Record Data Preview\n\n")
                f.write("```json\n")
                f.write(json.dumps(result.get("record_data", {}), indent=2, ensure_ascii=False))
                f.write("\n```\n\n")
                
                f.write("---\n\n")
        
        print(f"✓ Report generated: {output_file}")


def main():
    print("=" * 80)
    print("Parquet Schema Validation Tool")
    print("=" * 80)
    print()
    
    validator = ParquetSchemaValidator()
    
    # Sample and validate records
    validator.sample_and_validate(num_files=7, total_records=25)
    
    # Generate report
    validator.generate_markdown_report("validation_result.md")
    
    print("\n" + "=" * 80)
    print("Validation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

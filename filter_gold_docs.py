#!/usr/bin/env python3
"""
Script to filter CSV file by checking if gold_docs entries are present in top 10 ranked columns.
If any gold entry is found in columns 1st through 10th, the row is dropped.
"""

import pandas as pd
import ast
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_parse_list(value):
    """
    Safely parse a string representation of a list into actual list
    """
    if pd.isna(value) or value == '':
        return []
    
    # Handle string representation of list
    if isinstance(value, str):
        try:
            # Try to parse as literal list
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
            else:
                return [str(parsed)]  # Convert single value to list
        except (ValueError, SyntaxError):
            # If parsing fails, treat as single string value
            return [value.strip()]
    
    # If already a list
    if isinstance(value, list):
        return value
    
    # Convert other types to string and wrap in list
    return [str(value)]

def filter_csv_by_gold_presence(input_file, output_file):
    """
    Filter CSV file by removing rows where any gold_docs entry is present in top 10 columns
    """
    logger.info(f"Loading CSV file: {input_file}")
    
    try:
        # Load the CSV file
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} rows from CSV")
        
        # Define the ranking columns (1st through 10th)
        ranking_columns = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th']
        
        # Check if all required columns exist
        missing_cols = [col for col in ranking_columns + ['gold_docs'] if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns in CSV: {missing_cols}")
            return False
        
        logger.info(f"Found columns: {list(df.columns)}")
        logger.info(f"Processing gold_docs against ranking columns: {ranking_columns}")
        
        # Track filtering statistics
        total_rows = len(df)
        rows_to_keep = []
        
        for idx, row in df.iterrows():
            # Parse gold_docs
            gold_docs = safe_parse_list(row['gold_docs'])
            
            # Get values from ranking columns (handle NaN values)
            ranking_values = []
            for col in ranking_columns:
                val = row[col]
                if pd.notna(val) and val != '':
                    ranking_values.append(str(val))
            
            # Check if any gold doc is present in ranking columns
            gold_found = False
            for gold_doc in gold_docs:
                gold_doc_str = str(gold_doc).strip()
                if gold_doc_str in ranking_values:
                    gold_found = True
                    break
            
            # Keep row only if NO gold docs are found in ranking columns
            if not gold_found:
                rows_to_keep.append(idx)
            
            # Log progress every 100 rows
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{total_rows} rows...")
        
        # Create filtered dataframe
        filtered_df = df.iloc[rows_to_keep].copy()
        
        # Statistics
        removed_rows = total_rows - len(filtered_df)
        logger.info(f"\n📊 FILTERING RESULTS:")
        logger.info(f"   Original rows: {total_rows:,}")
        logger.info(f"   Rows with gold docs in top 10: {removed_rows:,} ({removed_rows/total_rows*100:.1f}%)")
        logger.info(f"   Filtered rows (kept): {len(filtered_df):,} ({len(filtered_df)/total_rows*100:.1f}%)")
        
        # Save filtered CSV
        logger.info(f"Saving filtered CSV to: {output_file}")
        filtered_df.to_csv(output_file, index=False)
        logger.info(f"✅ Successfully saved {len(filtered_df):,} rows to {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        return False

def analyze_sample_rows(input_file, num_samples=5):
    """
    Analyze a few sample rows to understand the data structure
    """
    logger.info(f"Analyzing sample rows from {input_file}")
    
    try:
        df = pd.read_csv(input_file)
        
        ranking_columns = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th']
        
        logger.info(f"\n🔍 SAMPLE ANALYSIS (first {num_samples} rows):")
        
        for idx in range(min(num_samples, len(df))):
            row = df.iloc[idx]
            logger.info(f"\n--- Row {idx + 1} ---")
            
            # Parse gold_docs
            gold_docs = safe_parse_list(row['gold_docs'])
            logger.info(f"Gold docs: {gold_docs}")
            
            # Check ranking columns
            found_matches = []
            for col in ranking_columns:
                val = row[col]
                if pd.notna(val) and val != '':
                    val_str = str(val)
                    # Check if this value matches any gold doc
                    for gold_doc in gold_docs:
                        if str(gold_doc).strip() == val_str.strip():
                            found_matches.append(f"{col}: {val_str}")
            
            if found_matches:
                logger.info(f"✅ Gold matches found: {found_matches}")
                logger.info("   → This row would be DROPPED")
            else:
                logger.info("❌ No gold matches found")
                logger.info("   → This row would be KEPT")
        
    except Exception as e:
        logger.error(f"Error analyzing samples: {e}")

def main():
    # File paths
    input_file = "/shared/khoja/CogComp/output/dense_sparse_average_results.csv"
    output_file = "/shared/khoja/CogComp/output/dense_sparse_average_results_filtered.csv"
    
    # Check if input file exists
    if not Path(input_file).exists():
        logger.error(f"Input file not found: {input_file}")
        return
    
    # Analyze sample rows first
    analyze_sample_rows(input_file, num_samples=3)
    
    print("\n" + "="*60)
    response = input("Continue with filtering? (y/n): ")
    if response.lower() != 'y':
        logger.info("Filtering cancelled by user")
        return
    
    # Perform filtering
    success = filter_csv_by_gold_presence(input_file, output_file)
    
    if success:
        logger.info(f"\n🎉 Filtering completed!")
        logger.info(f"📁 Original file: {input_file}")
        logger.info(f"📁 Filtered file: {output_file}")
    else:
        logger.error("❌ Filtering failed!")

if __name__ == "__main__":
    main()
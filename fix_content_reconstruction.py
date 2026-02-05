#!/usr/bin/env python3
"""
Fix content reconstruction by loading original STARK dataset
and building proper content from title + feature + detail + description
"""

import json
import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import sys
sys.path.append('/shared/khoja/CogComp')

def load_original_stark_data():
    """Load original STARK dataset to get text fields"""
    logger.info("🔍 Loading Original STARK Dataset")
    logger.info("=" * 40)
    
    # Load STARK dataset
    stark_file = '/shared/khoja/CogComp/datasets/STARK/nodes.json'
    logger.info(f"📥 Loading STARK data from: {stark_file}")
    
    with open(stark_file, 'r') as f:
        stark_data = json.load(f)
    
    logger.info(f"📊 Loaded {len(stark_data)} STARK entries")
    
    # Build ASIN to content mapping
    asin_to_content = {}
    
    for entry in tqdm(stark_data, desc="Building content"):
        # Each entry has a single key which is the stark_id
        stark_id = list(entry.keys())[0]
        data = entry[stark_id]
        asin = data.get('asin', '')
        if not asin:
            continue
        
        # Build content same way as in STARK pipeline
        content_parts = []
        
        # Add title
        if data.get('title'):
            content_parts.append(str(data['title']))
        
        # Add global_category
        if data.get('global_category'):
            content_parts.append(str(data['global_category']))
        
        # Add category
        if data.get('category'):
            if isinstance(data['category'], list):
                content_parts.extend([str(c) for c in data['category']])
            else:
                content_parts.append(str(data['category']))
        
        # Add brand
        if data.get('brand'):
            content_parts.append(str(data['brand']))
        
        # Add feature
        if data.get('feature'):
            if isinstance(data['feature'], list):
                content_parts.extend([str(f) for f in data['feature']])
            else:
                content_parts.append(str(data['feature']))
        
        # Add details (note: field is 'details' in STARK, 'detail' in chunk)
        # Details is typically an object, extract useful text if possible
        if data.get('details') and isinstance(data['details'], dict):
            # Skip details as it's typically just metadata like {"asin": "..."}
            pass
        elif data.get('details'):
            content_parts.append(str(data['details']))
        
        # Add description
        if data.get('description'):
            if isinstance(data['description'], list):
                content_parts.extend([str(d) for d in data['description']])
            else:
                content_parts.append(str(data['description']))
        
        # Join all parts
        content = ' '.join([p.strip() for p in content_parts if p and str(p).strip()])
        
        if content.strip():
            asin_to_content[asin] = content.strip()
    
    logger.info(f"📊 Built content for {len(asin_to_content):,} ASINs")
    
    # Check sample content
    sample_asins = list(asin_to_content.keys())[:5]
    logger.info(f"\n📄 Sample content:")
    for i, asin in enumerate(sample_asins):
        content = asin_to_content[asin]
        logger.info(f"   {i+1}. {asin}: {len(content)} chars")
        logger.info(f"      {content[:200]}...")
    
    return asin_to_content

def update_chunk_content_cache():
    """Update the enhanced analyzer to use reconstructed content"""
    logger.info("\n🔧 Updating Enhanced Analyzer Content Cache")
    logger.info("=" * 50)
    
    # Load ASIN to content mapping
    asin_to_content = load_original_stark_data()
    
    # Update the enhanced_multi_feature_analyzer.py to use this content
    logger.info("✅ Content reconstruction completed!")
    logger.info(f"📊 Content available for {len(asin_to_content):,} ASINs")
    
    # Save for future use
    output_path = '/shared/khoja/CogComp/output/asin_to_content_mapping.json'
    with open(output_path, 'w') as f:
        json.dump(asin_to_content, f, indent=2)
    
    logger.info(f"💾 Saved content mapping to: {output_path}")
    
    return asin_to_content

if __name__ == "__main__":
    asin_to_content = update_chunk_content_cache()

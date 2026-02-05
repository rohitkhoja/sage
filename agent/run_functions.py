#!/usr/bin/env python3
"""
Direct function runner - load once, call functions directly
"""

import sys
import os
from loguru import logger

# Add the agent directory to the path
sys.path.append('/shared/khoja/CogComp/agent')

from mag_agent import MAGAgent
from flexible_mag_agent import FlexibleMAGAgent

# Global variables to keep loaded agents
mag_agent = None
flex_agent = None

def load_system():
    """Load the system once"""
    global mag_agent, flex_agent
    
    if mag_agent is not None:
        logger.info("✅ System already loaded")
        return True
    
    logger.info("🚀 Loading MAG Agent System...")
    
    try:
        # Load MAG Agent
        mag_agent = MAGAgent("/shared/khoja/CogComp/datasets/MAG/processed", 
                            "/shared/khoja/CogComp/output/mag_hnsw_indices")
        mag_agent.load_all()
        logger.info("✅ MAG Agent loaded")
        
        # Load Flexible Agent
        flex_agent = FlexibleMAGAgent("/shared/khoja/CogComp/datasets/MAG/processed", 
                                    "/shared/khoja/CogComp/output/mag_hnsw_indices")
        flex_agent.load_all()
        logger.info("✅ Flexible Agent loaded")
        
        logger.info("🎉 System loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load system: {e}")
        return False

# Load system on import
if load_system():
    logger.info("🎯 System ready! You can now call functions:")
    logger.info("  - mag_agent.search_papers_by_title('query', 5)")
    logger.info("  - mag_agent.search_authors_by_name('name', 3)")
    logger.info("  - mag_agent.get_papers_by_year_range(2010, 2020)")
    logger.info("  - flex_agent.solve_query('query', 'session')")
    logger.info("  - And many more...")
else:
    logger.error("❌ System failed to load")

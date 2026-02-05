#!/usr/bin/env python3
"""
Simple test runner for MAG Agent system
"""

import sys
import os
from pathlib import Path
from loguru import logger

# Add agent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def run_component_test(component_name: str, test_function):
    """Run a single component test"""
    logger.info(f"🧪 Testing {component_name}...")
    try:
        success = test_function()
        if success:
            logger.info(f"✅ {component_name} test passed")
            return True
        else:
            logger.error(f"❌ {component_name} test failed")
            return False
    except Exception as e:
        logger.error(f"❌ {component_name} test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run individual component tests"""
    logger.info("🧪 Running MAG Agent Component Tests")
    logger.info("=" * 50)
    
    test_results = {}
    
    # Test Graph Loader
    from graph_loader import main as test_graph_loader
    test_results['graph_loader'] = run_component_test("Graph Loader", test_graph_loader)
    
    # Test HNSW Manager
    from hnsw_manager import main as test_hnsw_manager
    test_results['hnsw_manager'] = run_component_test("HNSW Manager", test_hnsw_manager)
    
    # Test Traversal Utils
    from traversal_utils import main as test_traversal_utils
    test_results['traversal_utils'] = run_component_test("Traversal Utils", test_traversal_utils)
    
    # Test Query Orchestrator
    from query_orchestrator import main as test_query_orchestrator
    test_results['query_orchestrator'] = run_component_test("Query Orchestrator", test_query_orchestrator)
    
    # Test MAG Agent
    from mag_agent import main as test_mag_agent
    test_results['mag_agent'] = run_component_test("MAG Agent", test_mag_agent)
    
    # Test Flexible MAG Agent
    from flexible_mag_agent import main as test_flexible_mag_agent
    test_results['flexible_mag_agent'] = run_component_test("Flexible MAG Agent", test_flexible_mag_agent)
    
    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    for component, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{component}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} components passed")
    
    if passed == total:
        logger.info("🎉 All component tests passed!")
        return True
    else:
        logger.error("❌ Some component tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

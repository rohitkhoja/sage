#!/usr/bin/env python3
"""
Client for Persistent MAG Agent Server
Connects to the running server and provides interactive interface
"""

import sys
import os
import time
from loguru import logger

# Add the agent directory to the path
sys.path.append('/shared/khoja/CogComp/agent')

from persistent_server import PersistentMAGServer


class MAGClient:
    """Client for connecting to persistent server"""
    
    def __init__(self):
        self.server = None
        self.connected = False
    
    def connect(self):
        """Connect to the persistent server"""
        try:
            # Initialize server (this will load everything if not already loaded)
            self.server = PersistentMAGServer("/shared/khoja/CogComp/datasets/MAG/processed", 
                                            "/shared/khoja/CogComp/output/mag_hnsw_indices")
            
            # Load system if not already loaded
            if not self.server.is_loaded:
                logger.info("Loading system for the first time...")
                if not self.server.load_system():
                    logger.error("Failed to load system")
                    return False
            
            self.connected = True
            logger.info("✅ Connected to persistent server")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    def execute(self, function_call: str):
        """Execute a function on the server"""
        if not self.connected:
            return {"error": "Not connected to server"}
        
        return self.server.execute_function(function_call)
    
    def get_info(self):
        """Get server information"""
        if not self.connected:
            return {"error": "Not connected to server"}
        
        return self.server.get_system_info()
    
    def interactive_mode(self):
        """Run interactive mode"""
        print("🎯 MAG Agent Interactive Client")
        print("=" * 40)
        print("Type 'help' for available functions")
        print("Type 'info' for system information")
        print("Type 'quit' to exit")
        print("=" * 40)
        
        while True:
            try:
                command = input("\nMAG> ").strip()
                
                if command.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                elif command.lower() == 'help':
                    self._show_help()
                elif command.lower() == 'info':
                    self._show_info()
                elif command:
                    result = self.execute(command)
                    self._display_result(result)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _show_help(self):
        """Show help information"""
        info = self.get_info()
        if 'error' in info:
            print(f"❌ {info['error']}")
            return
        
        print("\n📋 Available Functions:")
        print("-" * 30)
        
        functions = info.get('available_functions', [])
        for func in functions[:20]:  # Show first 20 functions
            print(f"  {func}")
        
        if len(functions) > 20:
            print(f"  ... and {len(functions) - 20} more functions")
        
        print("\n💡 Examples:")
        print("  search_papers_by_title(\"machine learning\", 5)")
        print("  get_papers_by_year_range(2010, 2020)")
        print("  solve_query(\"papers about AI\", \"session1\")")
        print("  get_system_info()")
    
    def _show_info(self):
        """Show system information"""
        info = self.get_info()
        if 'error' in info:
            print(f"❌ {info['error']}")
            return
        
        print(f"\n📊 System Information:")
        print(f"  Loaded: {info.get('is_loaded', False)}")
        print(f"  Load time: {info.get('load_time', 0):.2f}s")
        print(f"  Uptime: {info.get('uptime', 0):.2f}s")
        print(f"  Connections: {info.get('connections', 0)}")
        
        if 'graph_stats' in info and info['graph_stats']:
            stats = info['graph_stats']
            print(f"  Graph: {stats.get('total_nodes', 0):,} nodes, {stats.get('total_edges', 0):,} edges")
        
        if 'hnsw_stats' in info and info['hnsw_stats']:
            stats = info['hnsw_stats']
            print(f"  HNSW: {stats.get('total_embeddings', 0):,} embeddings")
    
    def _display_result(self, result):
        """Display execution result"""
        if isinstance(result, dict) and 'error' in result:
            print(f"❌ {result['error']}")
        elif isinstance(result, list):
            print(f"✅ Found {len(result)} results")
            if len(result) <= 5:
                for i, item in enumerate(result):
                    print(f"  {i+1}. {item}")
            else:
                for i, item in enumerate(result[:3]):
                    print(f"  {i+1}. {item}")
                print(f"  ... and {len(result) - 3} more")
        elif isinstance(result, dict):
            print("✅ Result:")
            for key, value in result.items():
                if isinstance(value, list) and len(value) > 3:
                    print(f"  {key}: {len(value)} items")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"✅ {result}")


def main():
    """Main function"""
    client = MAGClient()
    
    if not client.connect():
        print("❌ Failed to connect to server")
        return 1
    
    client.interactive_mode()
    return 0


if __name__ == "__main__":
    exit(main())

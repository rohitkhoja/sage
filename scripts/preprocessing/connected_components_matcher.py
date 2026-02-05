"""
Connected Components Matcher
Find which connected component a key belongs to and count matches from a list of values
"""

import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path

class ConnectedComponentsMatcher:
    """Find connected components and count matches for given keys"""
    
    def __init__(self, connected_components_file: str):
        """
        Initialize with connected components data
        
        Args:
            connected_components_file: Path to JSON file containing connected components
        """
        self.connected_components_file = connected_components_file
        self.connected_components = []
        self.chunk_to_component_map = {} # chunk_id -> component_index
        
        self._load_connected_components()
        self._build_chunk_mapping()
    
    def _load_connected_components(self):
        """Load connected components from JSON file"""
        with open(self.connected_components_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract just the connected components list
        if isinstance(data, dict) and 'connected_components' in data:
            self.connected_components = data['connected_components']
        elif isinstance(data, list):
            self.connected_components = data
        else:
            raise ValueError("Invalid connected components format")
        
        print(f"Loaded {len(self.connected_components)} connected components")
    
    def _build_chunk_mapping(self):
        """Build mapping from chunk_id to component index for fast lookup"""
        for component_idx, component in enumerate(self.connected_components):
            for chunk_id in component:
                self.chunk_to_component_map[chunk_id] = component_idx
        
        print(f"Built mapping for {len(self.chunk_to_component_map)} chunk IDs")
    
    def extract_base_id(self, chunk_id: str) -> str:
        """
        Extract base ID from chunk ID by removing chunk suffix
        
        Example: 
        'ottaqa-2011_Caloundra_International_0-Slovenia_chunk_0_cd2655d48243' 
        -> 'ottaqa-2011_Caloundra_International_0-Slovenia'
        """
        # Split by '_chunk_' and take the first part
        if '_chunk_' in chunk_id:
            return chunk_id.split('_chunk_')[0]
        return chunk_id
    
    def find_matching_chunks(self, base_id: str) -> List[str]:
        """
        Find all chunk IDs that start with the given base_id
        
        Args:
            base_id: Base ID to search for (e.g., 'ottaqa-2011_Caloundra_International_0-Slovenia')
            
        Returns:
            List of matching chunk IDs
        """
        matching_chunks = []
        
        for chunk_id in self.chunk_to_component_map.keys():
            chunk_base_id = self.extract_base_id(chunk_id)
            if chunk_base_id == base_id:
                matching_chunks.append(chunk_id)
        
        return matching_chunks
    
    def find_component_for_key(self, key_id: str) -> Optional[Tuple[int, List[str]]]:
        """
        Find which connected component contains chunks matching the key_id
        
        Args:
            key_id: Key ID to search for (e.g., 'ottaqa-2011_Caloundra_International_0-Slovenia')
            
        Returns:
            Tuple of (component_index, component_list) or None if not found
        """
        # Find matching chunks
        matching_chunks = self.find_matching_chunks(key_id)
        
        if not matching_chunks:
            print(f"No chunks found for key: {key_id}")
            return None
        
        # Get the component index from the first matching chunk
        # (all chunks with same base_id should be in same component)
        first_chunk = matching_chunks[0]
        component_idx = self.chunk_to_component_map.get(first_chunk)
        
        if component_idx is None:
            print(f"Component not found for chunk: {first_chunk}")
            return None
        
        component = self.connected_components[component_idx]
        
        print(f"Key '{key_id}' found in component {component_idx} with {len(component)} total chunks")
        print(f"Matching chunks for this key: {matching_chunks}")
        
        return component_idx, component
    
    def count_matches_in_component(self, key_id: str, values_to_match: List[str]) -> Dict[str, any]:
        """
        Find component for key_id and count how many values from values_to_match are in that component
        
        Args:
            key_id: Key ID to search for
            values_to_match: List of base IDs to count matches for
            
        Returns:
            Dictionary with match results
        """
        # Find component for the key
        result = self.find_component_for_key(key_id)
        if result is None:
            return {
                'key_id': key_id,
                'component_found': False,
                'component_index': None,
                'component_size': 0,
                'values_to_match': values_to_match,
                'matches_found': [],
                'match_count': 0,
                'match_details': {}
            }
        
        component_idx, component = result
        
        # Extract base IDs from all chunks in the component
        component_base_ids = set()
        for chunk_id in component:
            base_id = self.extract_base_id(chunk_id)
            component_base_ids.add(base_id)
        
        # Find matches
        matches_found = []
        match_details = {}
        
        for value in values_to_match:
            if value in component_base_ids:
                matches_found.append(value)
                # Find all chunks for this base_id in the component
                matching_chunks = [chunk for chunk in component if self.extract_base_id(chunk) == value]
                match_details[value] = matching_chunks
        
        return {
            'key_id': key_id,
            'component_found': True,
            'component_index': component_idx,
            'component_size': len(component),
            'values_to_match': values_to_match,
            'matches_found': matches_found,
            'match_count': len(matches_found),
            'match_details': match_details,
            'component_base_ids': sorted(list(component_base_ids))
        }
    
    def print_match_results(self, results: Dict[str, any]):
        """Print formatted match results"""
        print("=" * 60)
        print("MATCH RESULTS")
        print("=" * 60)
        print(f"Key ID: {results['key_id']}")
        print(f"Component Found: {results['component_found']}")
        
        if results['component_found']:
            print(f"Component Index: {results['component_index']}")
            print(f"Component Size: {results['component_size']} chunks")
            print(f"Values to Match: {len(results['values_to_match'])}")
            print(f"Matches Found: {results['match_count']}")
            print()
            
            if results['matches_found']:
                print("MATCHED VALUES:")
                for i, match in enumerate(results['matches_found'], 1):
                    chunks = results['match_details'][match]
                    print(f" {i}. {match}")
                    print(f" Chunks: {len(chunks)}")
                    for chunk in chunks:
                        print(f" - {chunk}")
                print()
            
            print("ALL BASE IDs IN COMPONENT:")
            for base_id in results['component_base_ids']:
                status = " MATCH" if base_id in results['matches_found'] else ""
                print(f" - {base_id} {status}")
        
        print("=" * 60)


def main():
    """Example usage"""
    # File paths
    connected_components_file = "/shared/khoja/CogComp/output/connected_components.json"
    
    # Initialize matcher
    matcher = ConnectedComponentsMatcher(connected_components_file)
    
    # Example 1: Single key with multiple values to match
    key_id = "ottaqa-2011_Caloundra_International_0-Slovenia"
    values_to_match = [
        "ottaqa-International_Children_s_Games_1-Slovenia",
        "ottaqa-Joint_issue_32-Slovenia",
        "ottaqa-Asian_Highway_Network_1-Indonesia", # This won't match (different component)
        "ottaqa-Boss__TV_series__0-Indonesia" # This won't match (different component)
    ]
    
    print("Example 1: Finding matches for Slovenia key")
    results = matcher.count_matches_in_component(key_id, values_to_match)
    matcher.print_match_results(results)
    
    print("\n" + "="*80 + "\n")
    
    # Example 2: Different key
    key_id2 = "ottaqa-Asian_Highway_Network_1-Indonesia"
    values_to_match2 = [
        "ottaqa-Boss__TV_series__0-Indonesia",
        "ottaqa-List_of_coffee_varieties_0-Indonesia",
        "ottaqa-List_of_diplomatic_missions_in_Malaysia_0-Indonesia",
        "ottaqa-2011_Caloundra_International_0-Slovenia" # This won't match (different component)
    ]
    
    print("Example 2: Finding matches for Indonesia key")
    results2 = matcher.count_matches_in_component(key_id2, values_to_match2)
    matcher.print_match_results(results2)
    
    # Example 3: Just get component info for a key
    print("\n" + "="*80 + "\n")
    print("Example 3: Just finding component for a key")
    component_info = matcher.find_component_for_key("ottaqa-Ontario_4-Kingston__Ontario")
    if component_info:
        comp_idx, comp_list = component_info
        print(f"Component {comp_idx} contains {len(comp_list)} chunks:")
        for chunk in comp_list:
            print(f" - {chunk}")


if __name__ == "__main__":
    main()

"""
Metadata-Based Graph Construction Script

This script constructs a graph with node and edge logic based on document and table metadata.
It handles three types of connections: Doc-to-Doc, Doc-to-Table, and Table-to-Table.
"""

import json
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configuration



class MetadataGraphConstructor:
    def __init__(self):
        print("Initializing Sentence Transformer model...")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.docs = []
        self.tables = []
        self.edges = []
        

    

    
 
    
    def calculate_entity_overlap(self, entities1, entities2, is_table_comparison=False):
        """Calculate overlap between entities"""
        if not entities1 or not entities2:
            return 0, {}, {}
        
        entity_names1 = set(entities1.keys())
        entity_names2 = set(entities2.keys())
        common_entities = entity_names1.intersection(entity_names2)
        
        if not common_entities:
            return 0, {}, {}
        
        # Calculate detailed overlap - return count instead of ratio for clarity
        overlap_score = len(common_entities)  # Return actual count
        
        combined_entities = {}
        relationship_overlap = {}
        
        for entity in common_entities:
            e1_data = entities1.get(entity, {})
            e2_data = entities2.get(entity, {})
            
            if is_table_comparison:
                # Handle table entity format: type, category, description
                combined_entities[entity] = {
                    "type": e1_data.get("type", e2_data.get("type", "")),
                    "category": e1_data.get("category", e2_data.get("category", "")),
                    "description": list(set(filter(None, [e1_data.get("description", ""), e2_data.get("description", "")])))
                }
                # No relationship overlap for table entities
                relationship_overlap[entity] = 0.0
            else:
                # Handle document entity format: relationships, actions, details
                combined_entities[entity] = {
                    "relationships": list(set(
                        tuple(rel) if isinstance(rel, list) else rel 
                        for rel in (e1_data.get("relationships", []) + e2_data.get("relationships", []))
                    )),
                    "actions": list(set(e1_data.get("actions", []) + e2_data.get("actions", []))),
                    "details": list(set(e1_data.get("details", []) + e2_data.get("details", [])))
                }
                
                # Calculate relationship overlap
                rel1 = set(tuple(rel) if isinstance(rel, list) else rel for rel in e1_data.get("relationships", []))
                rel2 = set(tuple(rel) if isinstance(rel, list) else rel for rel in e2_data.get("relationships", []))
                rel_overlap = len(rel1.intersection(rel2)) / max(len(rel1), len(rel2), 1)
                relationship_overlap[entity] = float(rel_overlap)
        
        return overlap_score, combined_entities, relationship_overlap
    
    def calculate_entity_overlap_mixed(self, doc_entities, table_entities):
        """Calculate overlap between document entities (doc format) and table entities (table format)"""
        if not doc_entities or not table_entities:
            return 0, {}, {}
        
        entity_names1 = set(doc_entities.keys())
        entity_names2 = set(table_entities.keys())
        common_entities = entity_names1.intersection(entity_names2)
        
        if not common_entities:
            return 0, {}, {}
        
        # Calculate detailed overlap - return count
        overlap_score = len(common_entities)
        
        combined_entities = {}
        relationship_overlap = {}
        
        for entity in common_entities:
            doc_data = doc_entities.get(entity, {})
            table_data = table_entities.get(entity, {})
            
            # Combine mixed formats - keep both doc and table information
            combined_entities[entity] = {
                "doc_format": {
                    "relationships": doc_data.get("relationships", []),
                    "actions": doc_data.get("actions", []),
                    "details": doc_data.get("details", [])
                },
                "table_format": {
                    "type": table_data.get("type", ""),
                    "category": table_data.get("category", ""),
                    "description": table_data.get("description", "")
                }
            }
            
            # No meaningful relationship overlap between different formats
            relationship_overlap[entity] = 0.0
        
        return overlap_score, combined_entities, relationship_overlap
    
    def calculate_event_overlap(self, events1, events2):
        """Calculate overlap between events"""
        if not events1 or not events2:
            return 0, {}
        
        event_names1 = set(events1.keys())
        event_names2 = set(events2.keys())
        common_events = event_names1.intersection(event_names2)
        
        if not common_events:
            return 0, {}
        
        overlap_score = len(common_events)  # Return actual count
        
        combined_events = {}
        for event in common_events:
            e1_data = events1.get(event, {})
            e2_data = events2.get(event, {})
            
            combined_events[event] = {
                "dates": list(set(filter(None, [e1_data.get("date"), e2_data.get("date")]))),
                "actions": list(set(e1_data.get("actions", []) + e2_data.get("actions", []))),
                "details": list(set(filter(None, [e1_data.get("details"), e2_data.get("details")])))
            }
        
        return overlap_score, combined_events
    
    def doc_to_doc_similarity(self, doc1, doc2):
        """Calculate similarity between two documents"""
        meta1 = doc1.get('metadata', {})
        meta2 = doc2.get('metadata', {})
        
        # Entity and event overlap
        entity_score, common_entities, rel_overlap = self.calculate_entity_overlap(
            meta1.get('entities', {}), meta2.get('entities', {})
        )
        event_score, common_events = self.calculate_event_overlap(
            meta1.get('events', {}), meta2.get('events', {})
        )
        
        # Topic similarity
        topic1 = meta1.get('topic', '')
        topic2 = meta2.get('topic', '')
        topic_sim = 0
        if topic1 and topic2:
            topic_emb1 = self.get_embedding(topic1)
            topic_emb2 = self.get_embedding(topic2)
            topic_sim = cosine_similarity([topic_emb1], [topic_emb2])[0][0]
        
        # Content embedding similarity
        content1 = doc1.get('content', '')
        content2 = doc2.get('content', '')
        content_sim = 0
        if content1 and content2:
            content_emb1 = self.get_embedding(content1)
            content_emb2 = self.get_embedding(content2)
            content_sim = cosine_similarity([content_emb1], [content_emb2])[0][0]
        
        # Entity relationship overlap score
        entity_rel_score = np.mean(list(rel_overlap.values())) if rel_overlap else 0
        
        
        
        return {
           
            'entity_relationship_overlap': float(entity_rel_score),
            'topic_similarity': float(topic_sim),
            'content_similarity': float(content_sim),
            'common_entities': common_entities,
            'common_events': common_events,
            'combined_timeline': list(set(
                meta1.get('timeline', []) + meta2.get('timeline', [])
            ))
        }
    
    def doc_to_table_similarity(self, doc, table):
        """Calculate similarity between document and table"""
        doc_meta = doc.get('metadata', {})
        table_meta = table.get('metadata', {})
        
        # Entity overlap (table entities with doc entities) - mixed format
        entity_score, common_entities, _ = self.calculate_entity_overlap_mixed(
            doc_meta.get('entities', {}), table_meta.get('entities', {})
        )
        
        # Event overlap (events can be entities in table)
        event_score = 0
        common_events = {}
        doc_events = doc_meta.get('events', {})
        table_entities = table_meta.get('entities', {})
        
        if doc_events and table_entities:
            event_entity_overlap = set(doc_events.keys()).intersection(set(table_entities.keys()))
            if event_entity_overlap:
                event_score = len(event_entity_overlap)  # Return count
                for event_name in event_entity_overlap:
                    common_events[event_name] = {
                        "dates": [doc_events[event_name].get("date")] if doc_events[event_name].get("date") else [],
                        "actions": doc_events[event_name].get("actions", []),
                        "details": [doc_events[event_name].get("details")] if doc_events[event_name].get("details") else []
                    }
        
        # Topic and table title/summary similarities
        doc_topic = doc_meta.get('topic', '')
        table_title = table_meta.get('table_title', '')
        table_summary = table_meta.get('table_summary', '')
        table_desc = table_meta.get('table_description', '')
        
        topic_title_sim = 0
        topic_summary_sim = 0
        if doc_topic:
            if table_title:
                topic_emb = self.get_embedding(doc_topic)
                title_emb = self.get_embedding(table_title)
                topic_title_sim = cosine_similarity([topic_emb], [title_emb])[0][0]
            
            if table_summary:
                topic_emb = self.get_embedding(doc_topic)
                summary_emb = self.get_embedding(table_summary)
                topic_summary_sim = cosine_similarity([topic_emb], [summary_emb])[0][0]
        
        # Content similarity
        doc_emb = doc.get('embedding', '')
        table_content = table.get('content', '')
        content_sim = 0
        if doc_content and table_content:
            doc_emb = self.get_embedding(doc_content)
            table_emb = self.get_embedding(table_content)
            content_sim = cosine_similarity([doc_emb], [table_emb])[0][0]
        
        # Column description similarities
        col_desc_sims = {}
        if doc_emb is not None and table_meta.get('col_desc'):
            for col_name, col_desc in table_meta['col_desc'].items():
                if col_desc:
                    col_emb = self.get_embedding(col_desc)
                    col_desc_sims[col_name] = cosine_similarity([doc_emb], [col_emb])[0][0]

        top_3_similarities = sorted(col_desc_sims, reverse=True)[:3]
        avg_col_desc_sim = np.mean(top_3_similarities) if col_desc_sims else 0
        
        # Row similarities
        row_sims = {}
        if doc_emb is not None and table.get('rows_with_headers'):
            for i, row in enumerate(table['rows_with_headers']):
                row_text = ' '.join(str(v) for v in row.values())
                if row_text:
                    row_emb = self.get_embedding(row_text)
                    row_sims[f'row_{i+1}'] = cosine_similarity([doc_emb], [row_emb])[0][0]
        
        top_3_similarities = sorted(row_sims, reverse=True)[:3]
        avg_row_sim = np.mean(top_3_similarities) if col_desc_sims else 0

     
   
        return {
           
            
            'topic_title_similarity': float(topic_title_sim),
            'topic_summary_similarity': float(topic_summary_sim),
            'content_similarity': float(content_sim),
            'avg_col_desc_similarity': float(avg_col_desc_sim),
            'avg_row_similarity': float(avg_row_sim),
            'common_entities': common_entities,
        }
    
    def table_to_table_similarity(self, table1, table2):
        """Calculate similarity between two tables"""
        meta1 = table1.get('metadata', {})
        meta2 = table2.get('metadata', {})
        
        # Entity overlap - use table format
        entity_score, common_entities, _ = self.calculate_entity_overlap(
            meta1.get('entities', {}), meta2.get('entities', {}), is_table_comparison=True
        )
        
        # Title similarity
        title1 = meta1.get('table_title', '')
        title2 = meta2.get('table_title', '')
        title_sim = 0
        if title1 and title2:
            title_emb1 = self.get_embedding(title1)
            title_emb2 = self.get_embedding(title2)
            title_sim = cosine_similarity([title_emb1], [title_emb2])[0][0]
        
        # Description similarity
        desc1 = meta1.get('table_description', '')
        desc2 = meta2.get('table_description', '')
        desc_sim = 0
        if desc1 and desc2:
            desc_emb1 = self.get_embedding(desc1)
            desc_emb2 = self.get_embedding(desc2)
            desc_sim = cosine_similarity([desc_emb1], [desc_emb2])[0][0]
        
        # Column description similarities
        col_sims = {}
        col_desc1 = meta1.get('col_desc', {})
        col_desc2 = meta2.get('col_desc', {})
        
        if col_desc1 and col_desc2:
            for col1_name, col1_desc in col_desc1.items():
                col_sims[col1_name] = {}
                if col1_desc:
                    col1_emb = self.get_embedding(col1_desc)
                    for col2_name, col2_desc in col_desc2.items():
                        if col2_desc:
                            col2_emb = self.get_embedding(col2_desc)
                            col_sims[col1_name][col2_name] = cosine_similarity([col1_emb], [col2_emb])[0][0]
        
        top_3_similarities = sorted(col_sims, reverse=True)[:3]
        avg_col_sim = np.mean(top_3_similarities) if top_3_similarities else 0
       
       
        

        return {
            
            'title_similarity': float(title_sim),
            'description_similarity': float(desc_sim),
            'avg_column_similarity': float(avg_col_sim),
            'common_entities': common_entities,
        }
    
    
if __name__ == "__main__":
    main()

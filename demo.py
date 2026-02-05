"""
Demonstration script for the RAG application
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.models import ProcessingConfig
from src.pipeline.rag_pipeline import RAGPipeline
from src.visualization.dash_app import run_dashboard
from loguru import logger
import pandas as pd


def create_sample_data():
    """Create sample data files for demonstration"""
    # Create data directory
    data_dir = Path("sample_data")
    data_dir.mkdir(exist_ok=True)
    
    # Create sample documents
    doc1_content = """
    Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. 
    It focuses on the development of computer programs that can access data and use it to learn for themselves.
    Deep learning is a subset of machine learning that uses neural networks with multiple layers.
    Neural networks are computing systems inspired by biological neural networks.
    """
    
    doc2_content = """
    Natural language processing (NLP) is a branch of artificial intelligence that deals with the interaction between computers and humans.
    NLP combines computational linguistics with statistical, machine learning, and deep learning models.
    Text classification is a common NLP task that involves categorizing text into predefined categories.
    Sentiment analysis is another popular NLP application that determines the emotional tone of text.
    """
    
    doc3_content = """
    Data science is an interdisciplinary field that uses scientific methods to extract knowledge from data.
    It combines domain expertise, programming skills, and knowledge of mathematics and statistics.
    Machine learning algorithms are frequently used in data science projects.
    Python and R are popular programming languages for data science.
    """
    
    with open(data_dir / "ml_basics.txt", "w") as f:
        f.write(doc1_content)
    
    with open(data_dir / "nlp_overview.txt", "w") as f:
        f.write(doc2_content)
    
    with open(data_dir / "data_science.txt", "w") as f:
        f.write(doc3_content)
    
    # Create sample tables
   
    
    # Table 1: AI Technologies
    ai_tech_data = {
        'Technology': ['Machine Learning', 'Deep Learning', 'Natural Language Processing', 'Computer Vision', 'Robotics'],
        'Category': ['AI Subset', 'ML Subset', 'AI Subset', 'AI Subset', 'AI Application'],
        'Primary_Use': ['Pattern Recognition', 'Complex Pattern Recognition', 'Text Processing', 'Image Processing', 'Automation'],
        'Difficulty_Level': ['Medium', 'High', 'Medium', 'High', 'High'],
        'Industry_Adoption': ['High', 'Medium', 'High', 'Medium', 'Low']
    }
    
    ai_tech_df = pd.DataFrame(ai_tech_data)
    ai_tech_df.to_csv(data_dir / "ai_technologies.csv", index=False)
    
    # Table 2: Programming Languages
    lang_data = {
        'Language': ['Python', 'R', 'Java', 'JavaScript', 'C++'],
        'Primary_Domain': ['Data Science', 'Statistics', 'Enterprise', 'Web Development', 'System Programming'],
        'Learning_Curve': ['Easy', 'Medium', 'Medium', 'Easy', 'Hard'],
        'AI_ML_Support': ['Excellent', 'Good', 'Good', 'Fair', 'Fair'],
        'Community_Size': ['Very Large', 'Large', 'Very Large', 'Very Large', 'Large']
    }
    
    lang_df = pd.DataFrame(lang_data)
    lang_df.to_csv(data_dir / "programming_languages.csv", index=False)
    
    # Table 3: ML Algorithms
    algo_data = {
        'Algorithm': ['Linear Regression', 'Random Forest', 'Neural Networks', 'SVM', 'K-Means'],
        'Type': ['Supervised', 'Supervised', 'Supervised/Unsupervised', 'Supervised', 'Unsupervised'],
        'Problem_Type': ['Regression', 'Classification/Regression', 'Classification/Regression', 'Classification', 'Clustering'],
        'Interpretability': ['High', 'Medium', 'Low', 'Medium', 'Medium'],
        'Performance': ['Medium', 'High', 'Very High', 'High', 'Medium']
    }
    
    algo_df = pd.DataFrame(algo_data)
    algo_df.to_csv(data_dir / "ml_algorithms.csv", index=False)
    
    logger.info(f"Sample data created in {data_dir}")
    return str(data_dir)


def run_basic_demo():
    """Run basic demonstration with command line output"""
    logger.info("=== Basic Command Line Demo ===")
    
    # Sample data
    create_sample_data()
    
    # Initialize with customized thresholds for different operations
    config = ProcessingConfig(
        # Document processing: stricter threshold for sentence merging
        sentence_similarity_threshold=0.5,
        
        # Table processing: more lenient threshold for row merging
        row_similarity_threshold=0.6,
        
        # Graph building: different thresholds for different node types
        table_to_table_threshold=0.6,        # Tables often have structured similarity
        table_to_document_threshold=0.4,     # Cross-type connections need lower threshold
        document_to_document_threshold=0.5,  # Documents can be topically related with lower similarity
        
        # Other settings
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        remove_stopwords=True
    )
    
    logger.info("=== RAG Application Demonstration ===")
    logger.info(f"Configuration:")
    logger.info(f"  - Sentence similarity threshold: {config.sentence_similarity_threshold}")
    logger.info(f"  - Row similarity threshold: {config.row_similarity_threshold}")
    logger.info(f"  - Table-to-table threshold: {config.table_to_table_threshold}")
    logger.info(f"  - Table-to-document threshold: {config.table_to_document_threshold}")
    logger.info(f"  - Document-to-document threshold: {config.document_to_document_threshold}")
    
    # Initialize pipeline
    logger.info("Initializing RAG Pipeline...")
    pipeline = RAGPipeline(config)
    
    # Process data
    logger.info("Processing documents and tables...")
    knowledge_graph = pipeline.process_directory(data_directory)
    
    # Display statistics
    logger.info("=== Knowledge Graph Statistics ===")
    stats = pipeline.get_graph_statistics()
    print(f"Nodes: {stats['nodes']['total']}")
    print(f"Node types: {stats['nodes']['by_type']}")
    print(f"Edges: {stats['edges']['total']}")
    print(f"Edge types: {stats['edges']['by_type']}")
    print(f"Graph density: {stats['connectivity']['density']:.3f}")
    
    # Demonstrate queries
    logger.info("=== Query Demonstrations ===")
    
    # Text query
    print("\n1. Text Query: 'machine learning algorithms'")
    text_results = pipeline.query_by_text("machine learning algorithms", top_k=3)
    for i, result in enumerate(text_results):
        print(f"   Result {i+1}: {result['source_name']} (similarity: {result['similarity']:.3f})")
        print(f"   Keywords: {result['keywords'][:3]}")
        print(f"   Summary: {result['content_preview'][:100]}...")
        print()
    
    # Keyword query
    print("\n2. Keyword Query: ['python', 'programming']")
    keyword_results = pipeline.query_by_keywords(['python', 'programming'], min_matches=1)
    for i, result in enumerate(keyword_results):
        print(f"   Result {i+1}: {result['source_name']}")
        print(f"   Type: {result['chunk_type']}")
        print(f"   Keywords: {result['keywords'][:5]}")
        print()
    
    # Export graph
    export_path = "sample_data/knowledge_graph.json"
    pipeline.export_graph(export_path)
    logger.info(f"Graph exported to {export_path}")
    
    return pipeline


def run_interactive_demo():
    """Run interactive dashboard demonstration"""
    logger.info("=== Interactive Dashboard Demo ===")
    
    # Sample data
    data_directory = create_sample_data()
    
    # Initialize with optimized thresholds for better graph connectivity
    config = ProcessingConfig(
        # Fine-tuned thresholds for demo data
        sentence_similarity_threshold=0.3,   # Allow more sentence merging
        row_similarity_threshold=0.3,       # More flexible row merging
        table_to_table_threshold=0.6,       # Enable table connections
        table_to_document_threshold=0.4,    # Cross-type connections
        document_to_document_threshold=0.5, # Topic-based document connections
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        remove_stopwords=True
    )
    
    # Initialize pipeline
    logger.info("Initializing RAG Pipeline...")
    pipeline = RAGPipeline(config)
    
    # Process data
    logger.info("Processing documents and tables...")
    knowledge_graph = pipeline.process_directory(data_directory)
    
    # Display statistics
    logger.info("=== Knowledge Graph Statistics ===")
    stats = pipeline.get_graph_statistics()
    print(f"Nodes: {stats['nodes']['total']}")
    print(f"Node types: {stats['nodes']['by_type']}")
    print(f"Edges: {stats['edges']['total']}")
    print(f"Edge types: {stats['edges']['by_type']}")
    print(f"Graph density: {stats['connectivity']['density']:.3f}")
    
    # Demonstrate queries
    logger.info("=== Query Demonstrations ===")
    
    # Text query
    print("\n1. Text Query: 'machine learning algorithms'")
    text_results = pipeline.query_by_text("machine learning algorithms", top_k=3)
    for i, result in enumerate(text_results):
        print(f"   Result {i+1}: {result['source_name']} (similarity: {result['similarity']:.3f})")
        print(f"   Keywords: {result['keywords'][:3]}")
        print(f"   Summary: {result['content_preview'][:100]}...")
        print()
    
    # Keyword query
    print("\n2. Keyword Query: ['python', 'programming']")
    keyword_results = pipeline.query_by_keywords(['python', 'programming'], min_matches=1)
    for i, result in enumerate(keyword_results):
        print(f"   Result {i+1}: {result['source_name']}")
        print(f"   Type: {result['chunk_type']}")
        print(f"   Keywords: {result['keywords'][:5]}")
        print()
    
    # Export graph
    export_path = "/shared/khoja/CogComp/output/full_pipeline/integrated_graph_20250623_184401.json"
    pipeline.export_graph(export_path)
    logger.info(f"Graph exported to {export_path}")
    
    # Launch dashboard
    logger.info("Starting interactive dashboard...")
    logger.info("Open http://localhost:8050 in your browser")
    logger.info("Press Ctrl+C to stop the server")
    
    try:
        run_dashboard(pipeline, port=8050, debug=False)
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
    
    print("RAG Application Demo")
    print("Choose an option:")
    print("1. Basic demonstration (command line)")
    print("2. Interactive dashboard demonstration")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            run_basic_demo()
        elif choice == "2":
            run_interactive_demo()
        else:
            print("Invalid choice. Running basic demonstration...")
            run_basic_demo()
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Error running demo: {e}")
        raise 

# UPDATED OUTLIER ANALYSIS METHODS - REPLACE IN graph_analysis_pipeline.py

def _analyze_doc_doc_outliers(self, df: pd.DataFrame, outlier_dir: Path):
    """FIXED: Doc-Doc outlier analysis with optimal thresholds"""
    logger.info("Analyzing doc-doc outliers with OPTIMAL STRATEGY...")
    
    doc_doc_df = df[df['edge_type'] == 'doc-doc'].copy()
    
    if len(doc_doc_df) == 0:
        logger.warning("No doc-doc edges found for outlier analysis")
        return
    
    section_dir = outlier_dir / "section_1_doc_doc_outliers"
    section_dir.mkdir(exist_ok=True)
    
    # OPTIMAL STRATEGY: 95th percentile similarities AND 80th percentile entity requirement
    
    # Calculate thresholds
    topic_threshold = doc_doc_df['topic_similarity'].quantile(0.95)
    content_threshold = doc_doc_df['content_similarity'].quantile(0.95)
    
    # Entity similarity threshold (using content similarity as proxy)
    entity_rows = doc_doc_df[doc_doc_df['entity_count'] > 0]
    if len(entity_rows) > 0:
        entity_sim_threshold = entity_rows['content_similarity'].quantile(0.80)
    else:
        entity_sim_threshold = 0.0
    
    logger.info(f"Doc-Doc OPTIMAL thresholds:")
    logger.info(f"  Topic similarity > {topic_threshold:.3f}")
    logger.info(f"  Content similarity > {content_threshold:.3f}")
    logger.info(f"  Entity similarity > {entity_sim_threshold:.3f}")
    
    # AND logic: High similarities AND entity matches with entity similarity requirement
    similarity_mask = (doc_doc_df['topic_similarity'] > topic_threshold) & \
                     (doc_doc_df['content_similarity'] > content_threshold)
    
    entity_mask = (doc_doc_df['entity_count'] >= 1) & \
                  (doc_doc_df['content_similarity'] > entity_sim_threshold)
    
    # FIXED: AND logic instead of OR logic
    optimal_mask = similarity_mask & entity_mask
    optimal_outliers = doc_doc_df[optimal_mask].copy()
    
    logger.info(f"Selected {len(optimal_outliers)} doc-doc edges with OPTIMAL strategy")
    
    # Store for graph building
    self._store_outliers_for_graph_building(optimal_outliers, 'doc-doc', 'optimal_95th_80th')
    
    # Save outliers with content
    self._save_outliers_with_content(
        optimal_outliers,
        section_dir / "optimal_strategy_outliers",
        "Optimal Strategy: 95th percentile similarities AND 80th percentile entity similarity",
        "optimal_strategy"
    )

def _analyze_doc_table_outliers(self, df: pd.DataFrame, outlier_dir: Path):
    """FIXED: Doc-Table outlier analysis with optimal thresholds"""
    logger.info("Analyzing doc-table outliers with OPTIMAL STRATEGY...")
    
    doc_table_df = df[df['edge_type'] == 'doc-table'].copy()
    
    if len(doc_table_df) == 0:
        logger.warning("No doc-table edges found for outlier analysis")
        return
    
    section_dir = outlier_dir / "section_2_doc_table_outliers"
    section_dir.mkdir(exist_ok=True)
    
    # OPTIMAL STRATEGY: 95th percentile similarities AND 80th percentile entity requirement
    
    # Calculate similarity thresholds
    col_threshold = doc_table_df['column_similarity'].quantile(0.95)
    title_threshold = doc_table_df['topic_title_similarity'].quantile(0.95)
    summary_threshold = doc_table_df['topic_summary_similarity'].quantile(0.95)
    
    # Entity similarity threshold
    entity_rows = doc_table_df[doc_table_df['entity_count'] > 0]
    if len(entity_rows) > 0:
        entity_sim_threshold = entity_rows['content_similarity'].quantile(0.80)
    else:
        entity_sim_threshold = 0.0
    
    logger.info(f"Doc-Table OPTIMAL thresholds:")
    logger.info(f"  Column similarity > {col_threshold:.3f}")
    logger.info(f"  Topic-title similarity > {title_threshold:.3f}")
    logger.info(f"  Topic-summary similarity > {summary_threshold:.3f}")
    logger.info(f"  Entity similarity > {entity_sim_threshold:.3f}")
    
    # AND logic: All similarities high AND entity matches with entity similarity requirement
    similarity_mask = (doc_table_df['column_similarity'] > col_threshold) & \
                     (doc_table_df['topic_title_similarity'] > title_threshold) & \
                     (doc_table_df['topic_summary_similarity'] > summary_threshold)
    
    entity_mask = (doc_table_df['entity_count'] >= 1) & \
                  (doc_table_df['content_similarity'] > entity_sim_threshold)
    
    # FIXED: AND logic instead of OR logic
    optimal_mask = similarity_mask & entity_mask
    optimal_outliers = doc_table_df[optimal_mask].copy()
    
    logger.info(f"Selected {len(optimal_outliers)} doc-table edges with OPTIMAL strategy")
    
    # Store for graph building
    self._store_outliers_for_graph_building(optimal_outliers, 'doc-table', 'optimal_95th_80th')
    
    # Save outliers with content
    self._save_outliers_with_content(
        optimal_outliers,
        section_dir / "optimal_strategy_outliers",
        "Optimal Strategy: 95th percentile similarities AND 80th percentile entity similarity",
        "optimal_strategy"
    )

def _analyze_table_table_outliers(self, df: pd.DataFrame, outlier_dir: Path):
    """FIXED: Table-Table outlier analysis with optimal thresholds"""
    logger.info("Analyzing table-table outliers with OPTIMAL STRATEGY...")
    
    table_table_df = df[df['edge_type'] == 'table-table'].copy()
    
    if len(table_table_df) == 0:
        logger.warning("No table-table edges found for outlier analysis")
        return
    
    section_dir = outlier_dir / "section_3_table_table_outliers"
    section_dir.mkdir(exist_ok=True)
    
    # OPTIMAL STRATEGY: 95th percentile similarities AND 80th percentile entity requirement
    
    # Calculate similarity thresholds
    col_threshold = table_table_df['column_similarity'].quantile(0.95)
    title_threshold = table_table_df['title_similarity'].quantile(0.95)
    desc_threshold = table_table_df['description_similarity'].quantile(0.95)
    
    # Entity similarity threshold
    entity_rows = table_table_df[table_table_df['entity_count'] > 0]
    if len(entity_rows) > 0:
        entity_sim_threshold = entity_rows['content_similarity'].quantile(0.80)
    else:
        entity_sim_threshold = 0.0
    
    logger.info(f"Table-Table OPTIMAL thresholds:")
    logger.info(f"  Column similarity > {col_threshold:.3f}")
    logger.info(f"  Title similarity > {title_threshold:.3f}")
    logger.info(f"  Description similarity > {desc_threshold:.3f}")
    logger.info(f"  Entity similarity > {entity_sim_threshold:.3f}")
    
    # AND logic: All similarities high AND entity matches with entity similarity requirement
    similarity_mask = (table_table_df['column_similarity'] > col_threshold) & \
                     (table_table_df['title_similarity'] > title_threshold) & \
                     (table_table_df['description_similarity'] > desc_threshold)
    
    entity_mask = (table_table_df['entity_count'] >= 1) & \
                  (table_table_df['content_similarity'] > entity_sim_threshold)
    
    # FIXED: AND logic instead of OR logic
    optimal_mask = similarity_mask & entity_mask
    optimal_outliers = table_table_df[optimal_mask].copy()
    
    logger.info(f"Selected {len(optimal_outliers)} table-table edges with OPTIMAL strategy")
    
    # Store for graph building
    self._store_outliers_for_graph_building(optimal_outliers, 'table-table', 'optimal_95th_80th')
    
    # Save outliers with content
    self._save_outliers_with_content(
        optimal_outliers,
        section_dir / "optimal_strategy_outliers",
        "Optimal Strategy: 95th percentile similarities AND 80th percentile entity similarity",
        "optimal_strategy"
    )

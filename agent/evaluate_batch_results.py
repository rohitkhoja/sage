#!/usr/bin/env python3
"""
Evaluate batch question results against gold using custom candidate construction rules.

Inputs:
- CSV: /shared/khoja/CogComp/output/BM25_stark_mag_human_rewritten.csv
- Agent outputs: /shared/khoja/CogComp/agent/output/batch_questions/question_{query_id}.json

Rules to build candidate ranking per question:
- If agent retrieved < 10 nodes: use agent nodes in their returned order.
- If 10 <= agent retrieved < 500: candidate list = intersection(top_100, agent_nodes), ordered as in top_100.
- If agent retrieved >= 500: candidate list = top_100 only.

Metrics:
- hit@1: 1 if any gold in top 1 else 0
- hit@5: 1 if any gold in top 5 else 0
- recall@20: |gold ∩ top20| / |gold|
- MRR: 1/rank of first gold hit (1-indexed), else 0
- MRR@20: 1/rank if first gold hit within top 20, else 0

Outputs:
- Prints overall averages
- Writes per-question CSV and overall JSON to /shared/khoja/CogComp/agent/output/batch_eval/
"""

import csv
import json
import os
import ast
from typing import List, Dict, Any, Tuple
from statistics import mean
from pathlib import Path

CSV_FILE = "/shared/khoja/CogComp/output/BM25_stark_mag_human_rewritten.csv"
AGENT_OUT_DIR = "/shared/khoja/CogComp/agent/output/batch_questions"
OUT_DIR = "/shared/khoja/CogComp/agent/output/batch_eval"


def parse_list(value: str) -> List[int]:
    """Parse a list of ints from CSV string like "[1, 2, 3]" safely."""
    if value is None:
        return []
    s = value.strip()
    if not s:
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [int(x) for x in parsed if isinstance(x, (int, str)) and str(x).isdigit()]
        return []
    except Exception:
        return []


def build_candidates(agent_nodes: List[int], top100: List[int]) -> List[int]:
    n = len(agent_nodes)
    if n == 0:
        return []
    if n < 10:
        # Use agent order as-is
        return [int(x) for x in agent_nodes]
    if n < 500:
        agent_set = set(int(x) for x in agent_nodes)
        # keep top_100 order
        return [pid for pid in top100 if pid in agent_set]
    # n >= 500
    return list(top100)


def mrr(ranked: List[int], gold: List[int]) -> float:
    gold_set = set(gold)
    for idx, pid in enumerate(ranked, 1):
        if pid in gold_set:
            return 1.0 / idx
    return 0.0


def mrr_at_k(ranked: List[int], gold: List[int], k: int) -> float:
    gold_set = set(gold)
    for idx, pid in enumerate(ranked[:k], 1):
        if pid in gold_set:
            return 1.0 / idx
    return 0.0


def hit_at_k(ranked: List[int], gold: List[int], k: int) -> float:
    gold_set = set(gold)
    return 1.0 if any(pid in gold_set for pid in ranked[:k]) else 0.0


def recall_at_k(ranked: List[int], gold: List[int], k: int) -> float:
    if not gold:
        return 0.0
    gold_set = set(gold)
    hits = sum(1 for pid in ranked[:k] if pid in gold_set)
    return hits / float(len(gold_set))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    per_question_rows: List[Dict[str, Any]] = []

    agg_hit1, agg_hit5, agg_rec20, agg_mrr, agg_mrr20 = [], [], [], [], []
    # Stratified recall@20 by agent size bucket
    rec20_lt10, rec20_10_499, rec20_ge500 = [], [], []
    base_rec20_lt10, base_rec20_10_499, base_rec20_ge500 = [], [], []

    # Improvement tracking
    deltas_all: List[float] = []
    deltas_lt10: List[float] = []
    deltas_10_499: List[float] = []
    deltas_ge500: List[float] = []
    improved = worsened = same = 0

    for row in rows:
        try:
            query_id = int(row['query_id'])
        except Exception:
            continue

        gold_docs = parse_list(row.get('gold_docs', ''))
        top100 = parse_list(row.get('top_100', ''))

        # Load agent results
        agent_file = os.path.join(AGENT_OUT_DIR, f"question_{query_id}.json")
        agent_nodes: List[int] = []
        if os.path.exists(agent_file):
            try:
                with open(agent_file, 'r', encoding='utf-8') as af:
                    data = json.load(af)
                    nodes = data.get('final_nodes', [])
                    if isinstance(nodes, list):
                        agent_nodes = [int(x) for x in nodes if isinstance(x, (int, str)) and str(x).isdigit()]
            except Exception:
                agent_nodes = []

        ranked = build_candidates(agent_nodes, top100)

        # Baseline recall@20 from top_100 only (top 20)
        base_rec20 = recall_at_k(top100, gold_docs, 20)

        q_hit1 = hit_at_k(ranked, gold_docs, 1)
        q_hit5 = hit_at_k(ranked, gold_docs, 5)
        q_rec20 = recall_at_k(ranked, gold_docs, 20)
        q_mrr = mrr(ranked, gold_docs)
        q_mrr20 = mrr_at_k(ranked, gold_docs, 20)

        n_agent = len(agent_nodes)
        # Special rule: if agent < 10 and recall@20 is zero, don't consider agent; use baseline (top100)
        if n_agent < 10 and q_rec20 == 0.0:
            q_hit1 = hit_at_k(top100, gold_docs, 1)
            q_hit5 = hit_at_k(top100, gold_docs, 5)
            q_rec20 = base_rec20
            q_mrr = mrr(top100, gold_docs)
            q_mrr20 = mrr_at_k(top100, gold_docs, 20)

        agg_hit1.append(q_hit1)
        agg_hit5.append(q_hit5)
        agg_rec20.append(q_rec20)
        agg_mrr.append(q_mrr)
        agg_mrr20.append(q_mrr20)

        # Bucketed recall@20
        if n_agent < 10:
            rec20_lt10.append(q_rec20)
            base_rec20_lt10.append(base_rec20)
            agent_bucket = '<10'
        elif n_agent < 500:
            rec20_10_499.append(q_rec20)
            base_rec20_10_499.append(base_rec20)
            agent_bucket = '10-499'
        else:
            rec20_ge500.append(q_rec20)
            base_rec20_ge500.append(base_rec20)
            agent_bucket = '>=500'

        # Delta tracking
        delta = q_rec20 - base_rec20
        deltas_all.append(delta)
        if n_agent < 10:
            deltas_lt10.append(delta)
        elif n_agent < 500:
            deltas_10_499.append(delta)
        else:
            deltas_ge500.append(delta)
        if delta > 1e-9:
            improved += 1
        elif delta < -1e-9:
            worsened += 1
        else:
            same += 1

        per_question_rows.append({
            'query_id': query_id,
            'question': row.get('query', ''),
            'num_gold': len(gold_docs),
            'num_top100': len(top100),
            'num_agent_nodes': len(agent_nodes),
            'num_ranked_used': len(ranked),
            'agent_bucket': agent_bucket,
            'hit@1': q_hit1,
            'hit@5': q_hit5,
            'recall@20': q_rec20,
            'baseline_recall@20(top100)': base_rec20,
            'delta_recall@20': delta,
            'mrr': q_mrr,
            'mrr@20': q_mrr20,
        })

    # Overall metrics
    overall = {
        'count': len(per_question_rows),
        'hit@1': mean(agg_hit1) if agg_hit1 else 0.0,
        'hit@5': mean(agg_hit5) if agg_hit5 else 0.0,
        'recall@20': mean(agg_rec20) if agg_rec20 else 0.0,
        'baseline_recall@20_top100': mean(base_rec20_lt10 + base_rec20_10_499 + base_rec20_ge500) if (base_rec20_lt10 or base_rec20_10_499 or base_rec20_ge500) else 0.0,
        'mrr': mean(agg_mrr) if agg_mrr else 0.0,
        'mrr@20': mean(agg_mrr20) if agg_mrr20 else 0.0,
        # Stratified recall@20
        'recall@20_<10': mean(rec20_lt10) if rec20_lt10 else 0.0,
        'recall@20_10_499': mean(rec20_10_499) if rec20_10_499 else 0.0,
        'recall@20_>=500': mean(rec20_ge500) if rec20_ge500 else 0.0,
        'baseline_recall@20_<10': mean(base_rec20_lt10) if base_rec20_lt10 else 0.0,
        'baseline_recall@20_10_499': mean(base_rec20_10_499) if base_rec20_10_499 else 0.0,
        'baseline_recall@20_>=500': mean(base_rec20_ge500) if base_rec20_ge500 else 0.0,
        # Delta aggregates
        'delta_recall@20_mean': mean(deltas_all) if deltas_all else 0.0,
        'delta_recall@20_mean_<10': mean(deltas_lt10) if deltas_lt10 else 0.0,
        'delta_recall@20_mean_10_499': mean(deltas_10_499) if deltas_10_499 else 0.0,
        'delta_recall@20_mean_>=500': mean(deltas_ge500) if deltas_ge500 else 0.0,
        'improved_count': improved,
        'worsened_count': worsened,
        'same_count': same,
        'count_<10': len(rec20_lt10),
        'count_10_499': len(rec20_10_499),
        'count_>=500': len(rec20_ge500),
    }

    # Save outputs
    json_out = os.path.join(OUT_DIR, 'summary.json')
    with open(json_out, 'w', encoding='utf-8') as jf:
        json.dump({'overall': overall, 'per_question': per_question_rows}, jf, indent=2)

    csv_out = os.path.join(OUT_DIR, 'per_question.csv')
    with open(csv_out, 'w', encoding='utf-8', newline='') as cf:
        fieldnames = list(per_question_rows[0].keys()) if per_question_rows else [
            'query_id','question','num_gold','num_top100','num_agent_nodes','num_ranked_used','hit@1','hit@5','recall@20','mrr','mrr@20']
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for r in per_question_rows:
            writer.writerow(r)

    print("\n=== Overall Metrics ===")
    for k, v in overall.items():
        if k.startswith('count') or k == 'count':
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.6f}")
    print(f"\nSaved: {json_out}\nSaved: {csv_out}")


if __name__ == '__main__':
    main()



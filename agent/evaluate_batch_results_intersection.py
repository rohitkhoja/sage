#!/usr/bin/env python3
"""
Evaluate with strict intersection rule per question:
- candidates = intersection(agent_nodes, top_100) ordered as in top_100
- if intersection empty → candidates = top_100

Metrics computed:
- hit@1, hit@5, recall@20, mrr, mrr@20

Inputs:
- CSV: /shared/khoja/CogComp/output/BM25_stark_mag_human_rewritten.csv
- Agent outputs JSON: /shared/khoja/CogComp/agent/output/batch_questions/question_{query_id}.json

Outputs:
- summary.json and per_question.csv under /shared/khoja/CogComp/agent/output/batch_eval_intersection/
"""

import csv
import json
import os
import ast
from typing import List, Dict, Any
from statistics import mean

CSV_FILE = "/shared/khoja/CogComp/output/BM25_stark_mag_human_rewritten.csv"
AGENT_OUT_DIR = "/shared/khoja/CogComp/agent/output/batch_questions"
OUT_DIR = "/shared/khoja/CogComp/agent/output/batch_eval_intersection"


def parse_list(value: str) -> List[int]:
    if value is None:
        return []
    s = value.strip()
    if not s:
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            out = []
            for x in parsed:
                if isinstance(x, int):
                    out.append(x)
                elif isinstance(x, str) and x.isdigit():
                    out.append(int(x))
            return out
        return []
    except Exception:
        return []


def build_candidates(agent_nodes: List[int], top100: List[int]) -> List[int]:
    # If the agent retrieved fewer than 20 nodes, use them directly
    if len(agent_nodes) < 20:
        return list(agent_nodes)

    # Otherwise, use intersection with top_100 preserving top_100 order
    aset = set(agent_nodes)
    inter = [pid for pid in top100 if pid in aset]
    if inter:
        return inter
    return list(top100)


def hit_at_k(ranked: List[int], gold: List[int], k: int) -> float:
    g = set(gold)
    return 1.0 if any(pid in g for pid in ranked[:k]) else 0.0


def recall_at_k(ranked: List[int], gold: List[int], k: int) -> float:
    if not gold:
        return 0.0
    g = set(gold)
    return sum(1 for pid in ranked[:k] if pid in g) / float(len(g))


def mrr(ranked: List[int], gold: List[int]) -> float:
    g = set(gold)
    for i, pid in enumerate(ranked, 1):
        if pid in g:
            return 1.0 / i
    return 0.0


def mrr_at_k(ranked: List[int], gold: List[int], k: int) -> float:
    g = set(gold)
    for i, pid in enumerate(ranked[:k], 1):
        if pid in g:
            return 1.0 / i
    return 0.0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    per_rows: List[Dict[str, Any]] = []
    flagged_rows: List[Dict[str, Any]] = [] # agent < 10 and recall@20 == 0
    hit1s, hit5s, rec20s, mrrs, mrr20s = [], [], [], [], []

    for row in rows:
        try:
            qid = int(row['query_id'])
        except Exception:
            continue

        gold = parse_list(row.get('gold_docs', ''))
        top100 = parse_list(row.get('top_100', ''))

        agent_nodes: List[int] = []
        afile = os.path.join(AGENT_OUT_DIR, f"question_{qid}.json")
        if os.path.exists(afile):
            try:
                data = json.load(open(afile, 'r', encoding='utf-8'))
                nodes = data.get('final_nodes', [])
                if isinstance(nodes, list):
                    for x in nodes:
                        if isinstance(x, int):
                            agent_nodes.append(x)
                        elif isinstance(x, str) and x.isdigit():
                            agent_nodes.append(int(x))
            except Exception:
                agent_nodes = []

        ranked = build_candidates(agent_nodes, top100)

        h1 = hit_at_k(ranked, gold, 1)
        h5 = hit_at_k(ranked, gold, 5)
        r20 = recall_at_k(ranked, gold, 20)
        mr = mrr(ranked, gold)
        mr20 = mrr_at_k(ranked, gold, 20)

        hit1s.append(h1); hit5s.append(h5); rec20s.append(r20); mrrs.append(mr); mrr20s.append(mr20)

        row_out = {
            'query_id': qid,
            'question': row.get('query', ''),
            'num_gold': len(gold),
            'num_top100': len(top100),
            'num_agent_nodes': len(agent_nodes),
            'num_ranked_used': len(ranked),
            'hit@1': h1,
            'hit@5': h5,
            'recall@20': r20,
            'mrr': mr,
            'mrr@20': mr20,
        }
        per_rows.append(row_out)

        if len(agent_nodes) < 10 and r20 == 0.0:
            flagged_rows.append({
                'query_id': qid,
                'question': row.get('query', ''),
                'num_agent_nodes': len(agent_nodes),
                'recall@20': r20,
                'num_gold': len(gold),
                'num_top100': len(top100),
            })

    summary = {
        'count': len(per_rows),
        'hit@1': mean(hit1s) if hit1s else 0.0,
        'hit@5': mean(hit5s) if hit5s else 0.0,
        'recall@20': mean(rec20s) if rec20s else 0.0,
        'mrr': mean(mrrs) if mrrs else 0.0,
        'mrr@20': mean(mrr20s) if mrr20s else 0.0,
    }

    json_out = os.path.join(OUT_DIR, 'summary.json')
    with open(json_out, 'w', encoding='utf-8') as jf:
        json.dump({'overall': summary, 'per_question': per_rows}, jf, indent=2)

    csv_out = os.path.join(OUT_DIR, 'per_question.csv')
    with open(csv_out, 'w', encoding='utf-8', newline='') as cf:
        import csv as _csv
        fields = list(per_rows[0].keys()) if per_rows else ['query_id','question','num_gold','num_top100','num_agent_nodes','num_ranked_used','hit@1','hit@5','recall@20','mrr','mrr@20']
        w = _csv.DictWriter(cf, fieldnames=fields)
        w.writeheader()
        for r in per_rows:
            w.writerow(r)

    flagged_csv = os.path.join(OUT_DIR, 'agent_lt10_zero_recall.csv')
    with open(flagged_csv, 'w', encoding='utf-8', newline='') as ff:
        import csv as _csv
        fields = ['query_id','question','num_agent_nodes','recall@20','num_gold','num_top100']
        w = _csv.DictWriter(ff, fieldnames=fields)
        w.writeheader()
        for r in flagged_rows:
            w.writerow(r)

    print("\n=== Intersection-Only Metrics ===")
    for k, v in summary.items():
        if k == 'count':
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.6f}")
    print(f"\nSaved: {json_out}\nSaved: {csv_out}\nSaved: {flagged_csv}")

    if flagged_rows:
        print("\nQuestions where agent nodes < 10 and recall@20 == 0:")
        for r in flagged_rows:
            print(f"- query_id={r['query_id']}, agent_nodes={r['num_agent_nodes']}, recall@20={r['recall@20']}, question={r['question'][:120]}")


if __name__ == '__main__':
    main()



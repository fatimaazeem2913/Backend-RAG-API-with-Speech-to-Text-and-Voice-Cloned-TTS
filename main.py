import os
import sys
import json
import time
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.rag_service import EnterpriseRAGService


def run_benchmark():
    print("================================================================================")
    print("Day 21: Enterprise RAG System Delivery — 20-Question Full Evaluation Matrix")
    print("================================================================================\n")

    eval_path = os.path.join(os.path.dirname(__file__), "data/evaluation_set.json")
    if not os.path.exists(eval_path):
        print(f"Error: Evaluation set not found at {eval_path}")
        return

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    rag_service = EnterpriseRAGService()
    strategies = ["dense", "bm25", "hybrid", "hierarchical"]
    
    results = {s: [] for s in strategies}
    metrics = {
        s: {
            "total_questions": len(eval_data),
            "correct_answers": 0,
            "citations_found": 0,
            "hallucination_count": 0,
            "total_latency_sec": 0.0
        }
        for s in strategies
    }

    for item in eval_data:
        q_id = item["id"]
        question = item["question"]
        expected_src = item["expected_source"]
        
        print(f"\n▶ Question {q_id}: {question}")

        for strat in strategies:
            session_id = f"eval_{strat}_q{q_id}"
            start_t = time.time()
            
            res = rag_service.chat(session_id=session_id, message=question, strategy=strat)
            lat = time.time() - start_t
            
            answer = res.get("answer", "")
            citations = res.get("citations", [])

            # Measure Precision & Grounding
            has_expected_citation = any(expected_src in c for c in citations) if expected_src != "None" else True
            is_unanswerable_expected = expected_src == "None" or "does not contain" in item["expected_answer"].lower()
            
            is_correct = False
            if is_unanswerable_expected:
                if "does not contain" in answer.lower():
                    is_correct = True
            else:
                if "does not contain" not in answer.lower() and len(answer) > 20:
                    is_correct = True

            if is_correct:
                metrics[strat]["correct_answers"] += 1
            if has_expected_citation and not is_unanswerable_expected:
                metrics[strat]["citations_found"] += 1
            if not is_correct and not is_unanswerable_expected and "does not contain" not in answer.lower():
                metrics[strat]["hallucination_count"] += 1
                
            metrics[strat]["total_latency_sec"] += lat

            status = "✓ CORRECT" if is_correct else "✗ MISS"
            print(f"  [{strat.upper():<12}] {status} | Latency: {round(lat, 2)}s")

            results[strat].append({
                "question_id": q_id,
                "question": question,
                "strategy": strat,
                "standalone_query": res.get("standalone_query"),
                "answer": answer,  # Saves the full, untruncated answer
                "citations": citations,
                "latency_sec": round(lat, 3),
                "is_correct": is_correct
            })
            
            time.sleep(0.3)

    # Summary Calculations
    summary = {}
    print("\n================================================================================")
    print("FINAL BENCHMARK RESULTS MATRIX")
    print("================================================================================")
    print(f"{'Strategy':<16} | {'Accuracy':<10} | {'Citation Prec':<14} | {'Hallucination':<14} | {'Avg Latency'}")
    print("--------------------------------------------------------------------------------")

    for strat in strategies:
        m = metrics[strat]
        acc = (m["correct_answers"] / m["total_questions"]) * 100
        prec = (m["citations_found"] / (m["total_questions"] - 2)) * 100
        hal = (m["hallucination_count"] / m["total_questions"]) * 100
        avg_lat = m["total_latency_sec"] / m["total_questions"]

        summary[strat] = {
            "mean_accuracy_pct": round(acc, 2),
            "citation_precision_pct": round(prec, 2),
            "hallucination_rate_pct": round(hal, 2),
            "avg_latency_ms": round(avg_lat * 1000, 2)
        }
        print(f"{strat.upper():<16} | {acc:>8.1f}% | {prec:>12.1f}% | {hal:>12.1f}% | {round(avg_lat*1000, 1)} ms")

    out_file = os.path.join(os.path.dirname(__file__), "outputs/evaluation_matrix.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print("\n✓ Full evaluation with complete answers saved to outputs/evaluation_matrix.json\n")


if __name__ == "__main__":
    run_benchmark()
"""
Benchmark Previous Chats Runner
Evaluates generated scientific scripts from recent chat sessions in SQLite against the 100-point rubric.
"""

import sys
import os
import json
import sqlite3
import argparse
from typing import List, Dict, Any, Optional

# Ensure UTF-8 stdout on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure repository root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from backend.config import settings
from backend.db.models import get_db_connection, insert_evaluation
from backend.evaluation.script_evaluator import ScriptEvaluator


def get_recent_chat_sessions(conn: sqlite3.Connection, limit: int = 2, specific_session_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Retrieves the most recent chat sessions that contain both a user prompt and an assistant reply."""
    cursor = conn.cursor()
    
    if specific_session_id:
        query = """
            SELECT s.id, s.title, s.created_at, s.last_message_at
            FROM sessions s
            WHERE s.id = ?
        """
        cursor.execute(query, (specific_session_id,))
        sessions = [dict(r) for r in cursor.fetchall()]
    else:
        # Fetch sessions that actually have assistant messages, ordered newest first
        query = """
            SELECT DISTINCT s.id, s.title, s.created_at, s.last_message_at
            FROM sessions s
            JOIN messages m ON s.id = m.session_id
            WHERE m.role = 'assistant'
            ORDER BY s.id DESC
            LIMIT ?
        """
        cursor.execute(query, (limit,))
        sessions = [dict(r) for r in cursor.fetchall()]
    
    # For each session, fetch messages
    results = []
    for s in sessions:
        cursor.execute(
            """
            SELECT id, role, content, agent_used, timestamp
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (s["id"],)
        )
        msgs = [dict(r) for r in cursor.fetchall()]
        
        user_msgs = [m for m in msgs if m["role"] == "user"]
        asst_msgs = [m for m in msgs if m["role"] == "assistant"]
        
        if user_msgs and asst_msgs:
            # Most recent user query and assistant response in this conversation
            prompt = user_msgs[-1]["content"]
            response = asst_msgs[-1]["content"]
            agent_used = asst_msgs[-1].get("agent_used") or "UnknownAgent"
            
            results.append({
                "session_id": s["id"],
                "title": s["title"],
                "created_at": s["created_at"],
                "prompt": prompt,
                "response": response,
                "agent_used": agent_used,
                "message_count": len(msgs)
            })
            
    return results


def render_progress_bar(pct: float, width: int = 14) -> str:
    """Renders a Unicode text progress bar."""
    filled = int(round(width * (pct / 100.0)))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def format_grade_badge(grade: str) -> str:
    """Returns a visual badge for the letter grade."""
    badges = {
        "A": "[ GRADE A - EXCELLENT ]",
        "B": "[ GRADE B - GOOD ]",
        "C": "[ GRADE C - SATISFACTORY ]",
        "D": "[ GRADE D - MARGINAL ]",
        "F": "[ GRADE F - FAILED ]"
    }
    return badges.get(grade.upper(), f"[ GRADE {grade} ]")


def run_benchmark_on_chats(limit: int = 2, session_id: Optional[int] = None, save_to_db: bool = True, show_code: bool = False):
    """Executes the benchmark against the target recent chats."""
    conn = get_db_connection()
    evaluator = ScriptEvaluator()
    
    print("\n" + "=" * 78)
    print(" 🧪 SciCompute AI — Scientific Code Benchmark for Recent Chats")
    print("=" * 78)
    
    sessions = get_recent_chat_sessions(conn, limit=limit, specific_session_id=session_id)
    
    if not sessions:
        print("\n❌ No eligible chat sessions with assistant messages found in chatbox.db.")
        conn.close()
        return

    print(f"Loaded {len(sessions)} chat session(s) from database: {settings.SQLITE_PATH}\n")
    
    benchmark_reports = []

    for idx, sess in enumerate(sessions, 1):
        s_id = sess["session_id"]
        title = sess["title"]
        prompt = sess["prompt"]
        raw_response = sess["response"]
        agent_used = sess["agent_used"]
        
        print("-" * 78)
        print(f" [{idx}/{len(sessions)}] Chat Session #{s_id}: \"{title}\"")
        print(f"      Created: {sess['created_at']}  |  Agent: {agent_used}")
        print(f"      Prompt: {prompt[:120]}..." if len(prompt) > 120 else f"      Prompt: {prompt}")
        print("-" * 78)
        
        # Execute Evaluation
        start_eval = True
        eval_result = evaluator.evaluate(raw_response, prompt=prompt, session_id=s_id if save_to_db else None)
        
        score = eval_result.get("score", 0.0)
        grade = eval_result.get("grade", "F")
        eval_type = eval_result.get("evaluation_type", "unknown")
        category_scores = eval_result.get("category_scores", {})
        passed_checks = eval_result.get("passed_checks", [])
        failed_checks = eval_result.get("failed_checks", [])
        extracted_code = eval_result.get("extracted_code", "")
        summary = eval_result.get("summary", "")
        
        code_lines = len(extracted_code.splitlines()) if extracted_code else 0
        
        # Display Results
        print(f"\n  🎯 Result: {score:5.1f} / 100.0  {format_grade_badge(grade)}")
        print(f"  📌 Rubric Mode: {eval_type.upper()}  |  Extracted Python Lines: {code_lines}")
        print(f"  📊 Checks: {len(passed_checks)} Passed, {len(failed_checks)} Failed")
        
        print("\n  Category Breakdown:")
        MAX_BY_CAT = {"execution_integrity": 25, "scientific_accuracy": 35, "task_compliance": 25, "code_quality": 15}
        for cat_key, cat_val in category_scores.items():
            if isinstance(cat_val, dict):
                cat_score = cat_val.get("score", 0)
                cat_max = cat_val.get("max", 25)
                cat_pct = cat_val.get("pct", int(round(100 * cat_score / max(cat_max, 1))))
            else:
                try:
                    cat_score = int(cat_val)
                except (ValueError, TypeError):
                    cat_score = 0
                cat_max = MAX_BY_CAT.get(cat_key, 25)
                cat_pct = int(round(100 * cat_score / max(cat_max, 1)))
            bar = render_progress_bar(cat_pct)
            cat_name = cat_key.replace("_", " ").title()
            print(f"    • {cat_name:<30} {bar} {cat_score:2d}/{cat_max:2d} pts ({cat_pct:3d}%)")
            
        if passed_checks:
            print(f"\n  ✅ Top Passed Checks ({min(4, len(passed_checks))} of {len(passed_checks)}):")
            for p in passed_checks[:4]:
                print(f"     [+] {p}")
                
        if failed_checks:
            print(f"\n  ⚠️ Failed / Missing Checks ({len(failed_checks)} total):")
            for f in failed_checks[:5]:
                print(f"     [-] {f}")
            if len(failed_checks) > 5:
                print(f"     ... and {len(failed_checks) - 5} more")

        print(f"\n  📝 Scientific Assessment:")
        print(f"     \"{summary}\"")
        
        if show_code and extracted_code:
            print(f"\n  💻 Extracted Code Preview (First 35 lines):")
            for line_no, line in enumerate(extracted_code.splitlines()[:35], 1):
                print(f"     {line_no:3d} | {line}")
            if code_lines > 35:
                print(f"     ... ({code_lines - 35} more lines)")
                
        # Persist to database evaluations table
        if save_to_db:
            eval_id = eval_result.get("evaluation_id")
            if not eval_id:
                try:
                    eval_id = insert_evaluation(
                        session_id=s_id,
                        model_name=eval_result.get("model", settings.MODEL_NAME),
                        prompt=prompt,
                        generated_code=extracted_code,
                        score=score,
                        grade=grade,
                        category_scores_json=json.dumps(category_scores),
                        evaluation_report_json=json.dumps({
                            "passed_checks": passed_checks,
                            "failed_checks": failed_checks,
                            "summary": summary,
                            "evaluation_type": eval_type
                        }),
                        latency_seconds=0.0
                    )
                    print(f"\n  💾 Stored evaluation report into database (Eval ID #{eval_id}).")
                    print(f"     This will now appear in the frontend 'Accuracy Benchmarks' sidebar list.")
                except Exception as e:
                    print(f"  ⚠️ Could not save evaluation to DB: {e}")
            else:
                print(f"\n  💾 Stored evaluation report into database (Eval ID #{eval_id}).")
                print(f"     This will now appear in the frontend 'Accuracy Benchmarks' sidebar list.")
                
        print("\n" + "=" * 78)
        
        benchmark_reports.append({
            "session_id": s_id,
            "title": title,
            "score": score,
            "grade": grade,
            "eval_type": eval_type,
            "passed": len(passed_checks),
            "failed": len(failed_checks)
        })

    # Summary table
    print("\n📋 Benchmark Execution Summary:")
    print("-" * 65)
    print(f" {'Session ID':<12} | {'Score':<10} | {'Grade':<8} | {'Type':<16} | Title")
    print("-" * 65)
    for rep in benchmark_reports:
        print(f" #{rep['session_id']:<11} | {rep['score']:<5.1f}/100 | {rep['grade']:<8} | {rep['eval_type']:<16} | {rep['title'][:20]}...")
    print("-" * 65)
    avg_score = sum(r["score"] for r in benchmark_reports) / len(benchmark_reports)
    print(f" Average Benchmark Score across {len(benchmark_reports)} session(s): {avg_score:.1f} / 100.0\n")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark previous frontend chat sessions against the scientific rubric.")
    parser.add_argument(
        "-n", "--count",
        type=int,
        default=2,
        help="Number of recent chat sessions to benchmark (default: 2)"
    )
    parser.add_argument(
        "-s", "--session-id",
        type=int,
        default=None,
        help="Benchmark a specific session ID instead of the most recent ones"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save evaluation records into the SQLite database"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print extracted Python code previews in the terminal output"
    )
    
    args = parser.parse_args()
    
    run_benchmark_on_chats(
        limit=args.count,
        session_id=args.session_id,
        save_to_db=not args.no_save,
        show_code=args.verbose
    )


if __name__ == "__main__":
    main()

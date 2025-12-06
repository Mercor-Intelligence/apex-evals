"""
Minimal single-task grading example.

Always create and activate a virtual environment before running this script.
"""

from dotenv import load_dotenv
import json

from grading import (
    GradingModelConfig,
    GradingResult,
    GradingTask,
    run_grading_task,
)


def main() -> None:
    # Load environment variables from .env (e.g. GOOGLE_API_KEY for Gemini)
    load_dotenv()

    solution = "Machine learning is a subset of AI where systems learn patterns from data."

    # Original rubric expressed as JSON (matches how dataset rubrics are stored).
    rubric_json = json.dumps(
        {
            "criterion 1": {
                "description": "Defines machine learning in a factually correct way.",
                "weight": "Primary objective(s)",
                "criterion_type": ["Reasoning"],
            },
            "criterion 2": {
                "description": "Mentions that ML systems learn from data or examples.",
                "weight": "Secondary objective(s)",
                "criterion_type": ["Factual"],
            },
        },
        ensure_ascii=False,
    )

    task = GradingTask(
        solution=solution,
        rubric=rubric_json,
        grading_model=GradingModelConfig(
            model_id="gemini-2.5-flash",
            max_tokens=2000,
            temperature=0.01,
        ),
    )

    result: GradingResult = run_grading_task(task)

    print("Grading result:")
    print(f"  score={result.points_earned}/{result.points_possible}")
    print(f"  percentage={result.percentage_score}%")

    if not result.criteria_results:
        return

    print("\nPer-criterion breakdown:")
    for criterion in result.criteria_results:
        key = criterion.get("criterion_key", "criterion")
        status = "PASS" if criterion.get("autorating") else "FAIL"
        reason = criterion.get("reason", "")
        print(f"- {key}: {status}")
        if reason:
            print(f"  reason: {reason}")

    grading_rubrics = json.loads(rubric_json)
    for criterion in result.criteria_results:
        key = criterion.get("criterion_key")
        if key in grading_rubrics and isinstance(grading_rubrics[key], dict):
            grading_rubrics[key]["autorating"] = bool(criterion.get("autorating"))
            grading_rubrics[key]["reason"] = criterion.get("reason", "")

    grading_rubrics_json = json.dumps(grading_rubrics, ensure_ascii=False, indent=2)

    print(grading_rubrics_json)


if __name__ == "__main__":
    main()



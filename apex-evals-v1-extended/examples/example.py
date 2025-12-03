"""
Runs a generation and grading task.
Requires API keys (e.g. OPENAI_API_KEY) and a virtual env.
"""

import logging
from generation import Attachment, GenerationTask, ModelConfig, run_generation_task
from grading import GradingTask, GradingModelConfig, run_grading_task
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    # Simple prompt asking for a product description
    prompt = """
    Write a compelling product description for a new smartphone called "TechPhone X1".
    
    Key features to highlight:
    - 6.5 inch OLED display
    - 108MP triple camera system
    - 5000mAh battery with fast charging
    - 5G connectivity
    - Water resistant (IP68 rating)
    
    The description should be professional, engaging, and no more than 200 words.
    """
    
    # No attachments for this simple example
    attachments = []

    generation_task = GenerationTask(
        prompt=prompt,
        models=[
            ModelConfig(
                model_id="gemini-2.5-flash",
                max_tokens=1000,
                max_input_tokens=10000,
                temperature=0.7,
            ),
        ],
        attachments=attachments,
    )

    generation_result = run_generation_task(generation_task)
    print("Generation Result Summary:")
    print(
        f"  completed={generation_result.completed} "
        f"failed={generation_result.failed} "
        f"tokens={generation_result.total_tokens} "
        f"cost=${generation_result.total_cost:.4f}"
    )

    successful_response = next(
        (result for result in generation_result.results if result.get("success")),
        None,
    )
    if not successful_response:
        print("No successful generations; skipping grading.")
        return

    # Define a simple rubric to grade the product description
    rubric = [
        {
            "criterion_1": {
                "description": "Mentions the product name 'TechPhone X1'",
                "sources": "",
                "justification": "",
                "weight": "Primary objective(s)",
                "human_rating": "False",
                "criterion_type": ["Factual"],
                "dependent_criteria": []
            }
        },
        {
            "criterion_2": {
                "description": "Mentions the 6.5 inch OLED display feature",
                "sources": "",
                "justification": "",
                "weight": "Primary objective(s)",
                "human_rating": "False",
                "criterion_type": ["Factual"],
                "dependent_criteria": []
            }
        },
        {
            "criterion_3": {
                "description": "Mentions the 108MP camera system",
                "sources": "",
                "justification": "",
                "weight": "Primary objective(s)",
                "human_rating": "False",
                "criterion_type": ["Factual"],
                "dependent_criteria": []
            }
        },
        {
            "criterion_4": {
                "description": "Mentions the battery (5000mAh) and/or fast charging",
                "sources": "",
                "justification": "",
                "weight": "Primary objective(s)",
                "human_rating": "False",
                "criterion_type": ["Factual"],
                "dependent_criteria": []
            }
        },
        {
            "criterion_5": {
                "description": "The description is professional and engaging in tone",
                "sources": "",
                "justification": "",
                "weight": "Primary objective(s)",
                "human_rating": "False",
                "criterion_type": ["Reasoning"],
                "dependent_criteria": []
            }
        },
        {
            "criterion_6": {
                "description": "The description is approximately 200 words or less",
                "sources": "",
                "justification": "",
                "weight": "Primary objective(s)",
                "human_rating": "False",
                "criterion_type": ["Reasoning"],
                "dependent_criteria": []
            }
        }
    ]

    grading_task = GradingTask(
        solution=successful_response["response"],
        rubric=rubric,
        grading_model=GradingModelConfig(
            model_id="gemini-2.5-flash",
            max_tokens=1000000,
            temperature=0.01,
        ),
    )

    grading_result = run_grading_task(grading_task)
    print("Grading Result Summary:")
    print(
        f"  score={grading_result.points_earned}/{grading_result.points_possible} "
        f"({grading_result.percentage_score}%) "
        f"criteria={len(grading_result.criteria_results)}"
    )

    if grading_result.criteria_results:
        print("\nCriterion breakdown:")
        for criterion in grading_result.criteria_results:
            status = "PASS" if criterion.get("autorating") else "FAIL"
            print(f"- {criterion.get('criterion_key', 'criterion')} -> {status}")
            print(f"  reason: {criterion.get('reason', 'n/a')}")


if __name__ == "__main__":
    main()



import csv
import json
from pathlib import Path

from agent.classification import classify_email
from agent.schemas import EmailInput


ROOT = Path(__file__).resolve().parent.parent
EMAILS_PATH = ROOT / "data" / "emails.csv"
EVAL_PATH = ROOT / "evals" / "eval_cases.json"

TEST_IDS = ["E013", "E015", "E025"]


def main():
    """Test classification against labeled evaluation cases."""

    with EMAILS_PATH.open(encoding="utf-8-sig", newline="") as file:
        emails = {
            row["email_id"]: row
            for row in csv.DictReader(file)
        }

    with EVAL_PATH.open(encoding="utf-8") as file:
        cases = {
            case["email_id"]: case
            for case in json.load(file)
        }

    for email_id in TEST_IDS:
        email = EmailInput.model_validate(emails[email_id])
        expected = cases[email_id]

        print(f"\n===== {email_id} =====")

        try:
            result = classify_email(email)

            print("Predicted:", result.model_dump())
            print("Expected:", {
                "intent": expected["expected_intent"],
                "primary_department": expected["expected_department"],
                "secondary_department":
                    expected["expected_secondary_department"],
                "priority": expected["expected_priority"]
            })

            print("Intent correct:",
                  result.intent == expected["expected_intent"])

            print("Department correct:",
                  result.primary_department ==
                  expected["expected_department"])

            print("Priority correct:",
                  result.priority == expected["expected_priority"])

        except Exception as error:
            print("Classification failed:", error)


if __name__ == "__main__":
    main()

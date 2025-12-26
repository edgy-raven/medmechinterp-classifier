import hashlib
import json
import re

OUTPUT_FILENAME = "first_responses.json"
_CASE_PROMPT_PATTERN = re.compile(
    r"CASE PRESENTATION\s*-+\s*(.*?)\s*-+\s*OUTPUT TEMPLATE",
    re.DOTALL,
)


def _hash_case_prompt(case_prompt):
    digest = hashlib.sha256(case_prompt.encode("utf-8")).hexdigest()
    return f"case_{digest}"


def _extract_case_prompt(text):
    match = _CASE_PROMPT_PATTERN.search(text)
    if not match:
        return None
    return match.group(1).strip()


def inputs(context):
    return {
        "input_root": context["input_root"],
        "output_root": context["output_root"],
    }


def outputs(context):
    return {"first_response_path": context["output_root"] / OUTPUT_FILENAME}


def run(input_root, output_root):
    collated_path = output_root / OUTPUT_FILENAME

    combined = []
    all_files = []
    for model_dir in input_root.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            for path in sorted(dataset_dir.glob("responses_*.graded.json")):
                if ".second." in path.name:
                    continue
                all_files.append((model_name, path))

    for model_name, path in all_files:
        data = json.load(path.open())
        for entry in data:
            if "pmcid" not in entry:
                continue
            combined.append(
                {
                    "pmcid": entry["pmcid"],
                    "model": model_name,
                    "is_correct": bool(entry.get("is_correct")),
                }
            )

    payload = {"first_response_grades": combined}
    with collated_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"Collated {len(all_files)} file(s) into {collated_path} with {len(combined)} responses")

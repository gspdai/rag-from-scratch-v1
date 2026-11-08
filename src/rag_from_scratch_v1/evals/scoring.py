import json

def response_accuracy(judge_path: str) -> tuple[list[dict], dict]:
    with open(judge_path, "r", encoding="utf-8") as file:
        responses = json.load(file)

    per_question = []
    category_average = {}

    for response in responses:
        category = response["question_category"]
        if category != "not_in_document":
            expected_pages = set(response["expected_pages"])
            actual_pages = set(response["citations"])
            correct = len(actual_pages & expected_pages)
            precision = correct / len(actual_pages) if actual_pages else 0
            recall = correct / len(expected_pages)
        else:
            precision = 0
            recall = 0

        per_question.append({
            "question": response["question"],
            "category": category,
            "precision": precision,
            "recall": recall,
        })

        if category not in category_average:
            category_average[category] = {"precision": 0, "recall": 0, "instances": 0, 
                                          "faithfulness":0, "relevance":0, "correctness":0}
        category_average[category]["precision"] += precision
        category_average[category]["recall"] += recall
        category_average[category]["faithfulness"] += response["faithfulness"]["score"]
        category_average[category]["relevance"] += response["relevance"]["score"]
        category_average[category]["correctness"] += response["correctness"]["score"]
        category_average[category]["instances"] += 1


    for scores in category_average.values():
        scores["precision"] = scores["precision"] / scores["instances"]
        scores["recall"] = scores["recall"] / scores["instances"]
        scores["faithfulness"] = scores["faithfulness"] / scores["instances"]
        scores["relevance"] = scores["relevance"] / scores["instances"]
        scores["correctness"] = scores["correctness"] / scores["instances"]

    return per_question, category_average
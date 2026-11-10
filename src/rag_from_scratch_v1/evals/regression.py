import json

def comparison(baseline_path: str, score:dict, threshold:float) -> tuple[dict, bool]:
    with open(baseline_path, encoding="utf-8") as file:
        baseline_result = json.load(file)

    comparison_dict = {}
    passed = True

    for key, value in baseline_result.items():
        if key not in comparison_dict:
            comparison_dict[key] = {"precision": 0, 
                                "recall": 0, 
                                "faithfulness":0, 
                                "relevance":0, 
                                "correctness":0}
        for measure_key in value.keys():
            if measure_key == "instances":
                continue
            difference = score[key][measure_key] - value[measure_key]
            comparison_dict[key][measure_key] = difference
            if difference < -threshold:
                passed = False

    return comparison_dict, passed


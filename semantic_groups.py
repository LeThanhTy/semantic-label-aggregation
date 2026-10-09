from collections import defaultdict


def group_predictions(predictions, label_to_group, pretty_group):
    """Aggregate fine-grained predictions without changing the model output."""
    groups = defaultdict(list)
    for item in predictions:
        label = item["label"]
        raw_group = label_to_group.get(label)
        if raw_group is not None:
            groups[pretty_group(raw_group)].append(item)

    return dict(groups)

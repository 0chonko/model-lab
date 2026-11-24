"""Score entry-level layout boxes with one-to-one IoU matching."""

from math import isfinite


def iou(a, b):
    for box in (a, b):
        if len(box) != 4 or not all(map(isfinite, box)):
            raise ValueError("expected four finite coordinates")
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("expected positive-area xyxy box")
    overlap_x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    overlap_y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = overlap_x * overlap_y
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def box_f1(predicted, gold, threshold=0.5):
    """Maximum matching prevents duplicate predictions from earning extra credit."""
    if not 0 < threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    edges = [
        [index for index, target in enumerate(gold) if iou(box, target) >= threshold]
        for box in predicted
    ]
    assigned = {}

    def match(prediction, seen):
        for target in edges[prediction]:
            if target in seen:
                continue
            seen.add(target)
            if target not in assigned or match(assigned[target], seen):
                assigned[target] = prediction
                return True
        return False

    for prediction in range(len(predicted)):
        match(prediction, set())
    count = len(predicted) + len(gold)
    return 2 * len(assigned) / count if count else 1.0


if __name__ == "__main__":
    gold = [(0, 0, 100, 20), (0, 20, 100, 40)]
    merged = [(0, 0, 100, 40)]
    for threshold in (0.5, 0.75):
        print(f"IoU >= {threshold}: merged F1={box_f1(merged, gold, threshold):.3f}, "
              f"split F1={box_f1(gold, gold, threshold):.3f}")

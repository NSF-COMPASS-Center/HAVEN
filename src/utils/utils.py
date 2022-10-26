
def get_label_vocabulary(labels):
    print(labels)
    label_idx_map = {}
    idx_label_map = {}

    for idx, label in enumerate(labels):
        label_idx_map[label] = idx
        idx_label_map[idx] = label
    return label_idx_map, idx_label_map

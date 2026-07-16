import csv


OUT_DIR = 'out_105'
FOLD_NUM = 5


def update(my_dict, key, value):
    if key in my_dict.keys():
        my_dict[key] += value
    else:
        my_dict[key] = value


def test():
    id_to_score = dict()    # key: mp_id, value: score sum

    for fold_it in range(FOLD_NUM):
        with open(f"{OUT_DIR}/loss_{fold_it}/loss_hi.csv", "r", newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                update(id_to_score, row[0], float(row[1]) / (FOLD_NUM - 2))     # k-2 train folds
    
    sorted_id_to_scores = sorted(list(id_to_score.items()), key=lambda x: x[1], reverse=True)
    
    with open(f"{OUT_DIR}/loss_hi_total.csv", "w", newline='', encoding='utf-8') as f:
        sorted_id_to_scores = [('MP ID', 'Score')] + sorted_id_to_scores
        writer = csv.writer(f)
        writer.writerows(sorted_id_to_scores)
    pass


if __name__ == "__main__":
    test()
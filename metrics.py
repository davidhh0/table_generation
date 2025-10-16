import statistics

import numpy as np
from collections import defaultdict
import pandas as pd

def get_metrics():
    import json
    import matplotlib.pyplot as plt
    import diskcache
    import git
    working_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    tbl_generated_db = diskcache.Cache(f'{working_dir}/local_dbs/tables/generated_tables.db')
    graph_cells = [(20, 100), (101, 250), (251, 500), (501, 1000), (1001, float('inf'))]
    cells_f1_scores_per_range = [[] for _ in graph_cells]
    graph_numeric_ratio = [(0.0, 0.0), (0.0, 0.67), (0.67, 1.0), (1.0, 1.0)]
    numeric_ratio_f1_scores_per_range = [[] for _ in graph_numeric_ratio]
    graph_popularity = [(0, 100), (101, 1000), (1001, 10_000), (10_001, float('inf'))]
    popularity_f1_scores_per_range = [[] for _ in graph_popularity]
    graph_num_of_rows = [(0, 10), (11, 50), (51, 100), (101, 500), (501, float('inf'))]
    num_of_rows_f1_scores_per_range = [[] for _ in graph_num_of_rows]
    graph_num_of_columns = [(0, 3), (4, 6), (7, 10), (11, 15), (16, float('inf'))]
    num_of_columns_f1_scores_per_range = [[] for _ in graph_num_of_columns]
    dtype_scores = dict()
    histogram_dict = defaultdict(list)
    raw_csv = []
    for key in tbl_generated_db.iterkeys():
        record = json.loads(tbl_generated_db.get(key))
        if 'scores' not in record or 'dtype_scores' not in record['scores']:
            continue
        f1_score = record['scores']['f1_score']
        numeric_ratio = len(
            [k for k in record['columns'].values() if k in ['int64', 'float64']]
        ) / len(record['columns'])
        popularity = record['article_metadata']['popularity']
        num_of_cells = record['shape'][0] * record['shape'][1]
        histogram_dict['numeric_ratio'].append(numeric_ratio)
        histogram_dict['popularity'].append(popularity)
        histogram_dict['num_of_cells'].append(num_of_cells)
        histogram_dict['num_of_rows'].append(record['shape'][0])
        histogram_dict['num_of_columns'].append(record['shape'][1])
        raw_csv.append({
            'article': record['article_name'],
            'url': record['url'],
            'dates': record['dates'],
            'rows': record['shape'][0],
            'columns': record['shape'][1],
            'cells': num_of_cells,
            'numeric_ratio': numeric_ratio,
            'popularity': popularity,
            'f1_score': f1_score,
        })
        for dtype, _dict in record['scores']['dtype_scores'].items():
            if dtype not in dtype_scores:
                dtype_scores[dtype] = {'recall': [], 'precision': [], 'f1_score': []}
            dtype_scores[dtype]['f1_score'].extend(_dict['f1_score'])
            dtype_scores[dtype]['recall'].extend(_dict['recall'])
            dtype_scores[dtype]['precision'].extend(_dict['precision'])

        for i, rng in enumerate(graph_cells):
            if rng[1] == float('inf'):
                if num_of_cells >= rng[0]:
                    cells_f1_scores_per_range[i].append(f1_score)
            else:
                if rng[0] <= num_of_cells <= rng[1]:
                    cells_f1_scores_per_range[i].append(f1_score)
        # By numeric ratio
        for i, rng in enumerate(graph_numeric_ratio):
            if rng[0] == rng[1]:
                if numeric_ratio == rng[0]:
                    numeric_ratio_f1_scores_per_range[i].append(f1_score)
            else:
                if rng[0] < numeric_ratio <= rng[1]:
                    numeric_ratio_f1_scores_per_range[i].append(f1_score)
        # By popularity
        for i, rng in enumerate(graph_popularity):
            if rng[1] == float('inf'):
                if popularity >= rng[0]:
                    popularity_f1_scores_per_range[i].append(f1_score)
            else:
                if rng[0] <= popularity <= rng[1]:
                    popularity_f1_scores_per_range[i].append(f1_score)
        # By number of rows
        for i, rng in enumerate(graph_num_of_rows):
            if rng[1] == float('inf'):
                if record['shape'][0] >= rng[0]:
                    num_of_rows_f1_scores_per_range[i].append(f1_score)
            else:
                if rng[0] <= record['shape'][0] <= rng[1]:
                    num_of_rows_f1_scores_per_range[i].append(f1_score)
        # By number of columns
        for i, rng in enumerate(graph_num_of_columns):
            if rng[1] == float('inf'):
                if record['shape'][1] >= rng[0]:
                    num_of_columns_f1_scores_per_range[i].append(f1_score)
            else:
                if rng[0] <= record['shape'][1] <= rng[1]:
                    num_of_columns_f1_scores_per_range[i].append(f1_score)


    for key, values in histogram_dict.items():
        plt.figure()
        limit = 100
        if key == 'num_of_cells':
            limit = 1000
        if key == 'popularity':
            limit = 10000
        plt.hist([k for k in map(lambda x:min(x, limit), values)], bins=10, color='skyblue', edgecolor='black',)
        plt.xlabel(key.replace('_', ' ').title())
        plt.ylabel('Count')
        plt.title(f'Histogram of {key.replace("_", " ").title()}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # By data type
    dtype_avg_f1_scores = {
        dtype: statistics.mean(scores['f1_score']) if scores else 0
        for dtype, scores in dtype_scores.items()
    }
    dtype_std = {
        dtype: np.std(scores['f1_score']) if scores else 0
        for dtype, scores in dtype_scores.items()
    }
    dtype_labels = [f"{i} (n={len(dtype_scores[i]['f1_score'])})" for i in dtype_avg_f1_scores.keys()]
    dtype_avg_f1_scores = list(dtype_avg_f1_scores.values())
    dtype_std = list(dtype_std.values())
    lower_errors = [min(avg, std) for avg, std in zip(dtype_avg_f1_scores, dtype_std)]
    upper_errors = dtype_std
    plt.errorbar(
        dtype_labels, dtype_avg_f1_scores, yerr=[lower_errors, upper_errors], marker='o', capsize=5
    )
    plt.xlabel('Data Type')
    plt.ylabel('Average F1 Score')
    plt.title('Average F1 Score by Data Type')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # By number of cells
    cells_avg_f1_scores = [
        sum(scores) / len(scores) if scores else 0
        for scores in cells_f1_scores_per_range
    ]
    cells_counts = [len(scores) for scores in cells_f1_scores_per_range]
    cells_labels = [
        f"{rng[0]}+" if rng[1] == float('inf') else f"{rng[0]}-{rng[1]}"
        for rng in graph_cells
    ]
    cells_labels = [
        f"{label}\n(n={count})" for label, count in zip(cells_labels, cells_counts)
    ]
    cells_std = [
        np.std(scores) if scores else 0 for scores in cells_f1_scores_per_range
    ]
    plt.errorbar(
        cells_labels, cells_avg_f1_scores, yerr=cells_std, marker='o', capsize=5
    )
    plt.xlabel('Number of Cells Range')
    plt.ylabel('Average F1 Score')
    plt.title('Average F1 Score by Number of Cells')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # ================================



    # By numeric ratio
    numeric_ratio_avg_f1_scores = [
        sum(scores) / len(scores) if scores else 0
        for scores in numeric_ratio_f1_scores_per_range
    ]
    numeric_ratio_labels = [
        f"{rng[0]}" if rng[0] == rng[1] else f"({rng[0]}, {rng[1]}]"
        for rng in graph_numeric_ratio
    ]
    numeric_ratio_counts = [len(scores) for scores in numeric_ratio_f1_scores_per_range]
    numeric_ratio_labels = [
        f"{label}\n(n={count})"
        for label, count in zip(numeric_ratio_labels, numeric_ratio_counts)
    ]
    numeric_ratio_std = [
        np.std(scores) if scores else 0 for scores in numeric_ratio_f1_scores_per_range
    ]
    plt.errorbar(
        numeric_ratio_labels, numeric_ratio_avg_f1_scores, yerr=numeric_ratio_std, marker='o', capsize=5
    )
    plt.xlabel('Numeric Ratio of Cells')
    plt.ylabel('Average F1 Score')
    plt.title('Average F1 Score by Numeric Ratio of Cells')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ================================


    # By popularity
    popularity_avg_f1_scores = [
        sum(scores) / len(scores) if scores else 0
        for scores in popularity_f1_scores_per_range
    ]
    popularity_labels = [
        f"{rng[0]}+" if rng[1] == float('inf') else f"{rng[0]}-{rng[1]}"
        for rng in graph_popularity
    ]
    popularity_counts = [len(scores) for scores in popularity_f1_scores_per_range]
    # Add counts to labels
    popularity_labels = [
        f"{label}\n(n={count})"
        for label, count in zip(popularity_labels, popularity_counts)
    ]
    popularity_std = [
        np.std(scores) if scores else 0 for scores in popularity_f1_scores_per_range
    ]
    plt.errorbar(
        popularity_labels, popularity_avg_f1_scores, yerr=popularity_std, marker='o', capsize=5
    )
    plt.xlabel('Popularity')
    plt.ylabel('Average F1 Score')
    plt.title('Average F1 Score by Popularity')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # ================================


    # By number of rows
    num_of_rows_avg_f1_scores = [
        sum(scores) / len(scores) if scores else 0
        for scores in num_of_rows_f1_scores_per_range
    ]
    num_of_rows_labels = [
        f"{rng[0]}+" if rng[1] == float('inf') else f"{rng[0]}-{rng[1]}"
        for rng in graph_num_of_rows
    ]
    num_of_rows_counts = [len(scores) for scores in num_of_rows_f1_scores_per_range]
    num_of_rows_labels = [
        f"{label}\n(n={count})"
        for label, count in zip(num_of_rows_labels, num_of_rows_counts)
    ]
    num_of_rows_std = [
        np.std(scores) if scores else 0 for scores in num_of_rows_f1_scores_per_range
    ]
    plt.errorbar(
        num_of_rows_labels, num_of_rows_avg_f1_scores, yerr=num_of_rows_std, marker='o', capsize=5
    )
    plt.xlabel('Number of Rows Range')
    plt.ylabel('Average F1 Score')
    plt.title('Average F1 Score by Number of Rows')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # ================================

    # By number of columns
    num_of_columns_avg_f1_scores = [
        sum(scores) / len(scores) if scores else 0
        for scores in num_of_columns_f1_scores_per_range
    ]
    num_of_columns_labels = [
        f"{rng[0]}+" if rng[1] == float('inf') else f"{rng[0]}-{rng[1]}"
        for rng in graph_num_of_columns
    ]
    num_of_columns_counts = [len(scores) for scores in num_of_columns_f1_scores_per_range]
    num_of_columns_labels = [
        f"{label}\n(n={count})"
        for label, count in zip(num_of_columns_labels, num_of_columns_counts)
    ]
    num_of_columns_std = [
        np.std(scores) if scores else 0 for scores in num_of_columns_f1_scores_per_range
    ]
    plt.errorbar(
        num_of_columns_labels, num_of_columns_avg_f1_scores, yerr=num_of_columns_std, marker='o', capsize=5
    )
    plt.xlabel('Number of Columns Range')
    plt.ylabel('Average F1 Score')
    plt.title('Average F1 Score by Number of Columns')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    plt.hist(cells_f1_scores_per_range[0])
    plt.show()

    pd.DataFrame(raw_csv).to_csv('raw_metrics.csv', index=False)

if __name__ == "__main__":
    get_metrics()
    # This function will be called to get the metrics from the Redis database.
    # It will return the metrics in a structured format.
    # The metrics will include the number of generated tables, the number of cells in each table,
    # the numeric ratio of each table, and the popularity of each table.

import os
from collections import defaultdict

import matplotlib.pyplot as plt
from utils.load_and_preprocess_redset import (
    load_and_preprocess_redset,
)


def get_sub_directories(directory):
    return [x[0] for x in os.walk(directory) if x[0] != directory]


def map_join_count_to_queries(query_stats):
    map_n_joins_to_queries = defaultdict(list)
    for filepath, stats in query_stats.items():
        map_n_joins_to_queries[stats["num_joins"]].append(filepath)

    # Sort to ensure determinism
    map_n_joins_to_queries = {k: sorted(v) for k, v in map_n_joins_to_queries.items()}
    return map_n_joins_to_queries


def map_join_count_to_templates(query_stats):
    res = defaultdict(list)
    for _, stats in query_stats.items():
        if stats["template"] not in res[stats["num_joins"]]:
            res[stats["num_joins"]].append(stats["template"])

    # Sort to ensure determinism
    res = {k: sorted(v) for k, v in res.items()}
    return res


def map_template_to_queries(query_stats):
    res = defaultdict(list)
    for filepath, stats in query_stats.items():
        res[stats["template"]].append(filepath)

    # Sort to ensure determinism
    res = {k: sorted(v) for k, v in res.items()}
    return res


def get_query_timeline(
    redset_filepath,
    cluster_id,
    database_id,
    start_date,
    end_date,
    redset_exclude_tables_never_read: bool,
    limit_redset_rows_read: int,
):
    # load the redset and preprocess it
    con = load_and_preprocess_redset(
        start_date=start_date,
        end_date=end_date,
        redset_path=redset_filepath,
        instance_id=cluster_id,
        database_id=database_id,
        include_copy=False,  # cannot handle copy
        include_analyze=False,  # cannot handle analyze
        include_ctas=False,  # cannot handle ctas
        exclude_tables_never_read=redset_exclude_tables_never_read,
        limit_rows=limit_redset_rows_read,
    )

    res = (
        con.execute(
            """
            select
                *
            from redset_preprocessed
            order by arrival_timestamp, query_hash asc
            """
        )
        .fetchdf()
        .to_dict(orient="records")
    )
    return res


def remove_file(filepath):
    os.system(f"rm -rf {filepath}")
    if os.path.dirname(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)


def wrap(text, width):
    res = []
    while True:
        if len(text) <= width:
            res.append(text)
            break
        idx = text.find(" ", width)
        if idx == -1:
            res.append(text)
            break
        res.append(text[:idx])
        text = text[idx + 1 :]
    return res


def draw_box_plot(
    xs, ys, xlabel, ylabel, save_path=None, log_scale_y=False, title=None
):
    fig, ax = plt.subplots()
    plt.boxplot(ys, vert=True, patch_artist=True)
    plt.xticks(list(range(1, len(ys) + 1)), xs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    if log_scale_y:
        ax.set_yscale("log")
    if title is not None:
        plt.title("\n".join(wrap(title, 60)))
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def draw_bar_plot(
    xs, ys, xlabel, ylabel, save_path=None, log_scale_y=False, title=None
):
    fig, ax = plt.subplots()
    plt.bar(xs, ys)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    if log_scale_y:
        ax.set_yscale("log")
    if title is not None:
        plt.title("\n".join(wrap(title, 60)))
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def parse_user_key(user_key):
    return {
        "user_id": int(user_key.split("#")[0]),
        "instance_id": int(user_key.split("#")[1]),
    }

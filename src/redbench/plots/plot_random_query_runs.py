import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def count_swaps(queries, qig):
    swap_count = 0
    in_group = queries[0].structural_repetition_id == qig

    for q in queries[1:]:
        current_in_group = q.structural_repetition_id == qig
        if current_in_group != in_group:
            swap_count += 1
            in_group = current_in_group

    return swap_count


def plot_query_runs(path, output_path=None):
    def plot_query_run(queries, ax, plot_num, qig):
        timestamps_read = []
        timestamps_write = []
        colors_read = []
        colors_write = []
        sizes_read = []
        sizes_write = []
        alphas = []

        for query in queries:
            if query.structural_repetition_id == qig:
                timestamps_read.append(query.arrival_timestamp)
                if (
                    hasattr(query, "label_future")
                    and query.label_future == "PolicyDecisionAction.INC"
                ):
                    colors_read.append("green")
                elif (
                    hasattr(query, "label_future")
                    and query.label_future == "PolicyDecisionAction.ADD"
                ):
                    colors_read.append("yellow")
                elif (
                    hasattr(query, "label_future")
                    and query.label_future == "PolicyDecisionAction.IRRELEVANT"
                ):
                    colors_read.append("brown")
                else:
                    colors_read.append("blue")
                sizes_read.append(10)
                alphas.append(0.4)
            else:
                timestamps_write.append(query.arrival_timestamp)
                colors_write.append("red")
                sizes_write.append(10)

        jitter_read = np.random.uniform(-0.1, 0.1, size=len(timestamps_read))
        jitter_write = np.random.uniform(-0.1, 0.1, size=len(timestamps_write))

        ax.scatter(
            timestamps_read,
            1 + jitter_read,
            c=colors_read,
            alpha=alphas,
            s=sizes_read,
            label="Read",
        )
        ax.scatter(
            timestamps_write,
            2 + jitter_write,
            c=colors_write,
            alpha=0.4,
            s=sizes_write,
            label="Write",
        )

        ax.set_title(f"Query Run {plot_num + 1}")
        ax.set_xticklabels([])
        ax.set_yticks([1, 2])
        ax.set_yticklabels(["Read", "Write"])
        ax.set_xlabel("Timestamp")
        ax.legend()

    df = pd.read_csv(path, parse_dates=["arrival_timestamp"])
    df["arrival_timestamp"] = pd.to_datetime(df["arrival_timestamp"], format="mixed")

    df["unique_db_instance"] = 0
    grouped_query_traces = list(df.groupby(["unique_db_instance"]))

    seed = 44
    rand_state = np.random.RandomState(seed)
    rand_state.shuffle(grouped_query_traces)

    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(16, 20))
    axes = axes.flatten()

    all_queries = df[df["query_type"].isin(["select", "analyze", "insert"])].sample(
        frac=1, random_state=seed
    )

    ids = []
    encountered_queries = set()
    total_runs = 0

    for query in all_queries.itertuples(index=False):
        qig, read_t, db_instance = (
            query.structural_repetition_id,
            query.join_tables,
            query.unique_db_instance,
        )

        if qig in ids:
            continue

        relevant_queries = df[
            (df["unique_db_instance"] == db_instance)
            & (
                (df["structural_repetition_id"] == qig)
                | (df["write_table"].isin(str(read_t).split(",")))
            )
        ].sort_values(by="arrival_timestamp", ascending=True)

        read_part = df[
            (df["unique_db_instance"] == db_instance)
            & (df["structural_repetition_id"] == qig)
        ]

        # only show query runs that have collisions...
        if len(relevant_queries) == len(read_part):
            continue

        queries = list(relevant_queries.itertuples(index=False))
        encountered_queries.update(
            relevant_queries.copy(deep=True).itertuples(index=False)
        )

        if total_runs < 20:
            plot_query_run(queries, axes[total_runs], total_runs, qig)
            total_runs += 1
            ids.append(qig)
        else:
            break

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        # print(f"Plot saved to {output_path}")
    else:
        plt.show()

def sample_acyclic_join(
    start_t,
    filter_tables,
    no_joins,
    relationships_table,
    table_column_information,
    randstate,
    left_outer_join_ratio,
):
    # randomly sample join
    joins = list()

    table_names = list(table_column_information.keys())
    assert len(table_names) > 0, table_column_information
    assert len(table_names) == len(table_column_information), table_column_information

    join_tables = {start_t}

    for i in range(no_joins):
        possible_joins = find_possible_joins(join_tables, relationships_table)

        filtered_joins = [
            (table_l, column_l, table_r, column_r)
            for table_l, column_l, table_r, column_r in possible_joins
            if table_l in filter_tables and table_r in filter_tables
        ]
        # randomly select one join
        if len(filtered_joins) > 0:
            t, column_l, table_r, column_r = rand_choice(randstate, filtered_joins)
            join_tables.add(table_r)

            left_outer_join = False
            if left_outer_join_ratio > 0 and randstate.rand() < left_outer_join_ratio:
                left_outer_join = True

            joins.append((t, column_l, table_r, column_r, left_outer_join))
        else:
            break
    return joins, join_tables


def find_possible_joins(join_tables, relationships_table):
    possible_joins = list()
    for t in join_tables:
        for column_l, table_r, column_r in relationships_table[t]:
            if table_r in join_tables:
                continue
            possible_joins.append((t, column_l, table_r, column_r))
    return possible_joins


def rand_choice(randstate, elems, no_elements=None, replace=False):
    if no_elements is None:
        idx = randstate.randint(0, len(elems))
        return elems[idx]
    else:
        idxs = randstate.choice(range(len(elems)), no_elements, replace=replace)
        return [elems[i] for i in idxs]

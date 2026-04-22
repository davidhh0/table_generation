

if __name__ == '__main__':
    scores_statistics = ""
    from count import count_retrieval

    scores_statistics += count_retrieval()
    from min import min_retrieval

    scores_statistics += min_retrieval()

    from max import max_retrieval

    scores_statistics += max_retrieval()

    from single_value_retrieval import cell_retrieval

    scores_statistics += cell_retrieval()

    from list_retrieval import list_retrieval

    scores_statistics += list_retrieval()

    with open("scores_statistics.csv", "w") as file:
        file.write(scores_statistics)

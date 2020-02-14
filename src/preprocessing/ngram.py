def make_to_ngram(data, n = 2):
    new_lst = ["<start>", *data, "<end>"]
    return ["_".join(x) for x in zip(*[new_lst[i:] for i in range(n)])]
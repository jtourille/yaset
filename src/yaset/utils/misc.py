def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def flatten(list_of_lists):
    for cur_list in list_of_lists:
        for item in cur_list:
            yield item

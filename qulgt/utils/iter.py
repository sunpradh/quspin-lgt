def cyclic_perm(lst, n=1):
    return lst[n:] + lst[:n]

def zip_nearest(lst, periodic=True, step=1):
    if periodic:
        return zip(lst, cyclic_perm(lst, n=step))
    else:
        return zip(lst[:-step], lst[step:])

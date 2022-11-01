

def gaussian_fn(ts, xs, mean, variance):
    '''
    Score function of Gaussian
    Args:
        - mean: function of ts
        - variance: function of ts
    Return:
        - score: array
    '''
    score = (mean(ts) - xs) / variance(ts)
    return score

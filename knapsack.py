from functools import lru_cache


class knapsack:
    def __init__(self, weights, values):
        self.weights = weights
        self.values = values

    @lru_cache()  # performs the memoization to cache the previously computed results.
    def solve(self, cap, i=0):
        if cap < 0: return -sum(self.values), []
        if i == len(self.weights): return 0, []
        res1 = self.solve(cap, i + 1)
        res2 = self.solve(cap - self.weights[i], i + 1)
        res2 = (res2[0] + self.values[i], [i] + res2[1])
        return res1 if res1[0] >= res2[0] else res2

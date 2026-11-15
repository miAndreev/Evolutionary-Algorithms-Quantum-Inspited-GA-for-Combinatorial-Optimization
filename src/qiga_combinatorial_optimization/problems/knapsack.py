import random


class KnapsackProblem:

    def __init__(
        self,
        n_items=50,
        capacity=500,
        seed=42
    ):
        rng = random.Random(seed)

        self.n_items = n_items
        self.capacity = capacity

        self.weights = [
            rng.randint(5, 50)
            for _ in range(n_items)
        ]

        self.values = [
            rng.randint(10, 100)
            for _ in range(n_items)
        ]

    def evaluate(self, individual):

        total_weight = sum(
            gene * weight
            for gene, weight in zip(
                individual, self.weights
            )
        )

        total_value = sum(
            gene * value
            for gene, value in zip(
                individual, self.values
            )
        )

        if total_weight > self.capacity:
            return (0,)

        return (total_value,)

    def get_solution_details(self, individual):

        total_weight = sum(
            gene * weight
            for gene, weight in zip(
                individual, self.weights
            )
        )

        total_value = sum(
            gene * value
            for gene, value in zip(
                individual, self.values
            )
        )

        return {
            "weight": total_weight,
            "value": total_value,
            "feasible": total_weight <= self.capacity
        }

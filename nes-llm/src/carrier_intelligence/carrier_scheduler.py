from dataclasses import dataclass

import torch


@dataclass
class CarrierAllocation:
    """
    Payload assigned to one carrier.
    """

    layer: int

    module: str

    score: float

    capacity: int

    allocated_bits: int


class CarrierScheduler:
    """
    Hamilton Largest-Remainder
    Payload Scheduler.

    Guarantees

        Σ allocated_bits
        =
        payload_size
    """

    def __init__(

        self,

        gamma=2.5,

        eps=1e-8,

    ):

        self.gamma = gamma

        self.eps = eps
    ##########################################################

    def normalize(
        self,
        scores,
    ):

        scores = scores.float()

        scores = scores.clamp(

            min=0

        )

        ######################################################
        # Non-linear amplification
        ######################################################

        scores = scores.pow(

            self.gamma

        )

        ######################################################
        # Normalize
        ######################################################

        return scores / (

            scores.sum()

            + self.eps

        )

    ##########################################################

    def allocate(

        self,

        carrier_profiles,

        carrier_scores,

        payload_size,

    ):

        weights = self.normalize(
            carrier_scores
        )

        ######################################################
        # Capacity Weighting
        ######################################################

        capacity = torch.tensor(

            [

                p["residual"].numel()

                for p in carrier_profiles

            ],

            dtype=torch.float32,

        )

        capacity = capacity / (

            capacity.mean()

            + self.eps

        )

        weights = weights * capacity

        weights = weights / (

            weights.sum()

            + self.eps

        )

        ######################################################
        # Ideal Allocation
        ######################################################

        ideal = (

            weights

            * payload_size

        )

        ######################################################
        # Integer Allocation
        ######################################################

        integer = torch.floor(
            ideal
        ).long()

        ######################################################
        # Largest Remainders
        ######################################################

        remainder = ideal - integer

        ######################################################
        # Missing Bits
        ######################################################

        remaining = (

            payload_size

            - integer.sum().item()

        )

        ######################################################
        # Hamilton Method
        ######################################################

        if remaining > 0:

            order = torch.argsort(

                remainder,

                descending=True,

            )

            integer[
                order[:remaining]
            ] += 1

        ######################################################
        # Build Allocation Records
        ######################################################

        allocations = []

        for profile, score, bits in zip(

            carrier_profiles,

            carrier_scores,

            integer,

        ):

            allocations.append(

                CarrierAllocation(

                    layer=profile["layer"],

                    module=profile["module"],

                    score=float(score),

                    capacity=profile[
                        "residual"
                    ].numel(),

                    allocated_bits=int(bits),

                )

            )

        allocated = torch.tensor(

            [

                a.allocated_bits

                for a in allocations

            ]

        )

        print()

        print("Carrier Allocation")

        print("-----------------------")

        print(

            "Mean :", allocated.float().mean().item()

        )

        print(

            "Std  :", allocated.float().std().item()

        )

        print(

            "Min  :", allocated.min().item()

        )

        print(

            "Max  :", allocated.max().item()

        )

        print()

        return allocations

    ##########################################################

    def statistics(

        self,

        allocations,

    ):

        allocated = sum(

            a.allocated_bits

            for a in allocations

        )

        capacity = sum(

            a.capacity

            for a in allocations

        )

        used = sum(

            a.allocated_bits > 0

            for a in allocations

        )

        return {

            "allocated_bits":

            allocated,

            "capacity":

            capacity,

            "active_carriers":

            used,

            "total_carriers":

            len(
                allocations
            ),

            "utilization":

            allocated
            /
            capacity,

        }


if __name__ == "__main__":

    profiles = []

    for i in range(10):

        profiles.append(

            {

                "layer": i,

                "module": "q_proj",

                "residual": torch.zeros(
                    1000
                ),

            }

        )

    scores = torch.rand(10)

    scheduler = CarrierScheduler()

    allocations = scheduler.allocate(

        profiles,

        scores,

        10000,

    )

    print()

    print(

        scheduler.statistics(

            allocations

        )
    )
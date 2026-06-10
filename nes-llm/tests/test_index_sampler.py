from src.carrier_selection.index_sampler import (
    IndexSampler
)


positions1 = (
    IndexSampler.sample_positions(
        "mehar123",
        100000,
        20
    )
)

positions2 = (
    IndexSampler.sample_positions(
        "mehar123",
        100000,
        20
    )
)

positions3 = (
    IndexSampler.sample_positions(
        "wrongkey",
        100000,
        20
    )
)

print(
    positions1 == positions2
)

print(
    positions1 == positions3
)

print(
    positions1[:10]
)
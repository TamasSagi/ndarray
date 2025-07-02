from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any, Type

"""
- indicies are not unique
- indicies can be empty
- exact get
- approxx get
"""


@dataclass
class Dimension:
    name: str
    size: int
    start_pos: int | None
    indices: list | None


@dataclass
class DimensionRef:
    by: str  # position or index
    value: int


@dataclass
class DimensionRefSubArray:
    by: str  # position or index
    start: int | None
    end: int | None
    start_position: int | None
    indices: list | None

    def __post_init__(self):
        # checks!
        self.indices = list(range(self.start, self.end + 1))


@dataclass
class MultiDimArray:
    data: list[Any]
    dimensions: list[Dimension]  # should be vec, order matters

    @classmethod
    def replicate(
        cls: Type[MultiDimArray], value: Any, dimensions: list[Dimension]
    ) -> MultiDimArray:
        # reduce(mul, arr, 1) is to calculate product of dimension ranks
        data = [value] * reduce(mul, [dimension.size for dimension in dimensions], 1)

        return cls(data, dimensions)

    @classmethod
    def fill_array(
        cls: Type[MultiDimArray], dimensions: list[Dimension], values: list[Any]
    ) -> MultiDimArray:
        size = reduce(mul, [dimension.size for dimension in dimensions], 1)

        if size != len(values):
            raise ValueError("...")

        return cls(values, dimensions)

    # hashmap, so order doesn't matter
    def get(self, dimension_refs: dict[str, DimensionRef]) -> Any:
        if len(dimension_refs) != len(self.dimensions):
            raise ValueError

        data_idx = 0
        for dim_idx, dim in reversed(list(enumerate(self.dimensions))):
            d_ref = dimension_refs[dim.name]

            # if .index replace d_ref.value to dim.indices.index(d_ref.value)
            # <- FIRT OCCURENCE!
            if dim_idx == 0:
                data_idx += d_ref.value
                continue

            data_idx += (
                reduce(
                    mul,
                    [
                        self.dimensions[prev_dim_idx].size
                        for prev_dim_idx in range(dim_idx - 1, -1, -1)
                    ],
                    1,
                )
                * d_ref.value
            )

        return self.data[data_idx]

    def get_subarr(self, dimension_refs: dict[str, DimensionRefSubArray]) -> list[Any]:
        if len(dimension_refs) != len(self.dimensions):
            raise ValueError

        result = []
        for indices, data in zip(self.iter_indices(), self.data):
            if all(
                [
                    index in dim.indices
                    for index, dim in zip(indices, dimension_refs.values())
                ]
            ):
                result.append((indices, data))

        x = 3

    def lookup(self, dimension_refs: dict[str, DimensionRef]) -> Any:
        if len(dimension_refs) != len(self.dimensions):
            raise ValueError

        data_idx = 0
        for dim_idx, dim in reversed(list(enumerate(self.dimensions))):
            d_ref = dimension_refs[dim.name]

            # if .index replace d_ref.value to dim.indices.index(d_ref.value)
            # <- FIRT OCCURENCE!
            if dim_idx == 0:
                data_idx += round(d_ref.value)
                continue

            data_idx += reduce(
                mul,
                [
                    self.dimensions[prev_dim_idx].size
                    for prev_dim_idx in range(dim_idx - 1, -1, -1)
                ],
                1,
            ) * round(d_ref.value)

        return self.data[data_idx]

    def iter_indices(self):
        dim_sizes = [dim.size for dim in self.dimensions[:-1]]

        for indices in itertools.product(
            *[range(size) for size in reversed(dim_sizes)]
        ):
            for idx in range(self.dimensions[-1].size):
                yield list(reversed(indices)) + [idx]


if __name__ == "__main__":
    dimensions_2d = [
        Dimension("dim1", 2, 0, None),
        Dimension("dim2", 3, 1, None),
    ]

    dimensions_3d = [
        Dimension("dim1", 3, 0, None),
        Dimension("dim2", 2, 1, None),
        Dimension("dim3", 2, 1, None),
    ]

    dimensions_5d = [
        Dimension("dim1", 2, 0, None),
        Dimension("dim2", 3, 1, None),
        Dimension("dim3", 3, 1, None),
        Dimension("dim4", 2, 1, None),
        Dimension("dim5", 3, 1, None),
    ]

    array = MultiDimArray.replicate(0, dimensions_2d)
    array2d = MultiDimArray.fill_array(dimensions_2d, [1, 2, 3, 4, 5, 6])
    array3d = MultiDimArray.fill_array(dimensions_3d, list(range(1, 13, 1)))
    array5d = MultiDimArray.fill_array(dimensions_5d, list(range(1, 109, 1)))

    # print(array5d.get({"dim1": DimensionRef("p", 1), "dim2": DimensionRef("p", 2), "dim4": DimensionRef("p", 1), "dim5": DimensionRef("p", 2), "dim3": DimensionRef("p", 2)}))
    # print(array5d.lookup({"dim1": DimensionRef("p", 1.34), "dim2": DimensionRef("p", 2.44), "dim4": DimensionRef("p", 0.52), "dim5": DimensionRef("p", 1.87), "dim3": DimensionRef("p", 2.0034)}))

    for i in array3d.iter_indices():
        print(i)

    """
        0  1
    0 0 1  2
    1 0 3  4
    2 0 5  6
    0 1 7  8
    1 1 9  10
    2 1 11 12
    """

    print(
        array3d.get_subarr(
            {
                "dim1": DimensionRefSubArray("p", 0, 2, None, None),
                "dim2": DimensionRefSubArray("p", 1, 1, None, None),
                "dim3": DimensionRefSubArray("p", 0, 1, None, None),
            }
        )
    )

    """
    1 2 3
    4 5 6
    """

    print(
        array2d.get_subarr(
            {
                "dim1": DimensionRefSubArray("p", 1, 1, None, None),
                "dim2": DimensionRefSubArray("p", 0, 2, None, None),
            }
        )
    )

    x = 3

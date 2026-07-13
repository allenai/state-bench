from __future__ import annotations

from itertools import product

type Transformation = tuple[int, ...]


def _compose(left: Transformation, right: Transformation) -> Transformation:
    return tuple(right[left[index]] for index in range(len(left)))


def _closure(generators: set[Transformation], identity: Transformation) -> set[Transformation]:
    monoid = {identity, *generators}
    while True:
        expanded = monoid | {_compose(left, right) for left in monoid for right in monoid}
        if len(expanded) == len(monoid):
            return monoid
        monoid = expanded


def test_write_once_transition_monoid_is_r_trivial_for_tiny_domain() -> None:
    states = list(product((None, 0, 1), repeat=2))
    state_index = {state: index for index, state in enumerate(states)}

    def write_once(target: int, value: int) -> Transformation:
        transformed = []
        for state in states:
            result = tuple(
                value if index == target and current is None else current
                for index, current in enumerate(state)
            )
            transformed.append(state_index[result])
        return tuple(transformed)

    identity = tuple(range(len(states)))
    monoid = _closure(
        {write_once(target, value) for target in range(2) for value in range(2)},
        identity,
    )
    principal_right_ideals = {
        element: frozenset(_compose(element, right) for right in monoid) for element in monoid
    }
    assert len(set(principal_right_ideals.values())) == len(monoid)


def test_constant_overwrite_transition_monoid_is_aperiodic_for_tiny_domain() -> None:
    states = list(product(range(2), repeat=2))
    state_index = {state: index for index, state in enumerate(states)}

    def overwrite(target: int, value: int) -> Transformation:
        transformed = []
        for state in states:
            result = tuple(value if index == target else current for index, current in enumerate(state))
            transformed.append(state_index[result])
        return tuple(transformed)

    identity = tuple(range(len(states)))
    monoid = _closure(
        {overwrite(target, value) for target in range(2) for value in range(2)},
        identity,
    )
    for element in monoid:
        power = element
        seen: set[Transformation] = set()
        while power not in seen:
            seen.add(power)
            next_power = _compose(power, element)
            if next_power == power:
                break
            power = next_power
        else:
            raise AssertionError("found a nontrivial cycle in the constant-overwrite monoid")


def test_reference_updates_admit_a_nontrivial_cycle() -> None:
    def update(state: tuple[int, int, int]) -> tuple[int, int, int]:
        values = list(state)
        values[0] = values[1]
        values[1] = values[2]
        values[2] = values[0]
        return values[0], values[1], values[2]

    first = update((0, 0, 1))
    second = update(first)
    third = update(second)
    assert first != second
    assert third == first

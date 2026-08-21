"""Gesture vocabulary for Ninapro DB5.

The ``restimulus`` codes in a DB5 ``.mat`` file are only meaningful alongside
the exercise the file belongs to, and DB5 numbers its movements from 1 within
each exercise. Rest is 0 in all three.
"""

from __future__ import annotations

REST = 0
REST_NAME = "rest"

# Exercise A: isolated finger movements.
EXERCISE_A: dict[int, str] = {
    1: "index flexion",
    2: "index extension",
    3: "middle flexion",
    4: "middle extension",
    5: "ring flexion",
    6: "ring extension",
    7: "little finger flexion",
    8: "little finger extension",
    9: "thumb adduction",
    10: "thumb abduction",
    11: "thumb flexion",
    12: "thumb extension",
}

# Exercise B: basic hand configurations and wrist movements. This is the
# default set. It carries the six wrist degrees of freedom plus fist, point
# and finger spread, which between them cover what a wrist-and-grip
# prosthesis is actually asked to do.
EXERCISE_B: dict[int, str] = {
    1: "thumb up",
    2: "index and middle extended, others flexed",
    3: "ring and little flexed, thumb over fingers",
    4: "thumb opposing little finger base",
    5: "all fingers abducted",
    6: "fingers flexed together in fist",
    7: "pointing index",
    8: "adduction of extended fingers",
    9: "wrist supination (middle finger axis)",
    10: "wrist pronation (middle finger axis)",
    11: "wrist supination (little finger axis)",
    12: "wrist pronation (little finger axis)",
    13: "wrist flexion",
    14: "wrist extension",
    15: "wrist radial deviation",
    16: "wrist ulnar deviation",
    17: "wrist extension with closed hand",
}

# Exercise C: grasps and functional movements.
EXERCISE_C: dict[int, str] = {
    1: "large diameter grasp",
    2: "small diameter grasp",
    3: "fixed hook grasp",
    4: "index finger extension grasp",
    5: "medium wrap",
    6: "ring grasp",
    7: "prismatic four finger grasp",
    8: "stick grasp",
    9: "writing tripod grasp",
    10: "power sphere grasp",
    11: "three finger sphere grasp",
    12: "precision sphere grasp",
    13: "tripod grasp",
    14: "prismatic pinch grasp",
    15: "tip pinch grasp",
    16: "quadpod grasp",
    17: "lateral grasp",
    18: "parallel extension grasp",
    19: "extension type grasp",
    20: "power disk grasp",
    21: "open a bottle with a tripod grasp",
    22: "turn a screw",
    23: "cut something",
}

EXERCISES: dict[str, dict[int, str]] = {
    "E1": EXERCISE_A,
    "E2": EXERCISE_B,
    "E3": EXERCISE_C,
}

# A functionally distinct subset of exercise B. Every pair in here commands a
# different actuator on a six-axis arm, so a confusion between two of them is
# a visible error rather than a near miss. Used for the reduced-vocabulary
# result reported alongside the full 18-class one.
PROSTHETIC_SUBSET_E2: list[int] = [1, 5, 6, 7, 9, 10, 13, 14]


def gesture_names(exercise: str, codes: list[int]) -> list[str]:
    """Human-readable names for restimulus codes within an exercise."""
    if exercise not in EXERCISES:
        raise ValueError(f"unknown exercise {exercise!r}; expected one of {sorted(EXERCISES)}")
    table = EXERCISES[exercise]
    names = []
    for code in codes:
        if code == REST:
            names.append(REST_NAME)
        elif code in table:
            names.append(table[code])
        else:
            raise ValueError(f"restimulus code {code} is not part of exercise {exercise}")
    return names


def short_names(exercise: str, codes: list[int]) -> list[str]:
    """Compact labels for plot axes, where the full descriptions do not fit."""
    out = []
    for code in codes:
        if code == REST:
            out.append("rest")
        else:
            out.append(f"{exercise[-1]}{code:02d}")
    return out

"""Action Atlas concept tables, remapped onto LeRobot LIBERO task ids.

Atlas scores concepts with Cohen's d over SAE features. We keep that protocol
and those concept-to-task sets, but replace the SAE with our transcoders.

Atlas task ids are alphabetical by prompt. LeRobot / this repo use the
LIBERO native order in ``run/pi0.5/official_tasks.yaml``. Concept lookup
always remaps through the prompt string.

Source: https://github.com/CWRU-AISM/action-atlas
(experiments/concept_identification.py, Apache-2.0)
"""

from __future__ import annotations

from typing import Any

# Atlas alphabetical prompts (their task_id space).
ATLAS_TASK_PROMPTS: dict[str, dict[int, str]] = {
    "libero_spatial": {
        0: "pick up the black bowl between the plate and the ramekin and place it on the plate",
        1: "pick up the black bowl from table center and place it on the plate",
        2: "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
        3: "pick up the black bowl next to the cookie box and place it on the plate",
        4: "pick up the black bowl next to the plate and place it on the plate",
        5: "pick up the black bowl next to the ramekin and place it on the plate",
        6: "pick up the black bowl on the cookie box and place it on the plate",
        7: "pick up the black bowl on the ramekin and place it on the plate",
        8: "pick up the black bowl on the stove and place it on the plate",
        9: "pick up the black bowl on the wooden cabinet and place it on the plate",
    },
    "libero_object": {
        0: "pick up the alphabet soup and place it in the basket",
        1: "pick up the bbq sauce and place it in the basket",
        2: "pick up the butter and place it in the basket",
        3: "pick up the chocolate pudding and place it in the basket",
        4: "pick up the cream cheese and place it in the basket",
        5: "pick up the ketchup and place it in the basket",
        6: "pick up the milk and place it in the basket",
        7: "pick up the orange juice and place it in the basket",
        8: "pick up the salad dressing and place it in the basket",
        9: "pick up the tomato sauce and place it in the basket",
    },
    "libero_goal": {
        0: "open the middle drawer of the cabinet",
        1: "open the top drawer and put the bowl inside",
        2: "push the plate to the front of the stove",
        3: "put the bowl on the plate",
        4: "put the bowl on the stove",
        5: "put the bowl on top of the cabinet",
        6: "put the cream cheese in the bowl",
        7: "put the wine bottle on the rack",
        8: "put the wine bottle on top of the cabinet",
        9: "turn on the stove",
    },
    "libero_10": {
        0: "pick up the book and place it in the back compartment of the caddy",
        1: "put both moka pots on the stove",
        2: "put both the alphabet soup and the cream cheese box in the basket",
        3: "put both the alphabet soup and the tomato sauce in the basket",
        4: "put both the cream cheese box and the butter in the basket",
        5: "put the black bowl in the bottom drawer of the cabinet and close it",
        6: "put the white mug on the left plate and put the yellow and white mug on the right plate",
        7: "put the white mug on the plate and put the chocolate pudding to the right of the plate",
        8: "put the yellow and white mug in the microwave and close it",
        9: "turn on the stove and put the moka pot on it",
    },
}

# LeRobot / this repo (LIBERO native order).
LEROBOT_TASK_PROMPTS: dict[str, dict[int, str]] = {
    "libero_spatial": {
        0: "pick up the black bowl between the plate and the ramekin and place it on the plate",
        1: "pick up the black bowl next to the ramekin and place it on the plate",
        2: "pick up the black bowl from table center and place it on the plate",
        3: "pick up the black bowl on the cookie box and place it on the plate",
        4: "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
        5: "pick up the black bowl on the ramekin and place it on the plate",
        6: "pick up the black bowl next to the cookie box and place it on the plate",
        7: "pick up the black bowl on the stove and place it on the plate",
        8: "pick up the black bowl next to the plate and place it on the plate",
        9: "pick up the black bowl on the wooden cabinet and place it on the plate",
    },
    "libero_object": {
        0: "pick up the alphabet soup and place it in the basket",
        1: "pick up the cream cheese and place it in the basket",
        2: "pick up the salad dressing and place it in the basket",
        3: "pick up the bbq sauce and place it in the basket",
        4: "pick up the ketchup and place it in the basket",
        5: "pick up the tomato sauce and place it in the basket",
        6: "pick up the butter and place it in the basket",
        7: "pick up the milk and place it in the basket",
        8: "pick up the chocolate pudding and place it in the basket",
        9: "pick up the orange juice and place it in the basket",
    },
    "libero_goal": {
        0: "open the middle drawer of the cabinet",
        1: "put the bowl on the stove",
        2: "put the wine bottle on top of the cabinet",
        3: "open the top drawer and put the bowl inside",
        4: "put the bowl on top of the cabinet",
        5: "push the plate to the front of the stove",
        6: "put the cream cheese in the bowl",
        7: "turn on the stove",
        8: "put the bowl on the plate",
        9: "put the wine bottle on the rack",
    },
    "libero_10": {
        0: "put both the alphabet soup and the tomato sauce in the basket",
        1: "put both the cream cheese box and the butter in the basket",
        2: "turn on the stove and put the moka pot on it",
        3: "put the black bowl in the bottom drawer of the cabinet and close it",
        4: "put the white mug on the left plate and put the yellow and white mug on the right plate",
        5: "pick up the book and place it in the back compartment of the caddy",
        6: "put the yellow and white mug in the microwave and close it",
        7: "put both moka pots on the stove",
        8: "put the yellow and white mug on the plate and put the chocolate pudding to the right of the plate",
        9: "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    },
}

# Atlas concept tables, still in Atlas alphabetical task-id space.
_ATLAS_CONCEPTS: dict[str, dict[str, dict[str, dict[str, list[int]]]]] = {
    "libero_spatial": {
        "motion": {
            "pick": {"tasks": list(range(10))},
            "place": {"tasks": list(range(10))},
        },
        "object": {
            "bowl": {"tasks": list(range(10))},
            "plate": {"tasks": list(range(10))},
            "ramekin": {"tasks": [0, 5, 7]},
            "cookie_box": {"tasks": [3, 6]},
            "stove": {"tasks": [8]},
            "cabinet": {"tasks": [2, 9]},
            "drawer": {"tasks": [2]},
        },
        "spatial": {
            "between": {"tasks": [0]},
            "center": {"tasks": [1]},
            "in_drawer": {"tasks": [2]},
            "next_to": {"tasks": [3, 4, 5]},
            "on": {"tasks": [6, 7, 8, 9]},
        },
    },
    "libero_object": {
        "motion": {
            "pick": {"tasks": list(range(10))},
            "place": {"tasks": list(range(10))},
        },
        "object": {
            "alphabet_soup": {"tasks": [0]},
            "bbq_sauce": {"tasks": [1]},
            "butter": {"tasks": [2]},
            "chocolate_pudding": {"tasks": [3]},
            "cream_cheese": {"tasks": [4]},
            "ketchup": {"tasks": [5]},
            "milk": {"tasks": [6]},
            "orange_juice": {"tasks": [7]},
            "salad_dressing": {"tasks": [8]},
            "tomato_sauce": {"tasks": [9]},
            "basket": {"tasks": list(range(10))},
        },
        "spatial": {
            "in": {"tasks": list(range(10))},
        },
    },
    "libero_goal": {
        "motion": {
            "put": {"tasks": [1, 3, 4, 5, 6, 7, 8]},
            "open": {"tasks": [0, 1]},
            "push": {"tasks": [2]},
            "interact": {"tasks": [9]},
        },
        "object": {
            "bowl": {"tasks": [1, 3, 4, 5, 6]},
            "plate": {"tasks": [2, 3]},
            "stove": {"tasks": [4, 9]},
            "cabinet": {"tasks": [0, 5, 8]},
            "drawer": {"tasks": [0, 1]},
            "wine_bottle": {"tasks": [7, 8]},
            "cream_cheese": {"tasks": [6]},
            "rack": {"tasks": [7]},
        },
        "spatial": {
            "on": {"tasks": [3, 4, 7, 8]},
            "in": {"tasks": [1, 6]},
            "top": {"tasks": [1, 5, 8]},
            "front": {"tasks": [2]},
            "middle": {"tasks": [0]},
        },
    },
    "libero_10": {
        "motion": {
            "pick": {"tasks": [0]},
            "put": {"tasks": [1, 2, 3, 4, 5, 6, 7, 8]},
            "close": {"tasks": [5, 8]},
            "turn_on": {"tasks": [9]},
        },
        "object": {
            "book": {"tasks": [0]},
            "caddy": {"tasks": [0]},
            "moka_pot": {"tasks": [1, 9]},
            "stove": {"tasks": [1, 9]},
            "alphabet_soup": {"tasks": [2, 3]},
            "cream_cheese": {"tasks": [2, 4]},
            "tomato_sauce": {"tasks": [3]},
            "butter": {"tasks": [4]},
            "basket": {"tasks": [2, 3, 4]},
            "bowl": {"tasks": [5]},
            "drawer": {"tasks": [5]},
            "cabinet": {"tasks": [5]},
            "mug": {"tasks": [6, 7, 8]},
            "plate": {"tasks": [6, 7]},
            "pudding": {"tasks": [7]},
            "microwave": {"tasks": [8]},
        },
        "spatial": {
            "on": {"tasks": [1, 6, 7, 9]},
            "in": {"tasks": [0, 2, 3, 4, 5, 8]},
            "left": {"tasks": [6]},
            "right": {"tasks": [6, 7]},
            "bottom": {"tasks": [5]},
        },
    },
}

_SUITE_ALIASES = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "10": "libero_10",
    "libero_long": "libero_10",
}


def normalize_suite(suite: str) -> str:
    key = suite.strip().lower()
    return _SUITE_ALIASES.get(key, key)


def normalize_prompt(text: str) -> str:
    return " ".join(text.strip().lower().split())


def atlas_to_lerobot_task_id(suite: str, atlas_task_id: int) -> int:
    suite = normalize_suite(suite)
    prompt = ATLAS_TASK_PROMPTS[suite][atlas_task_id]
    wanted = normalize_prompt(prompt)
    for task_id, candidate in LEROBOT_TASK_PROMPTS[suite].items():
        if normalize_prompt(candidate) == wanted:
            return task_id
    raise KeyError(f"No LeRobot task id for {suite} atlas task {atlas_task_id}: {prompt}")


def lerobot_to_atlas_task_id(suite: str, lerobot_task_id: int) -> int:
    suite = normalize_suite(suite)
    prompt = LEROBOT_TASK_PROMPTS[suite][lerobot_task_id]
    wanted = normalize_prompt(prompt)
    for task_id, candidate in ATLAS_TASK_PROMPTS[suite].items():
        if normalize_prompt(candidate) == wanted:
            return task_id
    raise KeyError(f"No Atlas task id for {suite} lerobot task {lerobot_task_id}: {prompt}")


def task_id_from_prompt(suite: str, prompt: str, *, space: str = "lerobot") -> int | None:
    suite = normalize_suite(suite)
    wanted = normalize_prompt(prompt)
    table = LEROBOT_TASK_PROMPTS if space == "lerobot" else ATLAS_TASK_PROMPTS
    if suite not in table:
        return None
    for task_id, candidate in table[suite].items():
        if normalize_prompt(candidate) == wanted:
            return task_id
    return None


def get_concept_task_mapping(suite: str, *, space: str = "lerobot") -> dict[str, dict[str, dict[str, Any]]]:
    """Return ``{type: {name: {tasks: [ids]}}}`` in the requested task-id space."""
    suite = normalize_suite(suite)
    raw = _ATLAS_CONCEPTS.get(suite)
    if raw is None:
        return {}
    if space == "atlas":
        return {
            concept_type: {
                name: {"tasks": list(info["tasks"])}
                for name, info in concepts.items()
            }
            for concept_type, concepts in raw.items()
        }

    remapped: dict[str, dict[str, dict[str, Any]]] = {}
    for concept_type, concepts in raw.items():
        remapped[concept_type] = {}
        for name, info in concepts.items():
            remapped[concept_type][name] = {
                "tasks": [atlas_to_lerobot_task_id(suite, task_id) for task_id in info["tasks"]]
            }
    return remapped

"""How each agent looks in every frontend: its piece, colors per look and copy.

This is the single source for agent identity. The web UI reads it through
``/api/agents``; no theme defines agent colors of its own. Entries are keyed by
registry key. Entries marked ``planned`` have no registry spec yet (Phase 6) and
always show as not available; when their agents are registered under the same
key they pick up this identity automatically.

Pieces: ``I J L O S T Z`` are the tetrominoes, ``mono`` is a single block and
``P`` is the P pentomino (XXX over XX.).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

#: Look ids, in switcher order.
LOOKS: tuple[str, ...] = ("tetromino", "terminal", "broadcast", "poster")

#: Lineup groups. ``main`` is the Home lineup and the first opponent group,
#: ``variant`` is listed under "Variants" on the opponent screen only, and
#: ``leaderboard`` entries are evaluated but never offered as opponents.
GROUPS: tuple[str, ...] = ("main", "variant", "leaderboard")

#: Cells of each piece shape as (row, col), for icons.
PIECE_CELLS: dict[str, list[tuple[int, int]]] = {
    "I": [(0, 0), (0, 1), (0, 2), (0, 3)],
    "J": [(0, 0), (1, 0), (1, 1), (1, 2)],
    "L": [(0, 2), (1, 0), (1, 1), (1, 2)],
    "O": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "S": [(0, 1), (0, 2), (1, 0), (1, 1)],
    "T": [(0, 0), (0, 1), (0, 2), (1, 1)],
    "Z": [(0, 0), (0, 1), (1, 1), (1, 2)],
    "mono": [(0, 0)],
    "P": [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)],
}

INK = "#1B1F3B"
WHITE = "#FFFFFF"
POSTER_DARK = "#14161F"


@dataclass(frozen=True)
class Identity:
    key: str
    name: str  # display name ("Dellacherie", not the registry's "Dellacherie (baseline)")
    family: str  # one or two words: "Value-based RL"
    blurb: str  # one-line description for the opponent list
    piece: str | None  # key of PIECE_CELLS, None for the human
    letter: str  # Terminal look roster id
    #: Tetromino look: fill color and the text color used on it.
    a_fill: str
    a_text: str
    #: Broadcast look: piece and accent color on the deep blue panels.
    c_fill: str
    #: Poster look: column field color and the text and board color on it.
    d_field: str
    d_text: str
    group: str = "main"
    planned: bool = False  # no registry spec yet

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cells"] = [list(c) for c in PIECE_CELLS[self.piece]] if self.piece else []
        d["looks"] = {
            "tetromino": {"fill": self.a_fill, "text": self.a_text},
            "terminal": {"letter": self.letter},
            "broadcast": {"fill": self.c_fill},
            "poster": {"field": self.d_field, "text": self.d_text},
        }
        for k in ("a_fill", "a_text", "c_fill", "d_field", "d_text"):
            del d[k]
        return d


def _variant(parent: Identity, **changes: Any) -> Identity:
    """Same piece and colors as ``parent`` (variants are told apart by name)."""
    return Identity(**{**asdict(parent), **changes})


_DQN = Identity(
    "dqn",
    "DQN",
    "Value-based RL",
    "A deep Q-network that scores every spot the piece could land and takes the best.",
    "T",
    "T",
    "#7A4FD0",
    WHITE,
    "#B07CFF",
    "#3D2BBF",
    WHITE,
)
_CEM = Identity(
    "cem",
    "CEM",
    "Cross-entropy method",
    "Weighs board features such as holes and height, with weights tuned by the "
    "cross-entropy method.",
    "Z",
    "Z",
    "#D53E2D",  # white is 4.61:1 here (4.47:1 on the original #D8402F)
    WHITE,
    "#FF5A5A",
    "#F2542D",
    "#1A0F0A",
)
_GA = Identity(
    "ga",
    "GA heuristic",
    "Genetic algorithm",
    "The same features, with weights evolved by a genetic algorithm.",
    "S",
    "S",
    "#2E9E55",
    INK,  # white is 3.4:1 on this fill
    "#4BE08A",
    "#1F7A3A",
    WHITE,
)
_PPO = Identity(
    "ppo",
    "PPO",
    "Policy-gradient RL",
    "A policy network that learned which placement to choose by trial and error.",
    "L",
    "L",
    "#E07A12",
    INK,
    "#FF9B3D",
    "#0B7566",
    WHITE,
)
_BC = Identity(
    "bc",
    "Behavior cloning",
    "Imitation learning",
    "Learned by copying recorded games of the GA heuristic.",
    "J",
    "J",
    "#2F5FD0",
    WHITE,
    "#5B8CFF",
    "#1E4FD8",
    WHITE,
)

IDENTITIES: dict[str, Identity] = {
    i.key: i
    for i in (
        _DQN,
        _CEM,
        _GA,
        _PPO,
        _BC,
        Identity(
            "bc_you",
            "BC (you)",
            "Imitation learning",
            "Behavior cloning fine-tuned on your recorded games.",
            "J",
            "j",
            "#729410",
            INK,
            "#C6F24E",
            "#A4CC1C",
            POSTER_DARK,
        ),
        Identity(
            "neat",
            "NEAT",
            "Neuroevolution",
            "Its network shape and weights were evolved together, with no gradients.",
            "O",
            "O",
            "#E6B800",
            INK,
            "#FFD23F",
            "#F2A900",
            POSTER_DARK,
        ),
        Identity(
            "dt",
            "Decision Transformer",
            "Offline RL",
            "A transformer that plays toward a score you set. Higher targets play harder.",
            "I",
            "I",
            "#1597A8",
            INK,  # white is 3.5:1 on this fill
            "#22D3E6",
            "#087A8F",
            WHITE,
            planned=True,
        ),
        Identity(
            "dellacherie",
            "Dellacherie",
            "Hand-tuned baseline",
            "Hand-picked weights from classic Tetris research. Everything else is measured "
            "against it.",
            "mono",
            ".",
            "#6B7080",
            WHITE,
            "#AEB4C8",
            "#6B7080",
            WHITE,
        ),
        Identity(
            "llm",
            "LLM agent",
            "Optional novelty",
            "Reads the board as text and picks a move. Slow, and just for fun.",
            "P",
            "P",
            "#C2408A",
            WHITE,
            "#FF7AC6",
            "#C2307E",
            WHITE,
            planned=True,
        ),
        _variant(
            _PPO,
            key="ppo_raw",
            name="PPO (raw actions)",
            blurb="The same method choosing among all 80 moves from the board alone. "
            "Weak, kept for comparison.",
            letter="l",
            group="variant",
        ),
        _variant(
            _BC,
            key="bc_pure",
            name="BC (pure)",
            blurb="Copies logged GA games only.",
            group="leaderboard",
        ),
        _variant(
            _BC,
            key="bc_dagger",
            name="BC + DAgger",
            blurb="Copies GA games, then two rounds where the GA labels the agent's own states.",
            group="leaderboard",
        ),
    )
}

HUMAN = Identity(
    "human",
    "You",
    "Human",
    "You.",
    None,
    "@",
    INK,
    WHITE,
    "#FF5A3C",
    "#FFE14D",
    POSTER_DARK,
    group="human",
)


def identity(key: str) -> Identity:
    """Identity for a registry key; unknown agents get a neutral grey block."""
    if key == HUMAN.key:
        return HUMAN
    found = IDENTITIES.get(key)
    if found is not None:
        return found
    return Identity(
        key, key, "Agent", "", "mono", "?", "#6B7080", WHITE, "#AEB4C8", "#6B7080", WHITE
    )


def lineup(group: str | None = None) -> list[Identity]:
    """Identities in display order, optionally one group only."""
    return [i for i in IDENTITIES.values() if group is None or i.group == group]

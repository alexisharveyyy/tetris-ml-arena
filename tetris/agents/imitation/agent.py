"""Behavior-cloning agents: the PPO afterstate policy, trained by imitation."""

from __future__ import annotations

from tetris.agents.ppo.agent import PPOAgent


class BCAgent(PPOAgent):
    """Plays the most probable legal placement of a behavior-cloned policy.

    The network is the PPO afterstate policy (each action's logit scores its
    afterstate), so checkpoints load the same way and can warm-start PPO.
    Difficulty mistakes pick among the next five placements by probability.
    """

    key = "bc"
    name = "Imitation (BC)"
    description = "Behavior cloning of the GA heuristic."
    kind = "ppo"

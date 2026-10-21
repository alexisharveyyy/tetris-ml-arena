#!/bin/bash
# Waits for the PPO option B queue (any exit: success or failure), then runs the Phase 5 queue.
PPO_PID=26224
cd /Users/alexisharvey/tetris-ml-arena-phase5
echo "$(date '+%Y-%m-%d %H:%M:%S')  waiting for PPO queue pid $PPO_PID" >> logs/phase4_queue.log
while kill -0 $PPO_PID 2>/dev/null; do sleep 60; done
echo "$(date '+%Y-%m-%d %H:%M:%S')  PPO queue exited; starting phase 5 queue at $(git rev-parse --short HEAD)" >> logs/phase4_queue.log
export PYTHONPATH=/Users/alexisharvey/tetris-ml-arena-phase5
exec caffeinate -i /Users/alexisharvey/tetris-ml-arena/.venv/bin/python scripts/run_phase5.py

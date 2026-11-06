// Delayed auto shift, auto repeat and soft drop repeat (tetris/match/das.py).
import type { Move } from "../engine/game";

const INSTANT_SHIFTS = 10;

export interface RepeatSettings {
  das: number; // seconds
  arr: number; // seconds, 0 is instant
  softDropRate: number; // rows per second
}

export class InputRepeater {
  private held: Move[] = [];
  private dasTimer = 0;
  private arrTimer = 0;
  private softHeld = false;
  private softTimer = 0;

  constructor(public settings: RepeatSettings) {}

  reset(): void {
    this.held = [];
    this.dasTimer = 0;
    this.arrTimer = 0;
    this.softHeld = false;
    this.softTimer = 0;
  }

  press(move: Move): Move[] {
    if (move === "left" || move === "right") {
      this.held = this.held.filter((m) => m !== move);
      this.held.push(move);
      this.dasTimer = 0;
      this.arrTimer = 0;
      return [move];
    }
    if (move === "soft") {
      this.softHeld = true;
      this.softTimer = 0;
      return ["soft"];
    }
    return [move];
  }

  release(move: Move): void {
    if (this.held.includes(move)) {
      const wasActive = this.held[this.held.length - 1] === move;
      this.held = this.held.filter((m) => m !== move);
      if (wasActive) {
        this.dasTimer = 0;
        this.arrTimer = 0;
      }
    } else if (move === "soft") this.softHeld = false;
  }

  update(dt: number): Move[] {
    const out: Move[] = [];
    if (this.held.length) {
      const dir = this.held[this.held.length - 1];
      const before = this.dasTimer;
      this.dasTimer += dt;
      const das = this.settings.das;
      if (this.dasTimer >= das) {
        if (this.settings.arr <= 0) {
          for (let i = 0; i < INSTANT_SHIFTS; i++) out.push(dir);
        } else {
          if (before < das) {
            out.push(dir);
            this.arrTimer = this.dasTimer - das;
          } else this.arrTimer += dt;
          while (this.arrTimer >= this.settings.arr) {
            this.arrTimer -= this.settings.arr;
            out.push(dir);
          }
        }
      }
    }
    if (this.softHeld && this.settings.softDropRate > 0) {
      this.softTimer += dt;
      const interval = 1 / this.settings.softDropRate;
      while (this.softTimer >= interval) {
        this.softTimer -= interval;
        out.push("soft");
      }
    }
    return out;
  }
}

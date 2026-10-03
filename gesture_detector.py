"""Deliberate whole-hand gestures that complement precise fingertip controls."""

from collections import deque
import math
import time


TIP_IDS = (4, 8, 12, 16, 20)


class GestureDetector:
    def __init__(self, cooldown: float, long_pinch_seconds: float = 0.55) -> None:
        self._cooldown = cooldown
        self._long_pinch_seconds = long_pinch_seconds
        self._wrist_history: dict[str, deque[tuple[float, float]]] = {}
        self._fist_held: set[str] = set()
        # hand -> (effect name, time it began, whether the long hold fired)
        self._pinch_state: dict[str, tuple[str, float, bool]] = {}
        self._last_event: dict[str, float] = {}

    def update(self, hand, now: float | None = None) -> list[str]:
        now = time.monotonic() if now is None else now
        hand_id = hand.handedness
        events: list[str] = []
        wrist_x, wrist_y = hand.wrist
        tip_distance = sum(math.dist(hand.landmarks[tip], (wrist_x, wrist_y)) for tip in TIP_IDS) / len(TIP_IDS)
        is_fist = tip_distance < 0.20
        if is_fist:
            self._pinch_state.pop(hand_id, None)
            if hand_id not in self._fist_held and self._available(hand_id, now):
                events.append("fist")
                self._last_event[hand_id] = now
            self._fist_held.add(hand_id)
            return events
        self._fist_held.discard(hand_id)

        # Left-hand effect controls: a quick pinch-and-release raises a
        # setting; holding the same pinch lowers it once.
        if hand_id == "left":
            pinch_actions = (
                (8, "bass"),    # thumb + index
                (12, "highs"),  # thumb + middle
                (16, "reverb"), # thumb + ring
                (20, "echo"),   # thumb + pinky
            )
            active_pinch = None
            for landmark_id, action in pinch_actions:
                if math.dist(hand.landmarks[4], hand.landmarks[landmark_id]) < 0.075:
                    active_pinch = action
                    break
            if active_pinch:
                state = self._pinch_state.get(hand_id)
                if state is None or state[0] != active_pinch:
                    self._pinch_state[hand_id] = (active_pinch, now, False)
                elif not state[2] and now - state[1] >= self._long_pinch_seconds and self._available(hand_id, now):
                    self._last_event[hand_id] = now
                    self._pinch_state[hand_id] = (active_pinch, state[1], True)
                    events.append(f"{active_pinch}_down")
                return events
            # A released, short pinch is the positive/increase action.
            state = self._pinch_state.pop(hand_id, None)
            if state is not None and not state[2] and self._available(hand_id, now):
                self._last_event[hand_id] = now
                events.append(f"{state[0]}_up")
        history = self._wrist_history.setdefault(hand_id, deque(maxlen=10))
        history.append((now, wrist_x))
        if len(history) >= 5 and self._available(hand_id, now):
            start_time, start_x = history[0]
            movement = wrist_x - start_x
            if now - start_time <= 0.55 and abs(movement) >= 0.20:
                events.append("swipe_right" if movement > 0 else "swipe_left")
                self._last_event[hand_id] = now
                history.clear()
        return events

    def _available(self, hand_id: str, now: float) -> bool:
        return now - self._last_event.get(hand_id, 0) >= self._cooldown

from collections import deque
from pathlib import Path

import cv2

from .classifier import FINGERS
from .config import CONFIG
from .dj_mixer import DjMixer
from .dj_controls import COMMAND_LABELS, DjControls, TapCommandRouter
from .hand_tracker import HandTracker
from .gesture_detector import GestureDetector
from .tap_detector import LandmarkTapDetector


SOUND_NAMES = {
    "left:thumb": "KICK", "left:index": "SNARE", "left:middle": "HI-HAT", "left:ring": "CLAP", "left:pinky": "TOM",
    "right:thumb": "C", "right:index": "D", "right:middle": "E", "right:ring": "G", "right:pinky": "HIGH C",
}
COLORS = {"thumb": (0, 194, 255), "index": (255, 130, 0), "middle": (80, 220, 80), "ring": (190, 80, 255), "pinky": (80, 80, 255)}


def _draw_hud(frame, recent: deque[str], dj: DjMixer, controls: DjControls) -> None:
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width, 132), (18, 18, 18), -1)
    cv2.putText(frame, "AIRBEAT  |  TWO-HAND MODE", (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2)
    cv2.putText(frame, "LEFT  EQ + FX:  Bass  Mid/Vocal  Highs  Reverb  Echo", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 190, 190), 1)
    cv2.putText(frame, "RIGHT  DECK:  Pause/Play  Volume  Mix  Transition  Cue", (18, 79), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (190, 190, 190), 1)
    if dj.has_next_track:
        deck_text = f"Now: {dj.current_track_name}  |  Next: {dj.next_track_name}  |  {'PLAYING' if dj.playing else 'PAUSED'}"
    elif dj.playing:
        deck_text = f"Now: {dj.current_track_name}  |  add song_2.mp3 for transitions"
    else:
        deck_text = "add assets/song.mp3 for background playback"
    help_text = f"{deck_text} | q / Esc: quit"
    cv2.putText(frame, controls.effects_label(), (18, 101), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 205, 255), 1)
    cv2.putText(frame, help_text, (18, 123), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (145, 145, 145), 1)
    cv2.putText(frame, "beats: " + "  ".join(recent), (18, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)


def run() -> None:
    camera = cv2.VideoCapture(CONFIG.camera_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.frame_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.frame_height)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam. Try camera_index=1 in config.py.")

    tracker, detector = HandTracker(CONFIG), LandmarkTapDetector(CONFIG)
    gestures = GestureDetector(CONFIG.gesture_cooldown_seconds, CONFIG.long_pinch_seconds)
    # A trained model replaces the rules automatically, but the prototype stays
    # usable before training data has been recorded.
    if Path(CONFIG.cnn_model_path).is_file():
        try:
            from .cnn_tap_detector import CnnTapDetector
            detector = CnnTapDetector(CONFIG)
        except (ImportError, RuntimeError, OSError):
            pass
    dj = DjMixer()
    controls, command_router = DjControls(), TapCommandRouter()
    try:
        dj.load_playlist(CONFIG.playlist_paths)
    except (OSError, RuntimeError, ValueError):
        pass
    recent: deque[str] = deque(maxlen=8)
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            hands = tracker.detect(frame)
            if not hands:
                detector.reset()
                cv2.putText(frame, "Show one or both hands to begin", (25, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (70, 220, 255), 2)
            else:
                height, width = frame.shape[:2]
                for hand in hands:
                    gesture_events = gestures.update(hand)
                    if gesture_events:
                        for gesture in gesture_events:
                            if gesture == "fist":
                                dj.toggle_playing()
                                recent.appendleft("GESTURE: PLAY" if dj.playing else "GESTURE: PAUSE")
                            elif gesture == "swipe_right":
                                dj.transition_to_next()
                                recent.appendleft("GESTURE: TRANSITION")
                            elif gesture == "bass_up":
                                recent.appendleft(controls.execute("left:thumb", 1, dj))
                            elif gesture == "bass_down":
                                recent.appendleft(controls.execute("left:thumb", 2, dj))
                            elif gesture == "reverb_up":
                                recent.appendleft(controls.execute("left:ring", 1, dj))
                            elif gesture == "reverb_down":
                                recent.appendleft(controls.execute("left:ring", 2, dj))
                            elif gesture == "highs_up":
                                recent.appendleft(controls.execute("left:middle", 1, dj))
                            elif gesture == "highs_down":
                                recent.appendleft(controls.execute("left:middle", 2, dj))
                            elif gesture == "echo_up":
                                recent.appendleft(controls.execute("left:pinky", 1, dj))
                            elif gesture == "echo_down":
                                recent.appendleft(controls.execute("left:pinky", 2, dj))
                            else:
                                dj.set_mix(0.0)
                                recent.appendleft("GESTURE: RETURN A")
                        continue
                    for finger, (x, y) in hand.fingertips.items():
                        point = (int(x * width), int(y * height))
                        cv2.circle(frame, point, 13, COLORS[finger], -1)
                        label = COMMAND_LABELS[f"{hand.handedness}:{finger}"]
                        cv2.putText(frame, label, (point[0] + 14, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.43, COLORS[finger], 1)
                    for hit in detector.update(hand):
                        # Play/pause needs to be immediate; it does not use
                        # the delayed double-tap router.
                        if hit.finger == "right:thumb":
                            recent.appendleft(controls.execute(hit.finger, 1, dj))
                            continue
                        for command in command_router.register(hit.finger):
                            label = controls.execute(*command, dj)
                            if label:
                                recent.appendleft(label)

            for command in command_router.due():
                label = controls.execute(*command, dj)
                if label:
                    recent.appendleft(label)

            dj.tick()
            _draw_hud(frame, recent, dj, controls)
            cv2.imshow("AirBeat", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("0"):
                recent.appendleft(controls.reset_effects())
    finally:
        camera.release()
        tracker.close()
        cv2.destroyAllWindows()

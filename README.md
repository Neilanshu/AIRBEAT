# AirBeat: two-hand DJ mode

AirBeat now tracks both hands at once.

| Hand | Fingertips, from thumb to pinky |
| --- | --- |
| Left | Kick, Snare, Hi-hat, Clap, Tom |
| Right | C, D, E, G, High C |

## DJ controls

Put songs you are allowed to use in `assets/` using these names:

```text
song.mp3
song_2.mp3
song_3.mp3
song_4.mp3
```

`song.mp3` begins playing normally in the background. Add as many of the next
three files as you want; missing files are skipped. AirBeat queues the next
available song and cycles through the playlist after each transition.

| Right fingertip | DJ action |
| --- | --- |
| Thumb | One tap: immediate play/pause toggle. |
| Index | One tap: move mix toward B. Two taps: toward A. |
| Middle | One tap: volume up. Two taps: volume down. |
| Ring | One tap: crossfade into the next playlist song. Two taps: return to the current song. |
| Pinky | Cue / restart the next playlist song. |

| Left fingertip | One tap | Two quick taps |
| --- | --- | --- |
| Thumb | Bass up | Bass down |
| Index | Mid/Vocal up | Mid/Vocal down |
| Middle | High EQ up | High EQ down |
| Ring | Reverb up | Reverb down |
| Pinky | Echo up | Echo down |

Each setting changes in 10% steps. Bass,
reverb, vocals, highs, and echo are now tracked in the interface; applying
them audibly to the background song is the next audio-engine step.

## Better finger control

AirBeat now measures each fingertip against its own wrist, so moving an entire
hand should not trigger every control. Each finger must lift before it can
trigger again. Make a quick **down then up** motion for a deliberate tap.

If controls miss taps, lower `tap_velocity_threshold` in `config.py` a little
(for example `0.006` to `0.005`). If you get accidental taps, raise it to
`0.007` or `0.008`.

## Whole-hand gestures

These controls work alongside fingertip taps and are intentionally larger so
they are harder to trigger by accident:

- Make a **closed fist**: play/pause the decks.
- Swipe a hand **right**: transition to deck B.
- Swipe a hand **left**: return the mix to deck A.
- Pinch **left thumb + index**: Bass.
- Pinch **left thumb + middle**: Treble/Highs.
- Pinch **left thumb + ring**: Reverb.
- Pinch **left thumb + pinky**: Echo.

Make a **quick pinch then release** to increase the setting by 10%. Hold the
pinch for about **half a second** to decrease it by 10%, then release before
the next adjustment. Press **0** to reset to neutral: Bass/Vocals/Highs 50%,
Reverb/Echo 0%.

Landmarks are now smoothed and tracked with MediaPipe's higher-accuracy model.

## CNN upgrade

The current landmark detector remains active until a trained model exists. The
new `TemporalTapCNN` learns a 12-frame motion sequence for all five fingers;
it automatically replaces the rule detector when `models/tap_cnn.pt` exists.

Install PyTorch before training:

```bash
pip install torch
```

Training data must contain `features` with shape `examples × 5 × 12` and
one-hot `labels` with shape `examples × 5`. Train it from the Desktop folder:

```bash
python -m airbeat.train_cnn --data airbeat/data/tap_examples.npz
```

The model supports either hand because it learns wrist-relative fingertip
motion rather than absolute screen positions.

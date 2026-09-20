# LOFOP Sentinel

**Edge Vision Intelligence Platform** — a PyQt6 desktop console for edge
computer-vision surveillance in two deployment scenarios: **classroom safety
monitoring** and **factory / workplace safety monitoring**.

The system performs **object detection and object tracking only**. It does not
perform facial recognition or identity inference, and it stores no biometric
data.

## Quick start

```bash
pip install -r requirements.txt
python app.py                      # opens in Simulation mode
```

Simulation mode needs **no model weights and no video files** — the built-in
synthetic CCTV source ships ground-truth annotations, so tracking, zones,
events, analytics and the dashboard are all demonstrable immediately.

```bash
python app.py --mode factory        # factory scenario
python app.py --video clip.mp4      # run over a local video file
python app.py --source webcam       # run over a capture device
python app.py --self-test           # headless pipeline check, no display needed
python -m sentinel --self-test      # identical, via the package
```

## Architecture

`app.py` is a thin entry point (~100 lines). Everything else lives in the
`sentinel` package, with imports flowing one way only — from the bottom of this
list upward.

```
app.py                       entry point: args, runtime check, GUI or self-test
sentinel/
├── config.py                paths, logging, application metadata
├── theme.py                 colour tokens + Qt style sheet (no Qt import)
├── deps.py                  optional-dependency probing; never assumes a package
├── models.py                enums + dataclasses shared by every layer
├── detection/               BaseDetector interface and the backends
│   ├── base.py              BaseDetector, NullDetector, device resolution
│   ├── rtdetr.py            RT-DETR via transformers
│   └── rfdetr.py            RF-DETR via the Roboflow rfdetr package
├── tracking/kalman.py       Kalman filter, Track, KalmanTracker
├── sources/                 frame producers
│   ├── video_file.py        local files (seek, loop, corrupt-frame handling)
│   ├── webcam.py            capture devices
│   └── simulation.py        synthetic CCTV scene + ground-truth boxes
├── vision/                  per-frame pipeline stages
│   ├── zones.py             scenario zone/line presets (normalised coords)
│   ├── night_vision.py      low-light detection + CLAHE/gamma enhancement
│   ├── thermal.py           false-colour palettes + hotspot analysis
│   ├── supervision_adapter.py  optional supervision bridge
│   ├── renderer.py          zones, boxes, IDs, HUD, thermal overlay
│   ├── events.py            event engine with severity grading + cooldowns
│   └── analytics.py         rolling time series and counters
├── storage/                 SQLite event store, settings + zone persistence
├── runtime/                 execution machinery
│   ├── frame_queue.py       bounded queue with a stale-frame drop policy
│   ├── telemetry.py         CPU / RAM / GPU sampling
│   ├── workers.py           CaptureWorker + InferenceWorker (QThread)
│   └── controller.py        PipelineController: owns source + threads
├── ui/                      Qt presentation layer
│   ├── widgets.py           StatCard, Sparkline, BarMeter, PipelineDiagram
│   ├── video_view.py        aspect-correct surface with zoom/pan/zone editing
│   ├── zone_editor.py       zone / counting-line editor dialogs
│   ├── live_panel.py        right-hand dock; editable zone + line status rows
│   ├── main_window.py       shell: chrome + wiring only
│   └── pages/               one self-contained widget per page
├── startup.py               QApplication setup, theming, autostart
└── selftest.py              headless pipeline verification
```

Only `sentinel.ui`, `sentinel.startup` and `sentinel.runtime` import Qt, so the
entire core pipeline runs and is testable without a display.

### Separation of concerns

* **Pages** own widgets, emit intent as signals, and expose explicit update
  methods. No page reaches into the pipeline or into another page.
* **`PipelineController`** owns everything with a lifecycle — source, worker
  threads, frame queue, database session — and publishes state as signals.
* **`MainWindow`** composes chrome and wires pages to the controller. It holds
  no pipeline state.

### Threading

```
CaptureWorker (QThread)  ->  bounded FrameQueue (drops stale frames)
                         ->  InferenceWorker (QThread)
                             detect -> Kalman track -> zones/events -> annotate
                         ->  latest-result slot
                         ->  GUI repaint timer (throttled)
```

The GUI thread never runs inference. When inference is slower than capture the
newest frame is processed and older ones are discarded, which bounds both
memory and latency. Model loading also happens on the worker thread, so the
interface stays responsive through a multi-second load.

## Detection backends

Both are **optional** and are reported honestly on the startup system-check
screen; a missing one never blocks the application.

| Backend | Package | Notes |
|---|---|---|
| RT-DETR | `transformers` | `RTDetrForObjectDetection` + `RTDetrImageProcessor`; RT-DETRv2 checkpoints select the v2 head automatically. |
| RF-DETR | `rfdetr` | `RFDETRNano/Small/Medium/Base/Large`; tolerates the class renames and the `COCO_CLASSES` module move across releases. |

This project deliberately does **not** depend on Ultralytics / YOLO.

> **RF-DETR resolution:** inference size is snapped to a multiple of **224**.
> That is the only step satisfying both the DINOv2 patch grid (56) and the
> windowed variants' block size (32). Other values are accepted by the
> constructor and then fail inside `predict()`.

CUDA is detected at runtime and the application falls back to CPU rather than
failing.

## Night vision (low-light enhancement)

Measures scene luminance and, when the scene is genuinely dark, rebuilds local
contrast so the detector has something to work with: CLAHE on the L channel,
gamma lift, and an optional edge-preserving denoise (gain amplifies sensor
noise, and noise produces phantom detections). `Auto` engages only below a
configurable luma threshold, with hysteresis so a borderline scene does not
flap. Costs ~6 ms/frame when engaged and nothing measurable when it is not.

Toggle it from the **Live Monitor** toolbar (or press **N**) — the button shows
`NIGHT OFF` / `NIGHT AUTO` / `NIGHT ON`, with a dot when enhancement is actually
running, which matters in Auto where the mode alone does not tell you. Right-click
the button for the three-way choice; it stays in sync with the Settings page
either way and takes effect on the next frame.

> **What it is not:** image enhancement, not a thermal or infrared sensor. It
> can only amplify light the camera actually captured — in total darkness there
> is nothing to amplify. It adds no detection capability of its own; a model
> backend is still required. Crossing between day and low light raises a
> `LOW_LIGHT` event (edge-triggered, one per transition).

## Thermal imaging

Two distinct modes, and the difference is deliberate:

| Mode | Works with | Reports |
|---|---|---|
| **False Colour** | any camera | image **intensity** as a percentage |
| **Radiometric** | a calibrated thermal camera only | **degrees Celsius** |

False Colour maps pixel brightness through a thermal ramp (iron, inferno,
rainbow, plasma, white-hot, black-hot) and finds bright-region "hotspots". It
is a display filter — the on-frame banner says so, and every reading is a
percentage.

Radiometric maps pixel values linearly onto a temperature span **you supply
from your camera's datasheet**, which is how the AGC-normalised greyscale
streams from common USB thermal cameras (FLIR Lepton/PureThermal, Seek,
Hikvision IR) are meant to be read.

> **A visible-light camera cannot measure temperature.** In False Colour mode
> every temperature field is `None` and the UI shows intensity percent, because
> printing an invented °C figure in a safety tool is worse than showing none.

Thermal is applied **after** detection: a palette-mapped frame is far outside a
COCO-trained model's input distribution, so the detector keeps receiving the
real (or night-enhanced) image.

## Editing zones live

Zone and counting-line rows in the right-hand Zone Status panel are editable in
place — double-click a row, or select it and press **Edit Selected**. The zone
editor covers name, type, severity, colour, occupancy limit and unattended
timeout; the line editor covers name, colour, enabled state and both normalised
endpoints. Changes apply to the running pipeline immediately and are persisted.

## Scope

Geometric and count-based rules only: zone entry, line crossing, occupancy,
crowding, unattended area, worker-machine proximity, and low-light transitions.

The application **cannot** infer PPE compliance, falls, weapons or fatigue.
Those are listed in the Models page explicitly as *not implemented*, together
with the model type each would require.

## Licence

MIT for this application source. Model weights and model packages remain under
their own upstream licences. Users are responsible for complying with those
licences and with all applicable privacy and surveillance regulations in their
jurisdiction.

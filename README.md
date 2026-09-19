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
│   ├── supervision_adapter.py  optional supervision bridge
│   ├── renderer.py          zones, boxes, IDs, HUD (OpenCV fallback built in)
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
│   ├── live_panel.py        right-hand live analytics dock
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

## Scope

Geometric and count-based rules only: zone entry, line crossing, occupancy,
crowding, unattended area, and worker-machine proximity.

The application **cannot** infer PPE compliance, falls, weapons or fatigue.
Those are listed in the Models page explicitly as *not implemented*, together
with the model type each would require.

## Licence

MIT for this application source. Model weights and model packages remain under
their own upstream licences. Users are responsible for complying with those
licences and with all applicable privacy and surveillance regulations in their
jurisdiction.

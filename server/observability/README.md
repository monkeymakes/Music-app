# Server-side observability module

Mounted on the main inference FastAPI app at `/v1/observability/*`.

## Components

| Module | Role |
| --- | --- |
| `telemetry_stream.py` | In-memory per-Pi sessions; micro-batch events before flush |
| `tensorboard_sink.py` | Writes scalar/text metrics to TensorBoard event files |
| `function_registry.py` | Cross-Pi function definitions and invocation |
| `routes.py` | FastAPI router |

## TensorBoard tags

Events are written with tags like:

```
pi/sensors/pi-sensors-01/distance_front_m
pi/voice/pi-voice-01/wake_word_wyoming/detected
pi/vision/pi-vision-01/depth/mean_depth_m
```

Dict values expand into sub-scalars (`metric/sub_key`).

## Smoke test

```bash
PYTHONPATH=. python3 server/observability/smoke_test.py
```

## View logs

```bash
tensorboard --logdir runs/wheelchair_observability
```

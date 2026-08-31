# Multi-Pi Streaming Observability

Streaming telemetry pipeline for a wheelchair with **three Raspberry Pis**, each with a different role. Events are micro-batched on the hub and flushed into **TensorBoard** for accumulation. Cross-Pi **function calling** lets any Pi (or the hub) invoke capabilities on another.

## Why streaming over batch?

| Approach | Latency | TensorBoard | Multi-Pi |
| --- | --- | --- | --- |
| Batch (cron/upload) | High — waits for full batch | Sparse points | Hard to correlate |
| **Streaming micro-batch** | Low — flush at N events or T ms | Continuous curves | Per-Pi sessions |

## Architecture

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  pi-sensors-01  │  │   pi-voice-01   │  │  pi-vision-01   │
│ distance + IMU  │  │ Wyoming/Aloha   │  │ stereo + depth  │
│                 │  │ mic + speaker   │  │ 3D recognition  │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │ stream events      │                    │
         └────────────────────┼────────────────────┘
                              ▼
              ┌───────────────────────────────┐
              │  Observability Hub (FastAPI)  │
              │  /v1/observability/stream/* │
              │  micro-batch → TensorBoard    │
              │  /v1/observability/functions  │
              └───────────────────────────────┘
                              │
                              ▼
                    runs/wheelchair_observability/
                    (tensorboard --logdir runs/wheelchair_observability)
```

## Pi roles

| Pi | Role | Telemetry | Function calls |
| --- | --- | --- | --- |
| **pi-sensors-01** | `sensors` | Distance (front/left/right/rear), IMU | `sensors.read_distance`, `sensors.read_imu` |
| **pi-voice-01** | `voice` | Wyoming/Aloha wake-word, audio RMS, speech activity | `voice.get_wake_word`, `voice.analyze_audio_level` |
| **pi-vision-01** | `vision` | Stereo frames, depth map stats, 3D recognition scores | `vision.capture_stereo_frame`, `vision.estimate_depth` |

## Quick start (MVP demo)

### 1. Start the hub

```bash
pip install -r server/inference/requirements.txt
export PYTHONPATH=.
export ALLOW_MOCK_INFERENCE=1
uvicorn server.inference.app:app --host 0.0.0.0 --port 8080
```

### 2. Simulate all three Pis

```bash
pip install -r edge/wheelchair/requirements.txt
python3 edge/wheelchair/run_all_pis_demo.py --hub http://127.0.0.1:8080 --duration 10
```

### 3. View TensorBoard

```bash
pip install tensorboard
tensorboard --logdir runs/wheelchair_observability --port 6006
```

Open http://localhost:6006 — you should see per-Pi scalar curves under `pi/{role}/{device_id}/`.

### 4. Call a function across Pis

```bash
curl -s -X POST http://127.0.0.1:8080/v1/observability/functions/call \
  -H 'Content-Type: application/json' \
  -d '{"function_name":"vision.estimate_depth","target_device_id":"pi-vision-01","arguments":{"region":"center"}}' | jq
```

## Run on real Pis

Copy `edge/wheelchair/config/pi_roles.example.json` and set your hub LAN IP. On each Pi:

```bash
# Sensor Pi
python3 edge/wheelchair/pi_sensors_stream.py --hub http://HUB_IP:8080 --device-id pi-sensors-01

# Voice Pi
python3 edge/wheelchair/pi_voice_stream.py --hub http://HUB_IP:8080 --device-id pi-voice-01 --engine wyoming

# Vision Pi
python3 edge/wheelchair/pi_vision_stream.py --hub http://HUB_IP:8080 --device-id pi-vision-01
```

Replace the mock readers in each `pi_*_stream.py` with your real sensor, Wyoming/Aloha, and picamera2 drivers.

## API reference

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/observability/health` | Hub + TensorBoard status |
| `POST` | `/v1/observability/stream/sessions` | Open per-Pi telemetry session |
| `POST` | `/v1/observability/stream/sessions/{id}/events` | Push one metric event |
| `POST` | `/v1/observability/stream/sessions/{id}/flush` | Force batch flush to TensorBoard |
| `GET` | `/v1/observability/functions` | List cross-Pi functions |
| `POST` | `/v1/observability/functions/call` | Invoke a function on a target Pi |

## Environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `OBSERVABILITY_TENSORBOARD_LOG_DIR` | `runs/wheelchair_observability` | TensorBoard event files |
| `OBSERVABILITY_TELEMETRY_BATCH_SIZE` | `16` | Default events per flush |
| `OBSERVABILITY_TELEMETRY_MAX_WAIT_MS` | `500` | Max wait before auto-flush |
| `OBSERVABILITY_TELEMETRY_IDLE_TTL_MS` | `120000` | Session idle expiry |

See also [`server/observability/README.md`](../../server/observability/README.md) for server-side details.

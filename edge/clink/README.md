# Clink — Apache Flink–inspired stream processing

[Clink](https://github.com/orhaugh/clink) is an embedded-first, Arrow-native C++23 stream processing engine (Apache License 2.0) built on the same conceptual model as **Apache Flink**: typed operator DAGs, event time, watermarks, keyed state, and exactly-once checkpoints — without a JVM.

Use it here when Music Vision needs **engine-grade stream semantics** (windowed frame aggregation, keyed session state, deterministic replay) on edge boxes or in-process beside Python inference.

## How this fits Music-app

| Layer | What runs today | Clink role |
| --- | --- | --- |
| iOS camera | `ServerSideStreamSession` → HTTP frames | Optional: emit NDJSON/Kafka events instead of direct HTTP batching |
| Python server | `server/inference/stream.py` in-memory sessions | Clink can window/aggregate before `/v1/infer/batch` |
| Raspberry Pi | `efficientnet_infer.py` single-shot | Clink batches frames locally with low footprint (no Java) |

Clink does **not** replace TFLite inference in this repo — it sits **in front of** inference to batch, window, and route frames with Flink-like guarantees.

## Quick start (Docker, no build)

From the repo root:

```bash
chmod +x edge/clink/run_pipeline.sh
./edge/clink/run_pipeline.sh music_vision_batch
```

Expected output (one upsert row per session as aggregates change):

```json
{"session_id":"sess-ios-001","model":"music_vision_efficientnet","channel":"production","device_class":"ios","frame_count":1,"avg_latency_ms":42.1,"peak_latency_ms":42.1}
...
{"session_id":"sess-ios-001","model":"music_vision_efficientnet","channel":"production","device_class":"ios","frame_count":5,"avg_latency_ms":41.32,"peak_latency_ms":43.0}
```

Event-time windows:

```bash
./edge/clink/run_pipeline.sh music_vision_windows
```

## Layout

| Path | Role |
| --- | --- |
| `pipelines/music_vision_batch.sql` | Per-session frame counts + latency stats |
| `pipelines/music_vision_windows.sql` | 1-second tumbling windows per session |
| `sample_events.ndjson` | Example frame events (iOS + Pi sessions) |
| `run_pipeline.sh` | Runs via local `clink` CLI or Docker runtime image |
| `compose.yaml` | One-service compose wrapper around the runtime image |

## Wiring to inference

A typical edge pipeline:

1. Camera / HTTP ingest writes NDJSON or Kafka topics with `{session_id, frame_id, ts_ms, ...}`.
2. Clink SQL aggregates or windows frames (this directory).
3. A sink connector (HTTP, file, or embedded `libclink` callback) triggers `efficientnet_infer.py` or `POST /v1/infer/batch` when `frame_count >= batch_size`.

See [`docs/clink-stream-processing.md`](../../docs/clink-stream-processing.md) for architecture notes and Flink comparison.

## Build from source (optional)

Clink targets Linux arm64/x86_64 and macOS Apple Silicon. See the upstream [README](https://github.com/orhaugh/clink#build-from-source).

```bash
git clone https://github.com/orhaugh/clink && cd clink
scripts/build-arrow.sh && scripts/build-iceberg-cpp.sh
cmake -S . -B build && cmake --build build --parallel
./build/clink run /path/to/music_vision_batch.sql
```

## Further reading

- [Clink vs Apache Flink](https://orhaugh.github.io/clink/guides/clink-vs-flink/)
- [Capability catalogue](https://orhaugh.github.io/clink/capabilities/)
- [Kafka → Clink → ClickHouse tutorial](https://orhaugh.github.io/clink/tutorials/kafka-to-clickhouse/)

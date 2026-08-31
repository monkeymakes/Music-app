-- Music Vision frame batching in Clink.
--
-- Mirrors the semantics of server/inference/stream.py:
--   - group camera frames by stream session
--   - emit rolling aggregates suitable for batched inference / analytics
--
-- Run locally (no broker required):
--   ./edge/clink/run_pipeline.sh music_vision_batch
--
-- Or with the published runtime image:
--   docker run --rm -v "$PWD:/work" -w /work \
--     ghcr.io/orhaugh/clink-runtime:latest run /work/edge/clink/pipelines/music_vision_batch.sql

CREATE TABLE vision_frames (
  ts_ms BIGINT,
  session_id VARCHAR,
  frame_id VARCHAR,
  model VARCHAR,
  channel VARCHAR,
  device_class VARCHAR,
  latency_ms DOUBLE
) WITH (
  connector = 'file',
  format = 'json',
  path = 'edge/clink/sample_events.ndjson'
);

-- Per-session aggregates: frame count and average inference latency.
-- Use this output to decide when a batch is ready for downstream TFLite infer.
SELECT
  session_id,
  model,
  channel,
  device_class,
  COUNT(*) AS frame_count,
  AVG(latency_ms) AS avg_latency_ms,
  MAX(latency_ms) AS peak_latency_ms
FROM vision_frames
GROUP BY session_id, model, channel, device_class;

-- Event-time tumbling windows over Music Vision camera frames.
--
-- Groups frames into 1-second windows per session. Useful when you want
-- time-bounded batches (similar to MUSIC_VISION_STREAM_MAX_WAIT_MS) rather
-- than count-based batch_size flushes alone.
--
-- Run:
--   ./edge/clink/run_pipeline.sh music_vision_windows

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

SELECT
  session_id,
  model,
  channel,
  device_class,
  COUNT(*) AS frames_in_window,
  AVG(latency_ms) AS avg_latency_ms
FROM vision_frames
GROUP BY
  session_id,
  model,
  channel,
  device_class,
  TUMBLE(ts_ms, INTERVAL '1' SECOND);

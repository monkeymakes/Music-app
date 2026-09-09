import type { MlRunResult } from '../lib/ml'
import type { NormalizedLocation } from '../lib/types'

type Props = {
  results: MlRunResult[]
  locations: NormalizedLocation[]
}

export function MlPanel({ results, locations }: Props) {
  return (
    <div className="ml-list">
      {results.map((r) => (
        <div className="ml-item" key={`${r.model}-${r.notes}`}>
          <strong>{r.model.toUpperCase()}</strong>
          <div className="loss-line">{r.notes}</div>
          <div className="spark" aria-label={`${r.model} loss`}>
            {r.lossHistory.map((v, i) => {
              const max = Math.max(...r.lossHistory, 1e-6)
              return (
                <span
                  key={i}
                  style={{ height: `${Math.max(8, (1 - v / max) * 100)}%`, animationDelay: `${i * 30}ms` }}
                />
              )
            })}
          </div>
          <div className="loss-line">
            loss {r.lossHistory[0]?.toFixed(4)} → {r.lossHistory.at(-1)?.toFixed(4)} over {r.epochs} epochs
          </div>
        </div>
      ))}

      {locations.length > 0 && (
        <div className="ml-item">
          <strong>Word2Vec location embeddings</strong>
          <div className="loss-line">
            Tokens from machine location, podman, hardware, cloud, and type — skip-gram vectors (16-d).
          </div>
          {locations.slice(0, 5).map((loc) => (
            <div key={loc.key} style={{ marginTop: '0.55rem' }}>
              <div className="loss-line">
                {loc.machineLocation} · {loc.cloud} · {loc.type}
              </div>
              <div className="embed-row" aria-hidden>
                {loc.embedding.slice(0, 16).map((v, i) => (
                  <i
                    key={i}
                    style={{
                      opacity: 0.25 + Math.min(0.75, Math.abs(v) * 2),
                      height: `${12 + Math.abs(v) * 28}px`,
                      background: v >= 0 ? 'var(--accent-2)' : 'var(--accent)',
                    }}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

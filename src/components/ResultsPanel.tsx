import type { CorrelationResult } from '../lib/types'
import { VennDiagram } from './VennDiagram'
import { TensorPreview } from './TensorPreview'
import { MlPanel } from './MlPanel'
import type { MlRunResult } from '../lib/ml'

type Props = {
  result: CorrelationResult
  ml: MlRunResult[]
}

export function ResultsPanel({ result, ml }: Props) {
  return (
    <div className="results">
      <section className="panel">
        <h2>Overlap</h2>
        <div className="meta-strip">
          <span className="chip accent">{result.operator}</span>
          {result.selected.map((s) => (
            <span className="chip" key={s}>
              {s}
            </span>
          ))}
          <span className="chip">{result.matchedKeys.length} matched assets</span>
          <span className="chip">{result.users.length} users</span>
          <span className="chip">{result.locations.length} locations</span>
        </div>
        <VennDiagram sources={result.selected} regions={result.regions} />
        {result.regions.length > 1 && (
          <div className="table-wrap" style={{ marginTop: '0.75rem' }}>
            <table>
              <thead>
                <tr>
                  <th>Region</th>
                  <th>Sources</th>
                  <th>Count</th>
                </tr>
              </thead>
              <tbody>
                {result.regions.map((r) => (
                  <tr key={r.id}>
                    <td>{r.label}</td>
                    <td>{r.sources.join(' · ')}</td>
                    <td>{r.keys.length}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="panel">
        <h2>Normalized assets</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Hostname</th>
                <th>IPs</th>
                <th>MACs</th>
                <th>Install</th>
                <th>Power</th>
                <th>Subnet</th>
                <th>Business domain</th>
                <th>Sources</th>
              </tr>
            </thead>
            <tbody>
              {result.assets.slice(0, 50).map((a) => (
                <tr key={a.key}>
                  <td>{a.hostname}</td>
                  <td>{a.ips.join(', ')}</td>
                  <td>{a.macs.join(', ')}</td>
                  <td>{a.installStatus}</td>
                  <td>{a.powerState}</td>
                  <td>{a.subnet}</td>
                  <td>{a.businessDomain}</td>
                  <td>{a.sources.join(', ')}</td>
                </tr>
              ))}
              {result.assets.length === 0 && (
                <tr>
                  <td colSpan={8}>No assets in this correlation region.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel">
        <h2>Level-3 tensors</h2>
        <p className="hint" style={{ marginTop: 0 }}>
          Shape = instances × timesteps × features. Asset features include install status, power
          state, IP/MAC encodings, and subnet→business-domain one-hots. Location tensor includes
          word2vec embeddings.
        </p>
        <TensorPreview
          assets={result.tensors.assets}
          users={result.tensors.users}
          locations={result.tensors.locations}
        />
      </section>

      <section className="panel">
        <h2>ML pass — CNN · RNN · GAN</h2>
        <MlPanel results={ml} locations={result.locations} />
      </section>
    </div>
  )
}

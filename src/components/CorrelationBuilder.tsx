import type { CorrelationOperatorId, DatasourceId } from '../lib/types'
import { CORRELATION_OPERATORS } from '../lib/types'

type Props = {
  available: DatasourceId[]
  slots: [DatasourceId | '', DatasourceId | '', DatasourceId | '']
  operator: CorrelationOperatorId
  onSlotsChange: (slots: [DatasourceId | '', DatasourceId | '', DatasourceId | '']) => void
  onOperatorChange: (op: CorrelationOperatorId) => void
  onRun: () => void
  onLoadSample: () => void
  onFile: (file: File) => void
  fileName: string | null
  sheetNames: string[]
  busy: boolean
  error: string | null
}

export function CorrelationBuilder({
  available,
  slots,
  operator,
  onSlotsChange,
  onOperatorChange,
  onRun,
  onLoadSample,
  onFile,
  fileName,
  sheetNames,
  busy,
  error,
}: Props) {
  const setSlot = (index: 0 | 1 | 2, value: DatasourceId | '') => {
    const next = [...slots] as [DatasourceId | '', DatasourceId | '', DatasourceId | '']
    next[index] = value
    onSlotsChange(next)
  }

  const optionsFor = (index: number) => {
    const taken = new Set(slots.filter((s, i) => s && i !== index))
    return available.filter((d) => !taken.has(d))
  }

  const canRun = slots.filter(Boolean).length >= 2 && !busy

  return (
    <aside className="panel">
      <h2>Correlation</h2>

      <div className="field">
        <label htmlFor="workbook">Workbook (multi-tab .xlsx)</label>
        <input
          id="workbook"
          type="file"
          accept=".xlsx,.xls,.csv"
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) onFile(f)
          }}
        />
        {fileName && <p className="hint">Loaded: {fileName}</p>}
        {sheetNames.length > 0 && (
          <div className="sheet-list" aria-label="Detected sheets">
            {sheetNames.map((n) => (
              <span className="sheet-pill" key={n}>
                {n}
              </span>
            ))}
          </div>
        )}
      </div>

      {([0, 1, 2] as const).map((i) => (
        <div className="field" key={i}>
          <label htmlFor={`ds-${i}`}>Datasource {i + 1}</label>
          <select
            id={`ds-${i}`}
            value={slots[i]}
            onChange={(e) => setSlot(i, e.target.value as DatasourceId | '')}
          >
            <option value="">{i === 2 ? 'Optional…' : 'Select source…'}</option>
            {optionsFor(i).map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
            {slots[i] && !optionsFor(i).includes(slots[i] as DatasourceId) && (
              <option value={slots[i]}>{slots[i]}</option>
            )}
          </select>
        </div>
      ))}

      <div className="field">
        <label>Operator</label>
        <div className="operator-grid" role="listbox" aria-label="Correlation operator">
          {CORRELATION_OPERATORS.map((op) => (
            <button
              key={op.id}
              type="button"
              className={`operator-btn${operator === op.id ? ' active' : ''}`}
              onClick={() => onOperatorChange(op.id)}
              aria-selected={operator === op.id}
            >
              <strong>
                {op.symbol} {op.label}
              </strong>
              <span>{op.description}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="cta-row">
        <button type="button" className="btn btn-primary" disabled={!canRun} onClick={onRun}>
          {busy ? 'Running…' : 'Run correlation'}
        </button>
        <button type="button" className="btn btn-secondary" onClick={onLoadSample} disabled={busy}>
          Load sample workbook
        </button>
      </div>

      {error && <p className="error">{error}</p>}
      <p className="hint">
        Pick up to three datasources, choose union / Venn / difference, then feed normalized max
        features into Level-3 tensors (instances × timesteps × features).
      </p>
    </aside>
  )
}

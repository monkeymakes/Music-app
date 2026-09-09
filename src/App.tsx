import { startTransition, useState } from 'react'
import { CorrelationBuilder } from './components/CorrelationBuilder'
import { ResultsPanel } from './components/ResultsPanel'
import { availableDatasources, runCorrelation } from './lib/correlate'
import { parseWorkbookFile, parseWorkbookBuffer } from './lib/excel'
import { runAllModels, type MlRunResult } from './lib/ml'
import type {
  CorrelationOperatorId,
  CorrelationResult,
  DatasourceId,
  WorkbookData,
} from './lib/types'

export default function App() {
  const [workbook, setWorkbook] = useState<WorkbookData | null>(null)
  const [slots, setSlots] = useState<[DatasourceId | '', DatasourceId | '', DatasourceId | '']>([
    '',
    '',
    '',
  ])
  const [operator, setOperator] = useState<CorrelationOperatorId>('intersection')
  const [result, setResult] = useState<CorrelationResult | null>(null)
  const [ml, setMl] = useState<MlRunResult[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const available = workbook ? availableDatasources(workbook) : []

  const ingest = (wb: WorkbookData) => {
    setWorkbook(wb)
    setResult(null)
    setMl([])
    setError(null)
    const ds = availableDatasources(wb)
    setSlots([ds[0] ?? '', ds[1] ?? '', ds[2] ?? ''])
  }

  const onFile = async (file: File) => {
    setBusy(true)
    setError(null)
    try {
      const wb = await parseWorkbookFile(file)
      ingest(wb)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to parse workbook')
    } finally {
      setBusy(false)
    }
  }

  const onLoadSample = async () => {
    setBusy(true)
    setError(null)
    try {
      const res = await fetch('/sample-inventory.xlsx')
      if (!res.ok) throw new Error('Sample workbook missing — run npm run sample')
      const buf = await res.arrayBuffer()
      ingest(parseWorkbookBuffer(buf, 'sample-inventory.xlsx'))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load sample')
    } finally {
      setBusy(false)
    }
  }

  const onRun = () => {
    if (!workbook) {
      setError('Upload or load a workbook first')
      return
    }
    setBusy(true)
    setError(null)
    startTransition(() => {
      try {
        const corr = runCorrelation(workbook, slots, operator)
        const models = runAllModels(corr.tensors)
        setResult(corr)
        setMl(models)
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Correlation failed')
        setResult(null)
        setMl([])
      } finally {
        setBusy(false)
      }
    })
  }

  return (
    <div className="app-shell">
      <header className="brand-hero">
        <h1>Correlator</h1>
        <p>
          Cross-source asset correlation from multi-tab Excel — Axonius, ServiceNow, VMware, Palo
          Alto, Active Directory, CrowdStrike, MDE, Illumio — into Level-3 tensors for CNN, RNN,
          and GAN workflows.
        </p>
      </header>

      <div className="workspace">
        <CorrelationBuilder
          available={available}
          slots={slots}
          operator={operator}
          onSlotsChange={setSlots}
          onOperatorChange={setOperator}
          onRun={onRun}
          onLoadSample={onLoadSample}
          onFile={onFile}
          fileName={workbook?.fileName ?? null}
          sheetNames={workbook?.sheets.map((s) => s.name) ?? []}
          busy={busy}
          error={error}
        />

        <main>
          {result ? (
            <ResultsPanel result={result} ml={ml} />
          ) : (
            <section className="panel">
              <h2>Ready when you are</h2>
              <p className="hint" style={{ marginTop: 0 }}>
                Load the sample workbook or drop your own spreadsheet with one tab per datasource.
                Select two or three sources, pick an operator (union, Venn intersection, difference),
                and run. Normalized rows keep the maximum feature set — IPs, MACs, install status,
                power state, subnet→business domain — plus user and location tensors with word2vec
                embeddings (machine location, podman, hardware, cloud, type).
              </p>
              <div className="cta-row">
                <button type="button" className="btn btn-primary" onClick={onLoadSample} disabled={busy}>
                  {busy ? 'Loading…' : 'Load sample workbook'}
                </button>
              </div>
            </section>
          )}
        </main>
      </div>
    </div>
  )
}

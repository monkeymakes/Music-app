import type { CorrelationRegion, DatasourceId } from '../lib/types'

type Props = {
  sources: DatasourceId[]
  regions: CorrelationRegion[]
}

function regionCount(regions: CorrelationRegion[], predicate: (r: CorrelationRegion) => boolean) {
  return regions.filter(predicate).reduce((s, r) => s + r.keys.length, 0)
}

export function VennDiagram({ sources, regions }: Props) {
  const a = sources[0] ?? 'A'
  const b = sources[1] ?? 'B'
  const c = sources[2]

  const onlyA = regionCount(regions, (r) => r.sources.length === 1 && r.sources[0] === a)
  const onlyB = regionCount(regions, (r) => r.sources.length === 1 && r.sources[0] === b)
  const onlyC = c ? regionCount(regions, (r) => r.sources.length === 1 && r.sources[0] === c) : 0
  const ab = regionCount(
    regions,
    (r) => r.sources.length === 2 && r.sources.includes(a) && r.sources.includes(b),
  )
  const ac = c
    ? regionCount(
        regions,
        (r) => r.sources.length === 2 && r.sources.includes(a) && r.sources.includes(c),
      )
    : 0
  const bc = c
    ? regionCount(
        regions,
        (r) => r.sources.length === 2 && r.sources.includes(b) && r.sources.includes(c),
      )
    : 0
  const abc = c
    ? regionCount(regions, (r) => r.sources.length === 3)
    : regionCount(regions, (r) => r.sources.length === 2 && r.sources.includes(a) && r.sources.includes(b))

  // For intersection/union operators that only return one region, fall back to matched size labels.
  const single = regions.length === 1 ? regions[0].keys.length : null

  return (
    <div className="venn-wrap">
      <svg className="venn-svg" viewBox="0 0 420 280" role="img" aria-label="Venn diagram">
        <defs>
          <filter id="soft">
            <feGaussianBlur stdDeviation="0.5" />
          </filter>
        </defs>
        <circle cx={c ? 155 : 160} cy={130} r={88} fill="rgba(11,61,74,0.18)" stroke="#0B3D4A" strokeWidth="2" />
        <circle cx={c ? 265 : 260} cy={130} r={88} fill="rgba(42,143,134,0.18)" stroke="#2A8F86" strokeWidth="2" />
        {c && (
          <circle cx={210} cy={195} r={88} fill="rgba(201,133,42,0.16)" stroke="#C9852A" strokeWidth="2" />
        )}
        <text x={c ? 100 : 110} y={90} fontSize="12" fill="#0B3D4A" fontWeight="700">
          {a}
        </text>
        <text x={c ? 290 : 285} y={90} fontSize="12" fill="#2A8F86" fontWeight="700">
          {b}
        </text>
        {c && (
          <text x={195} y={268} fontSize="12" fill="#C9852A" fontWeight="700">
            {c}
          </text>
        )}
        {single != null ? (
          <text x="210" y="140" textAnchor="middle" fontSize="22" fontWeight="800" fill="#10242B">
            {single}
          </text>
        ) : (
          <>
            <text x={c ? 115 : 120} y={135} textAnchor="middle" fontSize="14" fontWeight="700">
              {onlyA}
            </text>
            <text x={c ? 305 : 300} y={135} textAnchor="middle" fontSize="14" fontWeight="700">
              {onlyB}
            </text>
            {c && (
              <text x="210" y="230" textAnchor="middle" fontSize="14" fontWeight="700">
                {onlyC}
              </text>
            )}
            <text x="210" y={c ? 125 : 140} textAnchor="middle" fontSize="16" fontWeight="800">
              {c ? abc : ab || abc}
            </text>
            {c && (
              <>
                <text x="155" y="175" textAnchor="middle" fontSize="12" fontWeight="600">
                  {ac}
                </text>
                <text x="265" y="175" textAnchor="middle" fontSize="12" fontWeight="600">
                  {bc}
                </text>
                <text x="210" y="100" textAnchor="middle" fontSize="12" fontWeight="600">
                  {ab}
                </text>
              </>
            )}
          </>
        )}
      </svg>
    </div>
  )
}

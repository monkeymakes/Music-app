import type {
  CorrelationOperatorId,
  CorrelationRegion,
  CorrelationResult,
  DatasourceId,
  NormalizedAsset,
  NormalizedLocation,
  NormalizedUser,
  WorkbookData,
} from './types'
import { normalizeWorkbook } from './normalize'
import { buildAssetTensor, buildLocationTensor, buildUserTensor } from './tensors'

function setFor(bySource: Map<DatasourceId, Set<string>>, ds: DatasourceId): Set<string> {
  return bySource.get(ds) ?? new Set()
}

function intersectMany(sets: Set<string>[]): Set<string> {
  if (!sets.length) return new Set()
  let out = new Set(sets[0])
  for (let i = 1; i < sets.length; i++) {
    out = new Set([...out].filter((k) => sets[i].has(k)))
  }
  return out
}

function unionMany(sets: Set<string>[]): Set<string> {
  const out = new Set<string>()
  for (const s of sets) for (const k of s) out.add(k)
  return out
}

function exclusiveOnly(sets: Set<string>[]): Set<string> {
  const counts = new Map<string, number>()
  for (const s of sets) {
    for (const k of s) counts.set(k, (counts.get(k) ?? 0) + 1)
  }
  return new Set([...counts.entries()].filter(([, c]) => c === 1).map(([k]) => k))
}

function buildVennRegions(
  selected: DatasourceId[],
  bySource: Map<DatasourceId, Set<string>>,
): CorrelationRegion[] {
  const n = selected.length
  const regions: CorrelationRegion[] = []
  // Bitmask regions for up to 3 sets
  for (let mask = 1; mask < 1 << n; mask++) {
    const sources = selected.filter((_, i) => (mask & (1 << i)) !== 0)
    const include = sources.map((s) => setFor(bySource, s))
    const exclude = selected.filter((s) => !sources.includes(s)).map((s) => setFor(bySource, s))
    let keys = intersectMany(include)
    for (const ex of exclude) {
      keys = new Set([...keys].filter((k) => !ex.has(k)))
    }
    const label =
      sources.length === n
        ? 'All three overlap'
        : sources.length === 1
          ? `Only ${sources[0]}`
          : sources.join(' ∩ ')
    regions.push({
      id: `r${mask}`,
      label,
      sources,
      keys: [...keys],
    })
  }
  return regions
}

export function runCorrelation(
  workbook: WorkbookData,
  selected: [DatasourceId | '', DatasourceId | '', DatasourceId | ''],
  operator: CorrelationOperatorId,
): CorrelationResult {
  const sources = selected.filter(Boolean) as DatasourceId[]
  if (sources.length < 2) {
    throw new Error('Select at least two datasources')
  }

  const { assets, users, locations, bySource } = normalizeWorkbook(workbook)
  const sets = sources.map((s) => setFor(bySource, s))

  let matched: Set<string>
  let regions: CorrelationRegion[] = []

  switch (operator) {
    case 'union':
      matched = unionMany(sets)
      regions = [{ id: 'union', label: 'Union', sources, keys: [...matched] }]
      break
    case 'intersection':
      matched = intersectMany(sets)
      regions = [{ id: 'intersection', label: 'Intersection', sources, keys: [...matched] }]
      break
    case 'difference':
      matched = new Set(sets[0])
      for (let i = 1; i < sets.length; i++) {
        matched = new Set([...matched].filter((k) => !sets[i].has(k)))
      }
      regions = [{ id: 'difference', label: `${sources[0]} − others`, sources: [sources[0]], keys: [...matched] }]
      break
    case 'symmetric':
      matched = exclusiveOnly(sets)
      regions = [{ id: 'symmetric', label: 'Symmetric difference', sources, keys: [...matched] }]
      break
    case 'venn':
      regions = buildVennRegions(sources, bySource)
      matched = unionMany(sets)
      break
    default:
      matched = unionMany(sets)
  }

  const assetList = assets.filter((a) => matched.has(a.key))
  const userList = filterUsersForAssets(users, assetList)
  const locationList = filterLocationsForAssets(locations, sources)

  return {
    operator,
    selected: sources,
    matchedKeys: [...matched],
    regions,
    assets: assetList,
    users: userList,
    locations: locationList,
    tensors: {
      assets: buildAssetTensor(assetList),
      users: buildUserTensor(userList),
      locations: buildLocationTensor(locationList),
    },
  }
}

function filterUsersForAssets(users: NormalizedUser[], assets: NormalizedAsset[]): NormalizedUser[] {
  // Keep users that share a source with matched assets; prefer overlap density.
  const sourceSet = new Set(assets.flatMap((a) => a.sources))
  const filtered = users.filter((u) => u.sources.some((s) => sourceSet.has(s)))
  return filtered.length ? filtered : users.slice(0, Math.min(users.length, assets.length || 8))
}

function filterLocationsForAssets(
  locations: NormalizedLocation[],
  sources: DatasourceId[],
): NormalizedLocation[] {
  const filtered = locations.filter((l) => l.sources.some((s) => sources.includes(s)))
  return filtered.length ? filtered : locations
}

export function availableDatasources(workbook: WorkbookData): DatasourceId[] {
  const set = new Set<DatasourceId>()
  for (const sheet of workbook.sheets) {
    if (sheet.datasource) set.add(sheet.datasource)
  }
  return [...set]
}

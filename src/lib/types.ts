/** Known inventory / security datasources (workbook sheet names). */
export const DATASOURCES = [
  'Axonius',
  'ServiceNow',
  'VMware',
  'Palo Alto',
  'Active Directory',
  'CrowdStrike',
  'MDE',
  'Illumio',
] as const

export type DatasourceId = (typeof DATASOURCES)[number]

export const CORRELATION_OPERATORS = [
  { id: 'union', label: 'Union', symbol: '∪', description: 'All assets present in any selected source' },
  { id: 'intersection', label: 'Intersection (Venn)', symbol: '∩', description: 'Assets present in every selected source' },
  { id: 'difference', label: 'Difference', symbol: '−', description: 'In first source only (exclude others)' },
  { id: 'symmetric', label: 'Symmetric Diff', symbol: 'Δ', description: 'In exactly one of the selected sources' },
  { id: 'venn', label: 'Full Venn Regions', symbol: '◎', description: 'Partition into exclusive and overlapping regions' },
] as const

export type CorrelationOperatorId = (typeof CORRELATION_OPERATORS)[number]['id']

/** Maximum feature set harvested across all datasources. */
export type NormalizedAsset = {
  key: string
  hostname: string
  ips: string[]
  macs: string[]
  installStatus: string
  powerState: string
  subnet: string
  businessDomain: string
  sources: DatasourceId[]
  raw: Record<string, unknown>
  /** Timestep snapshots when available (sheet history / last-seen series). */
  timesteps: AssetTimestep[]
}

export type AssetTimestep = {
  t: number
  installStatus: string
  powerState: string
  ipPrimary: string
  subnet: string
}

export type NormalizedUser = {
  key: string
  username: string
  email: string
  department: string
  groups: string[]
  lastLogin: string
  sources: DatasourceId[]
}

export type NormalizedLocation = {
  key: string
  machineLocation: string
  podman: string
  hardware: string
  cloud: string
  type: string
  sources: DatasourceId[]
  /** Word2vec-style embedding of location tokens. */
  embedding: number[]
}

export type SheetTable = {
  name: string
  datasource: DatasourceId | null
  headers: string[]
  rows: Record<string, string>[]
}

export type WorkbookData = {
  fileName: string
  sheets: SheetTable[]
}

/** Level-3 tensors: instances × timesteps × features */
export type Level3Tensor = {
  name: string
  shape: [number, number, number]
  featureNames: string[]
  /** Flat row-major: [i * T * F + t * F + f] */
  data: Float32Array
  instanceKeys: string[]
}

export type CorrelationRegion = {
  id: string
  label: string
  sources: DatasourceId[]
  keys: string[]
}

export type CorrelationResult = {
  operator: CorrelationOperatorId
  selected: DatasourceId[]
  matchedKeys: string[]
  regions: CorrelationRegion[]
  assets: NormalizedAsset[]
  users: NormalizedUser[]
  locations: NormalizedLocation[]
  tensors: {
    assets: Level3Tensor
    users: Level3Tensor
    locations: Level3Tensor
  }
}

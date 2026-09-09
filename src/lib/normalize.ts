import {
  assetKey,
  resolveColumn,
  splitMulti,
  subnetToBusinessDomain,
} from './schema'
import type {
  AssetTimestep,
  DatasourceId,
  NormalizedAsset,
  NormalizedLocation,
  NormalizedUser,
  SheetTable,
  WorkbookData,
} from './types'
import { trainLocationEmbeddings } from './embeddings'

function col(row: Record<string, string>, headers: string[], canonical: string): string {
  const key = resolveColumn(headers, canonical)
  return key ? (row[key] ?? '') : ''
}

function buildTimesteps(
  installStatus: string,
  powerState: string,
  ipPrimary: string,
  subnet: string,
): AssetTimestep[] {
  // Synthesize a short history series so L3 tensors always have T>1.
  const states = [
    { installStatus: installStatus || 'unknown', powerState: powerState === 'poweredOn' ? 'poweredOff' : (powerState || 'unknown') },
    { installStatus: installStatus || 'installed', powerState: powerState || 'poweredOn' },
    { installStatus: installStatus || 'installed', powerState: powerState || 'poweredOn' },
  ]
  return states.map((s, t) => ({
    t,
    installStatus: s.installStatus,
    powerState: s.powerState,
    ipPrimary,
    subnet,
  }))
}

export function normalizeWorkbook(workbook: WorkbookData): {
  assets: NormalizedAsset[]
  users: NormalizedUser[]
  locations: NormalizedLocation[]
  bySource: Map<DatasourceId, Set<string>>
} {
  const assetMap = new Map<string, NormalizedAsset>()
  const userMap = new Map<string, NormalizedUser>()
  const locationMap = new Map<string, NormalizedLocation>()
  const bySource = new Map<DatasourceId, Set<string>>()

  for (const sheet of workbook.sheets) {
    if (!sheet.datasource) continue
    ingestSheet(sheet, sheet.datasource, assetMap, userMap, locationMap, bySource)
  }

  const locations = [...locationMap.values()]
  const embeddings = trainLocationEmbeddings(locations, 16)
  for (const loc of locations) {
    loc.embedding = embeddings.get(loc.key) ?? new Array(16).fill(0)
  }

  return {
    assets: [...assetMap.values()],
    users: [...userMap.values()],
    locations,
    bySource,
  }
}

function ingestSheet(
  sheet: SheetTable,
  source: DatasourceId,
  assetMap: Map<string, NormalizedAsset>,
  userMap: Map<string, NormalizedUser>,
  locationMap: Map<string, NormalizedLocation>,
  bySource: Map<DatasourceId, Set<string>>,
) {
  if (!bySource.has(source)) bySource.set(source, new Set())
  const sourceKeys = bySource.get(source)!

  for (const row of sheet.rows) {
    const hostname = col(row, sheet.headers, 'hostname')
    const ips = splitMulti(col(row, sheet.headers, 'ips'))
    const macs = splitMulti(col(row, sheet.headers, 'macs'))
    const installStatus = col(row, sheet.headers, 'installStatus')
    const powerState = col(row, sheet.headers, 'powerState')
    const subnet = col(row, sheet.headers, 'subnet')
    const businessDomain = subnetToBusinessDomain(subnet)
    const key = assetKey(hostname, ips, macs)

    if (hostname || ips.length || macs.length) {
      sourceKeys.add(key)
      const existing = assetMap.get(key)
      if (existing) {
        existing.sources = [...new Set([...existing.sources, source])]
        existing.ips = [...new Set([...existing.ips, ...ips])]
        existing.macs = [...new Set([...existing.macs, ...macs])]
        if (!existing.installStatus && installStatus) existing.installStatus = installStatus
        if (!existing.powerState && powerState) existing.powerState = powerState
        if (!existing.subnet && subnet) {
          existing.subnet = subnet
          existing.businessDomain = businessDomain
        }
        Object.assign(existing.raw, row)
      } else {
        assetMap.set(key, {
          key,
          hostname: hostname || ips[0] || macs[0] || key,
          ips,
          macs,
          installStatus: installStatus || 'unknown',
          powerState: powerState || 'unknown',
          subnet: subnet || '',
          businessDomain,
          sources: [source],
          raw: { ...row },
          timesteps: buildTimesteps(installStatus, powerState, ips[0] ?? '', subnet),
        })
      }
    }

    const username = col(row, sheet.headers, 'username')
    const email = col(row, sheet.headers, 'email')
    if (username || email) {
      const ukey = (email || username).toLowerCase()
      const existing = userMap.get(ukey)
      const groups = splitMulti(col(row, sheet.headers, 'groups'))
      if (existing) {
        existing.sources = [...new Set([...existing.sources, source])]
        existing.groups = [...new Set([...existing.groups, ...groups])]
      } else {
        userMap.set(ukey, {
          key: ukey,
          username: username || email.split('@')[0],
          email: email || '',
          department: col(row, sheet.headers, 'department'),
          groups,
          lastLogin: col(row, sheet.headers, 'lastLogin'),
          sources: [source],
        })
      }
    }

    const machineLocation = col(row, sheet.headers, 'machineLocation')
    const podman = col(row, sheet.headers, 'podman')
    const hardware = col(row, sheet.headers, 'hardware')
    const cloud = col(row, sheet.headers, 'cloud')
    const type = col(row, sheet.headers, 'type')
    if (machineLocation || podman || hardware || cloud || type) {
      const lkey = [machineLocation, podman, hardware, cloud, type]
        .map((x) => x.toLowerCase())
        .join('|')
      const existing = locationMap.get(lkey)
      if (existing) {
        existing.sources = [...new Set([...existing.sources, source])]
      } else {
        locationMap.set(lkey, {
          key: lkey,
          machineLocation: machineLocation || 'unknown',
          podman: podman || 'none',
          hardware: hardware || 'unknown',
          cloud: cloud || 'on-prem',
          type: type || 'unknown',
          sources: [source],
          embedding: [],
        })
      }
    }
  }
}

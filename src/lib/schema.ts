import type { DatasourceId } from './types'
import { DATASOURCES } from './types'

/** Map sheet / column noise to a known datasource id. */
export function matchDatasource(name: string): DatasourceId | null {
  const n = name.trim().toLowerCase().replace(/[_-]+/g, ' ')
  const aliases: Record<string, DatasourceId> = {
    axonius: 'Axonius',
    servicenow: 'ServiceNow',
    'service now': 'ServiceNow',
    snow: 'ServiceNow',
    vmware: 'VMware',
    vsphere: 'VMware',
    esxi: 'VMware',
    palo: 'Palo Alto',
    'palo alto': 'Palo Alto',
    pan: 'Palo Alto',
    panorama: 'Palo Alto',
    'active directory': 'Active Directory',
    ad: 'Active Directory',
    ldap: 'Active Directory',
    crowdstrike: 'CrowdStrike',
    falcon: 'CrowdStrike',
    mde: 'MDE',
    'microsoft defender': 'MDE',
    defender: 'MDE',
    'defender for endpoint': 'MDE',
    illumio: 'Illumio',
  }
  if (aliases[n]) return aliases[n]
  for (const ds of DATASOURCES) {
    if (n.includes(ds.toLowerCase())) return ds
  }
  return null
}

/** Column aliases → canonical normalized field. */
export const COLUMN_ALIASES: Record<string, string[]> = {
  hostname: [
    'hostname',
    'host name',
    'host',
    'name',
    'asset name',
    'device name',
    'computer name',
    'ci name',
    'vm name',
    'system name',
  ],
  ips: ['ip', 'ips', 'ip address', 'ip addresses', 'ipv4', 'primary ip', 'network ip', 'mgmt ip'],
  macs: ['mac', 'macs', 'mac address', 'mac addresses', 'nic mac', 'hardware address'],
  installStatus: [
    'install status',
    'installation status',
    'status',
    'agent status',
    'sensor status',
    'compliance status',
    'operational status',
  ],
  powerState: ['power state', 'power', 'powerstatus', 'vm power state', 'guest state', 'on/off'],
  subnet: ['subnet', 'network', 'cidr', 'vlan', 'segment', 'ip network'],
  username: ['username', 'user', 'user name', 'samaccountname', 'account', 'owner', 'assigned to'],
  email: ['email', 'mail', 'user email', 'email address'],
  department: ['department', 'dept', 'ou', 'org unit', 'business unit'],
  groups: ['groups', 'group', 'member of', 'security groups', 'roles'],
  lastLogin: ['last login', 'last logon', 'last seen', 'lastseen', 'last activity'],
  machineLocation: [
    'location',
    'machine location',
    'site',
    'datacenter',
    'dc',
    'facility',
    'building',
    'rack location',
  ],
  podman: ['podman', 'pod', 'container runtime', 'runtime', 'k8s node', 'orchestration'],
  hardware: ['hardware', 'hw', 'model', 'device model', 'chassis', 'serial', 'manufacturer'],
  cloud: ['cloud', 'cloud provider', 'provider', 'aws', 'azure', 'gcp', 'hypervisor'],
  type: ['type', 'asset type', 'ci class', 'device type', 'os type', 'platform', 'category'],
}

export function normalizeHeader(h: string): string {
  return h.trim().toLowerCase().replace(/[_-]+/g, ' ').replace(/\s+/g, ' ')
}

export function resolveColumn(headers: string[], canonical: string): string | null {
  const aliases = COLUMN_ALIASES[canonical] ?? [canonical]
  const normalized = headers.map((h) => ({ raw: h, n: normalizeHeader(h) }))
  for (const alias of aliases) {
    const hit = normalized.find((h) => h.n === alias)
    if (hit) return hit.raw
  }
  for (const alias of aliases) {
    const hit = normalized.find((h) => h.n.includes(alias) || alias.includes(h.n))
    if (hit) return hit.raw
  }
  return null
}

/** Subnet CIDR / prefix → business domain. */
export const SUBNET_DOMAIN_MAP: Record<string, string> = {
  '10.10.0.0/16': 'Corporate Campus',
  '10.20.0.0/16': 'Engineering Labs',
  '10.30.0.0/16': 'Production Datacenter',
  '10.40.0.0/16': 'DMZ / Edge',
  '10.50.0.0/16': 'OT / Plant Floor',
  '10.60.0.0/16': 'Cloud Transit',
  '172.16.0.0/16': 'VDI Desktops',
  '192.168.1.0/24': 'Guest Wi-Fi',
  '100.64.0.0/16': 'CGNAT Remote Access',
}

export function subnetToBusinessDomain(subnet: string): string {
  const s = subnet.trim()
  if (!s) return 'Unmapped'
  if (SUBNET_DOMAIN_MAP[s]) return SUBNET_DOMAIN_MAP[s]
  // Prefix match on /16 families
  const m = s.match(/^(\d+\.\d+)\./)
  if (m) {
    const family = `${m[1]}.0.0/16`
    if (SUBNET_DOMAIN_MAP[family]) return SUBNET_DOMAIN_MAP[family]
  }
  if (s.startsWith('10.10.')) return 'Corporate Campus'
  if (s.startsWith('10.20.')) return 'Engineering Labs'
  if (s.startsWith('10.30.')) return 'Production Datacenter'
  if (s.startsWith('10.40.')) return 'DMZ / Edge'
  if (s.startsWith('10.50.')) return 'OT / Plant Floor'
  if (s.startsWith('10.60.')) return 'Cloud Transit'
  if (s.startsWith('172.16.')) return 'VDI Desktops'
  if (s.startsWith('192.168.')) return 'Guest Wi-Fi'
  return 'Unmapped'
}

export function splitMulti(value: string): string[] {
  if (!value) return []
  return value
    .split(/[,|;/\s]+/)
    .map((v) => v.trim())
    .filter(Boolean)
}

export function assetKey(hostname: string, ips: string[], macs: string[]): string {
  const host = hostname.trim().toLowerCase()
  if (host) return `host:${host}`
  if (macs[0]) return `mac:${macs[0].toLowerCase()}`
  if (ips[0]) return `ip:${ips[0]}`
  return `row:${Math.random().toString(36).slice(2)}`
}

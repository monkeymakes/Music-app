import type {
  Level3Tensor,
  NormalizedAsset,
  NormalizedLocation,
  NormalizedUser,
} from './types'

const STATUS_VOCAB = ['unknown', 'installed', 'missing', 'pending', 'retired', 'active', 'inactive']
const POWER_VOCAB = ['unknown', 'poweredOn', 'poweredOff', 'suspended', 'standby']
const DOMAIN_VOCAB = [
  'Unmapped',
  'Corporate Campus',
  'Engineering Labs',
  'Production Datacenter',
  'DMZ / Edge',
  'OT / Plant Floor',
  'Cloud Transit',
  'VDI Desktops',
  'Guest Wi-Fi',
  'CGNAT Remote Access',
]

function oneHot(value: string, vocab: string[]): number[] {
  const v = value || 'unknown'
  const idx = Math.max(0, vocab.findIndex((x) => x.toLowerCase() === v.toLowerCase()))
  const out = new Array(vocab.length).fill(0)
  out[idx === -1 ? 0 : idx] = 1
  return out
}

function ipToFeatures(ip: string): number[] {
  const parts = ip.split('.').map((p) => Number(p))
  if (parts.length !== 4 || parts.some((n) => Number.isNaN(n))) return [0, 0, 0, 0]
  return parts.map((n) => n / 255)
}

function macToFeatures(mac: string): number[] {
  const hex = mac.replace(/[^0-9a-fA-F]/g, '')
  if (hex.length < 12) return [0, 0, 0]
  const bytes = [0, 2, 4].map((i) => parseInt(hex.slice(i * 2, i * 2 + 2), 16) / 255)
  return bytes
}

function hashEmbed(text: string, dim: number): number[] {
  const out = new Array(dim).fill(0)
  const tokens = text.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean)
  for (const tok of tokens) {
    let h = 2166136261
    for (let i = 0; i < tok.length; i++) {
      h ^= tok.charCodeAt(i)
      h = Math.imul(h, 16777619)
    }
    const idx = Math.abs(h) % dim
    out[idx] += 1
    out[(idx + 1) % dim] += 0.5
  }
  const norm = Math.sqrt(out.reduce((s, x) => s + x * x, 0)) || 1
  return out.map((x) => x / norm)
}

export function buildAssetTensor(assets: NormalizedAsset[]): Level3Tensor {
  const featureNames = [
    ...STATUS_VOCAB.map((s) => `install:${s}`),
    ...POWER_VOCAB.map((s) => `power:${s}`),
    ...DOMAIN_VOCAB.map((s) => `domain:${s}`),
    'ip_o', 'ip_1', 'ip_2', 'ip_3',
    'mac_0', 'mac_1', 'mac_2',
    'subnet_embed_0', 'subnet_embed_1', 'subnet_embed_2', 'subnet_embed_3',
  ]
  const F = featureNames.length
  const T = Math.max(1, ...assets.map((a) => a.timesteps.length), 3)
  const I = Math.max(assets.length, 1)
  const data = new Float32Array(I * T * F)

  assets.forEach((asset, i) => {
    const steps = asset.timesteps.length
      ? asset.timesteps
      : [{ t: 0, installStatus: asset.installStatus, powerState: asset.powerState, ipPrimary: asset.ips[0] ?? '', subnet: asset.subnet }]
    for (let t = 0; t < T; t++) {
      const step = steps[Math.min(t, steps.length - 1)]
      const feats = [
        ...oneHot(step.installStatus, STATUS_VOCAB),
        ...oneHot(step.powerState, POWER_VOCAB),
        ...oneHot(asset.businessDomain, DOMAIN_VOCAB),
        ...ipToFeatures(step.ipPrimary || asset.ips[0] || ''),
        ...macToFeatures(asset.macs[0] || ''),
        ...hashEmbed(step.subnet || asset.subnet || 'none', 4),
      ]
      const base = i * T * F + t * F
      for (let f = 0; f < F; f++) data[base + f] = feats[f] ?? 0
    }
  })

  return {
    name: 'assets',
    shape: [I, T, F],
    featureNames,
    data,
    instanceKeys: assets.map((a) => a.key),
  }
}

export function buildUserTensor(users: NormalizedUser[]): Level3Tensor {
  const featureNames = [
    'user_embed_0', 'user_embed_1', 'user_embed_2', 'user_embed_3', 'user_embed_4', 'user_embed_5', 'user_embed_6', 'user_embed_7',
    'dept_embed_0', 'dept_embed_1', 'dept_embed_2', 'dept_embed_3',
    'group_count',
    'has_email',
  ]
  const F = featureNames.length
  const T = 3
  const I = Math.max(users.length, 1)
  const data = new Float32Array(I * T * F)

  users.forEach((user, i) => {
    const baseFeats = [
      ...hashEmbed(user.username || user.key, 8),
      ...hashEmbed(user.department || 'none', 4),
      Math.min(user.groups.length / 10, 1),
      user.email ? 1 : 0,
    ]
    for (let t = 0; t < T; t++) {
      const drift = 1 + t * 0.02
      const base = i * T * F + t * F
      for (let f = 0; f < F; f++) data[base + f] = (baseFeats[f] ?? 0) * (f < 8 ? drift : 1)
    }
  })

  return {
    name: 'users',
    shape: [I, T, F],
    featureNames,
    data,
    instanceKeys: users.map((u) => u.key),
  }
}

export function buildLocationTensor(locations: NormalizedLocation[]): Level3Tensor {
  const embDim = locations[0]?.embedding.length || 16
  const names = [
    ...Array.from({ length: embDim }, (_, i) => `w2v_${i}`),
    'loc_0', 'loc_1', 'loc_2', 'loc_3',
    'pod_0', 'pod_1', 'pod_2', 'pod_3',
    'hw_0', 'hw_1', 'hw_2', 'hw_3',
    'cloud_0', 'cloud_1', 'cloud_2', 'cloud_3',
    'type_0', 'type_1', 'type_2', 'type_3',
  ]
  const F = names.length
  const T = 3
  const I = Math.max(locations.length, 1)
  const data = new Float32Array(I * T * F)

  locations.forEach((loc, i) => {
    const emb = loc.embedding.length ? loc.embedding : new Array(embDim).fill(0)
    const feats = [
      ...emb,
      ...hashEmbed(loc.machineLocation, 4),
      ...hashEmbed(loc.podman, 4),
      ...hashEmbed(loc.hardware, 4),
      ...hashEmbed(loc.cloud, 4),
      ...hashEmbed(loc.type, 4),
    ]
    for (let t = 0; t < T; t++) {
      const base = i * T * F + t * F
      for (let f = 0; f < F; f++) data[base + f] = feats[f] ?? 0
    }
  })

  return {
    name: 'locations',
    shape: [I, T, F],
    featureNames: names,
    data,
    instanceKeys: locations.map((l) => l.key),
  }
}

export function tensorSummary(t: Level3Tensor): string {
  const [I, T, F] = t.shape
  const mean = t.data.reduce((s, x) => s + x, 0) / Math.max(t.data.length, 1)
  return `${t.name} [${I} × ${T} × ${F}] · mean=${mean.toFixed(4)}`
}

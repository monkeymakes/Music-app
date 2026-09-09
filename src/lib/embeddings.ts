import type { NormalizedLocation } from './types'

function tokenizeLocation(loc: NormalizedLocation): string[] {
  return [loc.machineLocation, loc.podman, loc.hardware, loc.cloud, loc.type]
    .flatMap((s) => s.toLowerCase().split(/[^a-z0-9]+/))
    .filter((t) => t.length > 1)
}

function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/**
 * Lightweight skip-gram style word2vec over location tokens
 * (machine location, podman, hardware, cloud, type).
 */
export function trainLocationEmbeddings(
  locations: NormalizedLocation[],
  dim = 16,
  epochs = 40,
): Map<string, number[]> {
  const vocab = new Map<string, number>()
  const docs = locations.map((l) => tokenizeLocation(l))
  for (const toks of docs) {
    for (const t of toks) {
      if (!vocab.has(t)) vocab.set(t, vocab.size)
    }
  }

  const V = Math.max(vocab.size, 1)
  const rand = mulberry32(42)
  const W = Array.from({ length: V }, () => Array.from({ length: dim }, () => (rand() - 0.5) / dim))
  const lr = 0.05

  for (let epoch = 0; epoch < epochs; epoch++) {
    for (const toks of docs) {
      for (let i = 0; i < toks.length; i++) {
        const center = vocab.get(toks[i])!
        for (let j = Math.max(0, i - 2); j <= Math.min(toks.length - 1, i + 2); j++) {
          if (i === j) continue
          const ctx = vocab.get(toks[j])!
          // Dot product
          let dot = 0
          for (let d = 0; d < dim; d++) dot += W[center][d] * W[ctx][d]
          const sig = 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, dot))))
          const g = lr * (1 - sig)
          for (let d = 0; d < dim; d++) {
            const a = W[center][d]
            const b = W[ctx][d]
            W[center][d] += g * b
            W[ctx][d] += g * a
          }
        }
      }
    }
  }

  const out = new Map<string, number[]>()
  for (const loc of locations) {
    const toks = tokenizeLocation(loc)
    const vec = new Array(dim).fill(0)
    for (const t of toks) {
      const idx = vocab.get(t)
      if (idx == null) continue
      for (let d = 0; d < dim; d++) vec[d] += W[idx][d]
    }
    if (toks.length) {
      for (let d = 0; d < dim; d++) vec[d] /= toks.length
    }
    const norm = Math.sqrt(vec.reduce((s, x) => s + x * x, 0)) || 1
    out.set(loc.key, vec.map((x) => x / norm))
  }
  return out
}

export function cosine(a: number[], b: number[]): number {
  let dot = 0
  let na = 0
  let nb = 0
  const n = Math.min(a.length, b.length)
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  return dot / ((Math.sqrt(na) || 1) * (Math.sqrt(nb) || 1))
}

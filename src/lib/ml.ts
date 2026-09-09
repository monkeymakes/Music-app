import type { Level3Tensor } from './types'

export type MlRunResult = {
  model: 'cnn' | 'rnn' | 'gan'
  epochs: number
  lossHistory: number[]
  notes: string
}

function tensorMean(t: Level3Tensor): number {
  let s = 0
  for (let i = 0; i < t.data.length; i++) s += t.data[i]
  return s / Math.max(t.data.length, 1)
}

function sliceInstance(t: Level3Tensor, i: number): Float32Array {
  const [, T, F] = t.shape
  const out = new Float32Array(T * F)
  const base = i * T * F
  for (let k = 0; k < out.length; k++) out[k] = t.data[base + k] ?? 0
  return out
}

/** 1-D conv over timesteps for each feature channel — demo CNN. */
export function runCnn(tensor: Level3Tensor, epochs = 12): MlRunResult {
  const [I, T, F] = tensor.shape
  const kernel = new Float32Array([0.2, 0.5, 0.2])
  const lossHistory: number[] = []
  let weights = new Float32Array(F).map(() => 0.1)

  for (let e = 0; e < epochs; e++) {
    let loss = 0
    for (let i = 0; i < I; i++) {
      const x = sliceInstance(tensor, i)
      // Convolve over time for each feature
      let score = 0
      for (let f = 0; f < F; f++) {
        for (let t = 1; t < T - 1; t++) {
          const v =
            kernel[0] * x[(t - 1) * F + f] +
            kernel[1] * x[t * F + f] +
            kernel[2] * x[(t + 1) * F + f]
          score += v * weights[f]
        }
      }
      const target = tensorMean(tensor)
      const err = score / Math.max(F * (T - 2), 1) - target
      loss += err * err
      for (let f = 0; f < F; f++) weights[f] -= 0.01 * err
    }
    lossHistory.push(loss / Math.max(I, 1))
  }

  return {
    model: 'cnn',
    epochs,
    lossHistory,
    notes: `1-D CNN over timesteps on ${tensor.name} [${I}×${T}×${F}]`,
  }
}

/** Elman-style RNN over timesteps. */
export function runRnn(tensor: Level3Tensor, epochs = 12): MlRunResult {
  const [I, T, F] = tensor.shape
  const H = 8
  let Wx = Array.from({ length: H }, () => Array.from({ length: F }, () => 0.05))
  let Wh = Array.from({ length: H }, () => Array.from({ length: H }, () => 0.02))
  const lossHistory: number[] = []

  for (let e = 0; e < epochs; e++) {
    let loss = 0
    for (let i = 0; i < I; i++) {
      const x = sliceInstance(tensor, i)
      let h = new Array(H).fill(0)
      for (let t = 0; t < T; t++) {
        const next = new Array(H).fill(0)
        for (let hh = 0; hh < H; hh++) {
          let s = 0
          for (let f = 0; f < F; f++) s += Wx[hh][f] * x[t * F + f]
          for (let k = 0; k < H; k++) s += Wh[hh][k] * h[k]
          next[hh] = Math.tanh(s)
        }
        h = next
      }
      const pred = h.reduce((a, b) => a + b, 0) / H
      const err = pred - 0.5
      loss += err * err
      // Tiny gradient nudge
      for (let hh = 0; hh < H; hh++) {
        for (let f = 0; f < Math.min(F, 4); f++) Wx[hh][f] -= 0.005 * err
      }
    }
    lossHistory.push(loss / Math.max(I, 1))
  }

  return {
    model: 'rnn',
    epochs,
    lossHistory,
    notes: `RNN hidden=${H} on ${tensor.name} sequence length ${T}`,
  }
}

/** Tiny GAN: generator synthesizes fake location/asset rows vs real batch. */
export function runGan(tensor: Level3Tensor, epochs = 16): MlRunResult {
  const [I, , F] = tensor.shape
  const lossHistory: number[] = []
  let g = new Float32Array(F).map((_, i) => (i % 3 === 0 ? 0.2 : 0.05))
  let d = new Float32Array(F).map(() => 0.1)

  const realMean = new Float32Array(F)
  for (let i = 0; i < I; i++) {
    for (let f = 0; f < F; f++) {
      // average over timesteps for instance 0.. for stability use first timestep
      realMean[f] += tensor.data[i * tensor.shape[1] * F + f] ?? 0
    }
  }
  for (let f = 0; f < F; f++) realMean[f] /= Math.max(I, 1)

  for (let e = 0; e < epochs; e++) {
    // Discriminator score
    let realScore = 0
    let fakeScore = 0
    for (let f = 0; f < F; f++) {
      realScore += d[f] * realMean[f]
      fakeScore += d[f] * g[f]
    }
    const realP = 1 / (1 + Math.exp(-realScore))
    const fakeP = 1 / (1 + Math.exp(-fakeScore))
    // D wants real→1 fake→0; G wants fake→1
    const dLoss = -Math.log(realP + 1e-6) - Math.log(1 - fakeP + 1e-6)
    const gLoss = -Math.log(fakeP + 1e-6)
    for (let f = 0; f < F; f++) {
      d[f] += 0.02 * ((1 - realP) * realMean[f] - fakeP * g[f])
      g[f] += 0.02 * (1 - fakeP) * d[f]
    }
    lossHistory.push(dLoss + gLoss)
  }

  return {
    model: 'gan',
    epochs,
    lossHistory,
    notes: `GAN feature generator vs ${tensor.name} empirical mean (${F}d)`,
  }
}

export function runAllModels(tensors: {
  assets: Level3Tensor
  users: Level3Tensor
  locations: Level3Tensor
}): MlRunResult[] {
  return [
    runCnn(tensors.assets),
    runRnn(tensors.assets),
    runGan(tensors.locations),
    runRnn(tensors.users),
  ]
}

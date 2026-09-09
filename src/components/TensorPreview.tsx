import type { Level3Tensor } from '../lib/types'
import { tensorSummary } from '../lib/tensors'

type Props = {
  assets: Level3Tensor
  users: Level3Tensor
  locations: Level3Tensor
}

function Spark({ tensor }: { tensor: Level3Tensor }) {
  const [, , F] = tensor.shape
  const cols = Math.min(24, F)
  const vals: number[] = []
  for (let f = 0; f < cols; f++) {
    let s = 0
    const step = Math.max(1, Math.floor(tensor.data.length / cols))
    for (let i = f; i < tensor.data.length; i += step) s += Math.abs(tensor.data[i] ?? 0)
    vals.push(s)
  }
  const max = Math.max(...vals, 1e-6)
  return (
    <div className="spark" aria-hidden>
      {vals.map((v, i) => (
        <span key={i} style={{ height: `${Math.max(8, (v / max) * 100)}%`, animationDelay: `${i * 20}ms` }} />
      ))}
    </div>
  )
}

export function TensorPreview({ assets, users, locations }: Props) {
  const cards = [
    { title: 'Assets tensor', tensor: assets, blurb: 'assets · IPs · MACs · install · power · subnet→domain' },
    { title: 'Users tensor', tensor: users, blurb: 'identity · department · groups · login affinity' },
    { title: 'Locations tensor', tensor: locations, blurb: 'machine · podman · hardware · cloud · type + w2v' },
  ]

  return (
    <div className="tensor-grid">
      {cards.map((c) => (
        <article className="tensor-card" key={c.title}>
          <h3>{c.title}</h3>
          <p>{tensorSummary(c.tensor)}</p>
          <p>{c.blurb}</p>
          <Spark tensor={c.tensor} />
        </article>
      ))}
    </div>
  )
}

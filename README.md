# Correlator

Cross-source inventory correlation UI for multi-tab Excel workbooks (Axonius, ServiceNow, VMware, Palo Alto, Active Directory, CrowdStrike, MDE, Illumio).

## What it does

1. Upload a workbook (one sheet per datasource) or load the sample.
2. Select up to **three datasources** and an operator: **union**, **intersection (Venn)**, **difference**, **symmetric difference**, or **full Venn regions**.
3. Normalize the maximum feature set across sources (hostname, IPs, MACs, install status, power state, subnet → business domain, users, locations).
4. Build **Level-3 tensors** shaped `[instances × timesteps × features]` for assets, users, and locations.
5. Run lightweight in-browser **CNN / RNN / GAN** passes and **word2vec** embeddings over location tokens (machine location, podman, hardware, cloud, type).

## Develop

```bash
npm install
npm run sample   # writes public/sample-inventory.xlsx
npm run dev
```

## Build / Netlify

```bash
npm run build
```

`netlify.toml` publishes `dist` with SPA fallback. Local Vite uses the React plugin only (the Netlify Vite plugin’s edge emulator can fail in some sandboxes); production still deploys via Netlify from `netlify.toml`.

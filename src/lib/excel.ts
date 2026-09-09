import * as XLSX from 'xlsx'
import { matchDatasource } from './schema'
import type { SheetTable, WorkbookData } from './types'

function cellToString(v: unknown): string {
  if (v == null) return ''
  if (typeof v === 'string') return v.trim()
  if (typeof v === 'number' || typeof v === 'boolean') return String(v)
  if (v instanceof Date) return v.toISOString()
  return String(v)
}

export async function parseWorkbookFile(file: File): Promise<WorkbookData> {
  const buf = await file.arrayBuffer()
  return parseWorkbookBuffer(buf, file.name)
}

export function parseWorkbookBuffer(buf: ArrayBuffer, fileName: string): WorkbookData {
  const wb = XLSX.read(buf, { type: 'array', cellDates: true })
  const sheets: SheetTable[] = wb.SheetNames.map((name) => {
    const sheet = wb.Sheets[name]
    const matrix = XLSX.utils.sheet_to_json<(string | number | boolean | Date | null)[]>(sheet, {
      header: 1,
      defval: '',
      raw: false,
    }) as unknown[][]

    if (!matrix.length) {
      return { name, datasource: matchDatasource(name), headers: [], rows: [] }
    }

    const headerRow = (matrix[0] ?? []).map((c) => cellToString(c) || 'column')
    // Deduplicate headers
    const seen = new Map<string, number>()
    const headers = headerRow.map((h) => {
      const base = h || 'column'
      const count = seen.get(base) ?? 0
      seen.set(base, count + 1)
      return count === 0 ? base : `${base}_${count}`
    })

    const rows = matrix.slice(1).map((row) => {
      const obj: Record<string, string> = {}
      headers.forEach((h, i) => {
        obj[h] = cellToString(row?.[i])
      })
      return obj
    }).filter((row) => Object.values(row).some((v) => v.length > 0))

    return {
      name,
      datasource: matchDatasource(name),
      headers,
      rows,
    }
  })

  return { fileName, sheets }
}

export function sheetsByDatasource(workbook: WorkbookData): Map<string, SheetTable> {
  const map = new Map<string, SheetTable>()
  for (const sheet of workbook.sheets) {
    const key = sheet.datasource ?? sheet.name
    map.set(key, sheet)
  }
  return map
}

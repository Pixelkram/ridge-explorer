import type { GridStartRequest, GridStartResponse, GridStatusResponse, RefineRequest, RefineResponse, SeedProbeRequest, SeedProbeResponse, SeedProbeStatus, FastScanRequest, FastScanResponse, GenerateSelectedRequest, GenerateSelectedResponse } from './types';

const BASE = '';

export async function startGrid(req: GridStartRequest): Promise<GridStartResponse> {
  const res = await fetch(`${BASE}/api/grid/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return res.json();
}

export async function getGridStatus(jobId: string): Promise<GridStatusResponse> {
  const res = await fetch(`${BASE}/api/grid/${jobId}/status`);
  return res.json();
}

export async function refineGrid(jobId: string, req: RefineRequest): Promise<RefineResponse> {
  const res = await fetch(`${BASE}/api/grid/${jobId}/refine`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return res.json();
}

export async function startSeedProbe(jobId: string, req: SeedProbeRequest): Promise<SeedProbeResponse> {
  const res = await fetch(`${BASE}/api/grid/${jobId}/seed-probe`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return res.json();
}

export async function getSeedProbeStatus(probeId: string): Promise<SeedProbeStatus> {
  const res = await fetch(`${BASE}/api/grid/probe/${probeId}/status`);
  return res.json();
}

export async function startFastScan(req: FastScanRequest): Promise<FastScanResponse> {
  const res = await fetch(`${BASE}/api/grid/fast-scan`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return res.json();
}

export async function generateSelected(jobId: string, req: GenerateSelectedRequest): Promise<GenerateSelectedResponse> {
  const res = await fetch(`${BASE}/api/grid/${jobId}/generate-selected`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return res.json();
}

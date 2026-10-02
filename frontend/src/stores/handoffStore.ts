/**
 * Handoffs between panels: "open this recipe in another surface".
 *
 * The panels are siblings with their own local state, so a button in one that should open
 * another needs a channel neither owns. This is that channel and nothing more: the sender
 * posts a request with a fresh `nonce`, the receiving panel reacts to the nonce changing
 * (open itself, fill its form, scroll into view) and keeps its own state from there. Nothing
 * here survives a reload, and nothing here is a run -- the receiver still asks the backend.
 */
import { create } from 'zustand';

/** "Show me a microscope plane here." From a Cascade crossing (crossing mode) or a bare
 *  recipe (free mode: the centre). */
export interface ScopeHandoff {
  nonce: number;
  from: 'cascade' | 'desk';
  prompts: string[];
  weights: number[];
  cascadeRunId?: string | null;
  cid?: number | null;
  seed?: number;
  steps?: number;
}

interface HandoffState {
  scope: ScopeHandoff | null;
  openInMicroscope: (h: Omit<ScopeHandoff, 'nonce'>) => void;
}

export const useHandoff = create<HandoffState>((set, get) => ({
  scope: null,
  openInMicroscope: (h) => set({ scope: { ...h, nonce: (get().scope?.nonce ?? 0) + 1 } }),
}));

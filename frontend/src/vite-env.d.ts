/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Comma-separated features or presets to hide. See featureFlags.ts. */
  readonly VITE_RIDGE_HIDE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

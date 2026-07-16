export type CartState = 'forward' | 'backward' | 'stop' | 'left' | 'right' | 'search'
export type AppMode = 'label' | 'train' | 'infer'
export type DatasetSplit = 'train' | 'test'

export interface PredictionMessage {
  class_name: CartState
  class_index: number
  confidence: number
  commanded_state: CartState
  driving?: boolean
  paused?: boolean
  residual?: number
  forced_search?: boolean
}

export const KEY_TO_STATE: Record<string, CartState> = {
  w: 'forward',
  s: 'backward',
  q: 'stop',
  a: 'left',
  d: 'right',
  e: 'search',
}

export const STATE_LABELS: Record<CartState, string> = {
  forward: 'FORWARD',
  backward: 'BACKWARD',
  stop: 'STOP',
  left: 'LEFT',
  right: 'RIGHT',
  search: 'SEARCH',
}

export const EMPTY_COUNTS: Record<CartState, number> = {
  forward: 0,
  backward: 0,
  stop: 0,
  left: 0,
  right: 0,
  search: 0,
}

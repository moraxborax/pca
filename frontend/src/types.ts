export type CartState = 'forward' | 'backward' | 'stop' | 'left' | 'right' | 'search'
export type AppMode = 'label' | 'train' | 'infer'

export interface PredictionMessage {
  class_name: CartState
  class_index: number
  confidence: number
  commanded_state: CartState
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

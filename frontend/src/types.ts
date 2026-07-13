export type CartState = 'forward' | 'stop' | 'left' | 'right' | 'search'
export type AppMode = 'label' | 'infer'

export interface PredictionMessage {
  class_name: CartState
  class_index: number
  confidence: number
  commanded_state: CartState
}

export const KEY_TO_STATE: Record<string, CartState> = {
  w: 'forward',
  s: 'stop',
  a: 'left',
  d: 'right',
  q: 'search',
}

export const STATE_LABELS: Record<CartState, string> = {
  forward: 'FORWARD',
  stop: 'STOP',
  left: 'LEFT',
  right: 'RIGHT',
  search: 'SEARCH',
}

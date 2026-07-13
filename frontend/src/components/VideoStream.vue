<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'

const videoRef = ref<HTMLVideoElement | null>(null)
const status = ref('Connecting...')
let pc: RTCPeerConnection | null = null

async function startStream() {
  try {
    pc = new RTCPeerConnection()
    pc.addTransceiver('video', { direction: 'recvonly' })

    pc.ontrack = (event) => {
      if (videoRef.value) {
        videoRef.value.srcObject = event.streams[0] ?? null
        status.value = 'Connected'
      }
    }

    const offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    const response = await fetch('/offer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sdp: pc.localDescription?.sdp,
        type: pc.localDescription?.type,
      }),
    })

    if (!response.ok) {
      throw new Error(`Offer failed: ${response.status}`)
    }

    const answer = await response.json()
    await pc.setRemoteDescription(answer)
  } catch (error) {
    status.value = `Error: ${error instanceof Error ? error.message : 'unknown'}`
  }
}

onMounted(() => {
  startStream()
})

onUnmounted(() => {
  pc?.close()
})
</script>

<template>
  <div class="video-panel">
    <video ref="videoRef" autoplay playsinline muted />
    <span class="status">{{ status }}</span>
  </div>
</template>

<style scoped>
.video-panel {
  position: relative;
  background: #000;
  border-radius: 8px;
  overflow: hidden;
  aspect-ratio: 4 / 3;
}

video {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.status {
  position: absolute;
  bottom: 8px;
  left: 8px;
  background: rgba(0, 0, 0, 0.6);
  color: #fff;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
}
</style>

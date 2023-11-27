<template>

  <div class="camera-container">
    <video ref="videoElement" class="camera-preview"></video>
  </div>

</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref } from "vue";

const videoElement = ref(null);

onMounted(async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoElement.value.srcObject = stream;

    // listening loadedmetadata status
    videoElement.value.addEventListener('loadedmetadata', () => {
      console.log('Video metadata loaded:', videoElement.value.srcObject);
      
      // try play video
      videoElement.value.play().catch((error) => {
        console.error('Error playing video:', error);
      });
    });
  } catch (error) {
    console.error("Error accessing the camera: ", error);
  }
});

onUnmounted(() => {
  if (videoElement.value && videoElement.value.srcObject) {
    const tracks = (videoElement.value.srcObject as MediaStream).getTracks();
    tracks.forEach(track => track.stop());
  }
});
</script>

<style scoped>
.camera-container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
}

.camera-preview {
  width: 100%;
  max-width: 640px;
  height: auto;
}
</style>
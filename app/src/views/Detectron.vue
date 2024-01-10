<template>
  <div id="app">
    <!-- 上部分 You Only Look Once 字体的一半高度，并垂直居中在右上角 -->
    <div class="top-section">
        <div class="model-version">
          <label for="model-select" class="model-label">Select Model Version: </label>
          <select id="model-select" v-model="selectedModel" class="model-select">
            <option value="yolo-v7">Yolo v7</option>
            <option value="yolo-v8">Yolo v8</option>
          </select>
        </div>
      <div class="title">
        <h1 class="flower-font">You Only Look Once</h1>
      </div>
    </div>

    <!-- 下部分 不是9/10，而是剩余的部分 -->
    <div class="bottom-section">
      <div class="detection-box">
        <img ref="viewerElement" class="detection-image" />

        <!-- 这里假设有一个框来显示检测结果 -->
        <div class="detected-object">
          <!-- 检测到的对象信息 -->
        </div>
      </div>

      <!-- 横排按钮放在bottom-section的下边缘线上 -->
      <div class="buttons">
        <button @click="openCamera">Open Camera</button>
        <button @click="startDetection">Start Detection</button>
        <button @click="stopDetection">Stop Detection</button>
        <!-- 添加其他按钮 -->
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      socket: null,
      selectedModel: "yolo-v7", // 默认选择的模型版本
    };
  },

  mounted() {
    this.connectWebSocket();

    this.socket.addEventListener('open', (event) => {
      console.log('WebSocket connection opened:', event);
    });

    this.socket.addEventListener('error', (event) => {
      console.error('WebSocket error:', event);
    });

    this.socket.addEventListener('close', (event) => {
      console.log('WebSocket connection closed:', event);
    });

    this.socket.addEventListener('message', (event) => {
      // 处理从服务器接收到的图像数据
      const imageData = event.data;

      console.log('Received image data:', imageData);

      // 在网页上显示图像，这里假设服务器发送的是JPEG格式的图像
      const blob = new Blob([imageData], { type: 'image/jpeg' });
      const imageUrl = URL.createObjectURL(blob);

      // Assuming `dispElement` is a reference to your img element
      this.$refs.viewerElement.src = imageUrl;
    });
  },

  methods: {
    connectWebSocket() {
      this.socket = new WebSocket('ws://localhost:9002');
    },
    openCamera() {
      if (this.socket.readyState === WebSocket.OPEN) {
        this.socket.send('OpenCamera');
      } else {
        console.error('WebSocket connect is not opened.');
      }
    },
    startDetection() {
      // 开始目标检测的逻辑
    },
    stopDetection() {
      // 停止目标检测的逻辑
    },
    // 添加其他方法和逻辑
  },
};
</script>

<style scoped>
#app {
  text-align: center;
  height: 100vh;
  display: flex;
  flex-direction: column;
}

.top-section {
  flex: 1; /* You Only Look Once 字体的一半高度 */
  background-color: #ffffff;
  padding: 10px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.model-version {
  margin-right: 20px;
}

.model-label {
  font-size: 16px; /* 调整 label 的字体大小 */
}

.model-select {
  height: 30px; /* 调整选择框的高度 */
}

.title {
  font-size: 24px;
  margin-top: auto; /* You Only Look Once 垂直居中 */
}

.flower-font {
  font-family: 'Segoe Script', cursive; /* 使用花体字体 */
  margin-top: auto; /* 垂直居中 */
}

.bottom-section {
  flex: 19;/* 剩余的部分，不设置具体的flex值，占据剩下的高度 */
  background-color: #ffffff;
  padding: 20px;
  border-radius: 20px;
  overflow: hidden;
  position: relative;
}

.detection-box {
  height: 100%;
  position: relative;
  background-color: #f0f0f0; /* 框子内的浅灰色背景 */
  border: 2px solid #000000; /* 框子的黑色边框 */
  border-radius: 10px; /* 框子的边框弧度 */
  overflow: hidden;
}

.detected-object {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
}

.detection-image {
  width: 100%;
  height: 100%;
  object-fit: cover; /* 保持图片纵横比例并填满容器 */
}

.buttons {
  position: absolute;
  bottom: 0; /* 放在bottom-section的下边缘线上 */
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 10px;
}

button {
  padding: 10px;
  font-size: 16px;
}
</style>



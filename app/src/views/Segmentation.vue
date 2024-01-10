<template>
  <div class="button-list">
    <button
      v-for="(button, index) in buttons"
      :key="index"
      @click="activateWindow(index)"
      :class="{ active: activeIndex === index }"
    >
      {{ button }}
    </button>
  </div>
</template>

<script>
export default {
  data() {
    return {
      buttons: ['Button 1', 'Button 2', 'Button 3'], // Add more buttons as needed
      activeIndex: null,
    };
  },
  methods: {
    activateWindow(index) {
      this.activeIndex = index;
      this.$emit('activate-window', index);
    },
  },
};
</script>

<style scoped>
.button-list {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
}

button {
  padding: 10px;
  margin: 5px;
  border: none;
  cursor: pointer;
}

button.active {
  background-color: #e0e0e0;
}
</style>


  
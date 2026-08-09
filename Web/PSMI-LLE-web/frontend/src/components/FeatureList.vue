<template>
  <div class="feature-card">
    <h5>{{ title }}</h5>
    <div class="feature-list">
      <div v-for="feature in features" :key="feature.name" class="feature-row">
        <div class="feature-label">
          <span>{{ feature.name }}</span>
          <strong>{{ formatImportance(feature.importance) }}</strong>
        </div>
        <div class="feature-bar" :class="tone">
          <i :style="{ width: barWidth(feature.importance) }"></i>
        </div>
      </div>
      <p v-if="!features.length" class="empty-copy">No precomputed importance data found.</p>
    </div>
  </div>
</template>

<script>
export default {
  name: 'FeatureList',
  props: {
    title: String,
    features: {
      type: Array,
      default: () => [],
    },
    tone: {
      type: String,
      default: 'blue',
    },
  },
  computed: {
    maxImportance() {
      const values = this.features.map(item => Number(item.importance)).filter(Number.isFinite)
      return Math.max(...values, 1e-30)
    },
  },
  methods: {
    barWidth(value) {
      const num = Number(value)
      if (!Number.isFinite(num) || this.maxImportance <= 0) return '0%'
      return `${Math.max(4, (num / this.maxImportance) * 100)}%`
    },
    formatImportance(value) {
      const num = Number(value)
      if (!Number.isFinite(num)) return '-'
      if (num === 0) return '0'
      return num.toExponential(2)
    },
  },
}
</script>

<style scoped>
.feature-card {
  padding: 14px;
  border-radius: 14px;
  border: 1px solid rgba(133, 153, 181, 0.14);
  background: rgba(248, 251, 255, 0.96);
}

.feature-card h5 {
  margin: 0 0 10px;
  color: #172033;
  font-size: 13px;
}

.feature-list {
  display: grid;
  gap: 8px;
}

.feature-label {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 4px;
  font-size: 11px;
  color: #526071;
}

.feature-label strong {
  color: #172033;
  font-variant-numeric: tabular-nums;
}

.feature-bar {
  height: 7px;
  border-radius: 999px;
  background: rgba(47, 128, 237, 0.12);
  overflow: hidden;
}

.feature-bar.teal {
  background: rgba(20, 184, 166, 0.14);
}

.feature-bar i {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #2f80ed, #5aa9ff);
}

.feature-bar.teal i {
  background: linear-gradient(90deg, #14b8a6, #2dd4bf);
}

.empty-copy {
  margin: 0;
  color: #667085;
  font-size: 12px;
}
</style>

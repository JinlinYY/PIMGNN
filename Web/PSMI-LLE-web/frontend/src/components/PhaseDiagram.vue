<template>
  <el-card class="result-card" v-loading="loading" element-loading-text="Running model...">
    <template #header>
      <div class="card-header">
        <div>
          <span class="section-kicker">Prediction Results</span>
          <div class="title-row">
            <h3>Ternary Phase Diagram</h3>
            <span v-if="resultData?.temperature" class="temp-pill">
              T = {{ formatTemp(resultData.temperature) }} K
            </span>
            <span v-if="resultData?.pressure" class="temp-pill">
              P = {{ formatPressure(resultData.pressure) }} kPa
            </span>
          </div>
        </div>
        <div v-if="plotData" class="result-actions">
          <el-button size="small" @click="downloadPlot">Download PNG</el-button>
          <el-button size="small" @click="copyResult">Copy Results</el-button>
        </div>
      </div>
    </template>

    <div v-if="error" class="error-message">
      <el-alert :title="error" type="error" show-icon />
    </div>

    <div v-else-if="plotData" class="result-body">
      <div class="plot-container">
        <el-image
          :src="plotData"
          alt="Ternary diagram"
          fit="contain"
          :preview-src-list="[plotData]"
          preview-teleported
        />
      </div>
    </div>

    <div v-else class="empty-wrap">
      <el-empty description="Select a preset and click Predict" />
    </div>
  </el-card>
</template>

<script>
import { ElMessage } from 'element-plus'

export default {
  name: 'PhaseDiagram',
  props: {
    plotData: String,
    resultData: Object,
    error: String,
    loading: Boolean,
  },
  methods: {
    formatTemp(value) {
      const num = Number(value)
      return Number.isFinite(num) ? num.toFixed(2) : '-'
    },
    formatPressure(value) {
      const num = Number(value)
      return Number.isFinite(num) ? num.toFixed(3) : '-'
    },
    downloadPlot() {
      if (!this.plotData) return
      const link = document.createElement('a')
      link.href = this.plotData
      link.download = `lle-phase-diagram-${Date.now()}.png`
      link.click()
    },
    async copyResult() {
      if (!this.resultData) return
      try {
        await navigator.clipboard.writeText(JSON.stringify(this.resultData, null, 2))
        ElMessage.success('Prediction results copied')
      } catch (err) {
        ElMessage.warning('Copy failed')
      }
    },
  },
}
</script>

<style scoped>
.result-card {
  height: 100%;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.result-card :deep(.el-card__body) {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  padding: 10px 14px 12px;
}

.result-card :deep(.el-card__header) {
  padding: 10px 14px;
}

.card-header h3 {
  margin: 2px 0 0;
  font-size: 18px;
  color: #172033;
}

.section-kicker {
  color: #7c3aed;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.title-row {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.temp-pill {
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(47, 128, 237, 0.1);
  color: #245985;
  font-size: 12px;
  font-weight: 800;
}

.result-body {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
}

.plot-container {
  flex: 1;
  min-height: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid rgba(133, 153, 181, 0.18);
  border-radius: 12px;
  padding: 6px;
  background: #fbfdff;
}

.plot-container :deep(.el-image),
.plot-container :deep(img) {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.empty-wrap {
  flex: 1;
  min-height: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}

@media (max-width: 960px) {
  .result-card {
    height: auto;
  }
}
</style>

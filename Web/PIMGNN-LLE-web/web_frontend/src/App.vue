<template>
  <div id="app" class="app-shell">
    <header class="hero-section">
      <div class="hero-pattern" aria-hidden="true">
        <svg viewBox="0 0 200 120" class="hero-svg">
          <defs>
            <linearGradient id="heroGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stop-color="#2f80ed" stop-opacity="0.35" />
              <stop offset="100%" stop-color="#14b8a6" stop-opacity="0.25" />
            </linearGradient>
          </defs>
          <polygon points="100,8 148,36 148,84 100,112 52,84 52,36" fill="none" stroke="url(#heroGrad)" stroke-width="1.2" />
          <polygon points="100,28 128,44 128,76 100,92 72,76 72,44" fill="none" stroke="rgba(47,128,237,0.2)" stroke-width="0.8" />
          <circle cx="100" cy="8" r="3" fill="#2f80ed" opacity="0.7" />
          <circle cx="148" cy="36" r="3" fill="#14b8a6" opacity="0.7" />
          <circle cx="148" cy="84" r="3" fill="#7c3aed" opacity="0.6" />
          <circle cx="100" cy="112" r="3" fill="#2f80ed" opacity="0.6" />
          <circle cx="52" cy="84" r="3" fill="#14b8a6" opacity="0.6" />
          <circle cx="52" cy="36" r="3" fill="#7c3aed" opacity="0.6" />
        </svg>
      </div>
      <div class="hero-content">
        <div class="eyebrow">PIMGNN · Physics-Informed Molecular Graph Neural Network</div>
        <h1>Ternary LLE Prediction System</h1>
        <p class="hero-desc">
          Predict liquid-liquid phase equilibria from SMILES strings and generate ternary phase diagrams.
        </p>
        <div class="hero-chips">
          <span>SMILES Input</span>
          <span>Phase Diagram</span>
          <span>Molecular Analysis</span>
        </div>
      </div>
    </header>

    <el-alert
      v-if="backendOnline === false"
      class="status-alert"
      title="Backend service is unavailable. Please make sure the FastAPI server is running on port 8000."
      type="warning"
      show-icon
      :closable="false"
    />

    <main class="workspace">
      <el-row class="workspace-row" :gutter="16">
        <el-col class="workspace-col" :xs="24" :lg="8">
          <SmilesInput @predict="handlePredict" :loading="loading" />
        </el-col>
        <el-col class="workspace-col" :xs="24" :lg="16">
          <PhaseDiagram
            :plotData="plotData"
            :resultData="resultData"
            :error="error"
            :loading="loading"
          />
        </el-col>
      </el-row>
    </main>

    <AnalysisSection
      :resultData="resultData"
      :explain="explainData"
      :explainLoading="explainLoading"
    />
  </div>
</template>

<script>
import { ElMessage } from 'element-plus'
import SmilesInput from './components/SmilesInput.vue'
import PhaseDiagram from './components/PhaseDiagram.vue'
import AnalysisSection from './components/AnalysisSection.vue'

export default {
  name: 'App',
  components: { SmilesInput, PhaseDiagram, AnalysisSection },
  data() {
    return {
      loading: false,
      plotData: null,
      resultData: null,
      error: null,
      backendOnline: null,
      explainData: null,
      explainLoading: false,
    }
  },
  mounted() {
    this.checkHealth()
    this.loadExplainability()
  },
  methods: {
    async checkHealth() {
      try {
        const response = await this.$api.health()
        this.backendOnline = response.data.status === 'healthy' || response.data.model_loaded
      } catch (err) {
        this.backendOnline = false
      }
    },
    async loadExplainability() {
      this.explainLoading = true
      try {
        const response = await this.$api.loadExplainability()
        this.explainData = response.data
      } catch (err) {
        console.error(err)
      } finally {
        this.explainLoading = false
      }
    },
    formatError(err) {
      const detail = err?.response?.data?.detail
      if (Array.isArray(detail)) return detail.map(item => item.msg).join('; ')
      if (detail) return detail
      if (err?.response?.data?.message) return err.response.data.message
      if (err?.code === 'ECONNABORTED') return 'Request timed out.'
      return 'Network error or backend unavailable.'
    },
    async handlePredict(formData) {
      this.loading = true
      this.error = null
      this.plotData = null
      this.resultData = null
      try {
        const response = await this.$api.predict(formData)
        if (response.data.success) {
          this.plotData = response.data.plot_base64
          this.resultData = response.data.data
          this.backendOnline = true
          ElMessage.success('Prediction completed')
        } else {
          this.error = response.data.message
          ElMessage.error(response.data.message || 'Prediction failed')
        }
      } catch (err) {
        this.error = this.formatError(err)
        if (!err?.response) this.backendOnline = false
        ElMessage.error(this.error)
      } finally {
        this.loading = false
      }
    },
  },
}
</script>

<style>
* {
  box-sizing: border-box;
}

html,
body {
  margin: 0;
}

body {
  background:
    radial-gradient(circle at 10% -5%, rgba(64, 158, 255, 0.18), transparent 34%),
    radial-gradient(circle at 92% 6%, rgba(20, 184, 166, 0.14), transparent 30%),
    linear-gradient(135deg, #f7fbff 0%, #edf4fb 48%, #f9fbff 100%);
  color: #172033;
}

#app {
  font-family: Inter, "Microsoft YaHei", "PingFang SC", Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}

.app-shell {
  width: 100%;
  max-width: 1560px;
  margin: 0 auto;
  padding: 8px 12px 20px;
}

.hero-section {
  position: relative;
  overflow: hidden;
  margin-bottom: 8px;
  padding: 14px 22px;
  border: 1px solid rgba(133, 153, 181, 0.2);
  border-radius: 18px;
  background:
    linear-gradient(115deg, rgba(255, 255, 255, 0.95) 0%, rgba(240, 248, 255, 0.88) 55%, rgba(235, 245, 255, 0.82) 100%);
  box-shadow: 0 10px 28px rgba(32, 52, 84, 0.08);
}

.hero-section::before {
  content: "";
  position: absolute;
  inset: 0;
  background-image:
    radial-gradient(circle at 20% 30%, rgba(47, 128, 237, 0.06) 0 1px, transparent 1px),
    radial-gradient(circle at 60% 70%, rgba(20, 184, 166, 0.05) 0 1px, transparent 1px);
  background-size: 24px 24px, 32px 32px;
  pointer-events: none;
}

.hero-pattern {
  position: absolute;
  top: 50%;
  right: 28px;
  transform: translateY(-50%);
  width: 172px;
  height: 102px;
  opacity: 0.9;
  pointer-events: none;
}

.hero-svg {
  width: 100%;
  height: 100%;
}

.hero-content {
  position: relative;
  z-index: 1;
  max-width: 1380px;
}

.eyebrow {
  color: #2f80ed;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.hero-content h1 {
  margin: 6px 0 0;
  color: #0d1b34;
  font-size: 30px;
  line-height: 1.14;
  letter-spacing: -0.02em;
}

.hero-desc {
  margin: 8px 0 0;
  color: #5a6678;
  font-size: 14px;
  line-height: 1.5;
  max-width: 720px;
}

.hero-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
}

.hero-chips span {
  border: 1px solid rgba(47, 128, 237, 0.16);
  border-radius: 999px;
  padding: 7px 14px;
  background: rgba(255, 255, 255, 0.8);
  color: #265985;
  font-size: 12px;
  font-weight: 700;
}

.status-alert {
  margin-bottom: 10px;
}

/* 固定布局：左右栏统一 780px，底部对齐 */
.workspace {
  height: 780px;
  min-height: 780px;
  max-height: 780px;
  margin-bottom: 0;
}

.workspace-row {
  height: 780px;
  align-items: stretch;
}

.workspace-col {
  height: 780px;
  display: flex;
}

.workspace-col > * {
  flex: 1;
  width: 100%;
  height: 100%;
  min-height: 0;
}

.el-card {
  border: 1px solid rgba(133, 153, 181, 0.18);
  border-radius: 16px;
  box-shadow: 0 10px 28px rgba(32, 52, 84, 0.07);
  background: rgba(255, 255, 255, 0.92);
}

.el-card__header {
  padding: 11px 16px;
}

.el-card__body {
  padding: 12px 16px;
}

@media (max-width: 960px) {
  html,
  body {
    min-width: 0;
  }

  .app-shell {
    width: calc(100% - 24px);
    max-width: 100%;
  }

  .workspace,
  .workspace-row,
  .workspace-col {
    height: auto;
    min-height: auto;
    max-height: none;
  }

  .workspace-col {
    height: auto;
    margin-bottom: 14px;
  }

  .workspace-col > * {
    flex: none;
    height: auto;
    min-height: auto;
  }

  .hero-content {
    max-width: 100%;
  }

  .hero-pattern {
    display: none;
  }
}
</style>

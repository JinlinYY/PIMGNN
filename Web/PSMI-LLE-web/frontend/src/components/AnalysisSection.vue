<template>
  <section v-if="resultData" class="analysis-section">
    <div class="analysis-head">
      <span class="section-kicker">Detailed Analysis</span>
      <h2>Phase Composition · Molecules · Model Insights</h2>
    </div>

    <div class="phase-row">
      <div class="phase-card extract-card">
        <div class="phase-card-head">
          <span class="phase-badge extract">E</span>
          <div>
            <h4>Extract Phase</h4>
            <p>Equilibrium extract at t = 0.5</p>
          </div>
        </div>
        <div class="metric-list">
          <div v-for="item in extractMetrics" :key="item.label" class="metric-row">
            <div class="metric-label">
              <span>{{ item.label }}</span>
              <strong>{{ item.value }}</strong>
            </div>
            <div class="metric-bar"><i :style="{ width: item.percent + '%' }"></i></div>
          </div>
        </div>
      </div>

      <div class="phase-card raffinate-card">
        <div class="phase-card-head">
          <span class="phase-badge raffinate">R</span>
          <div>
            <h4>Raffinate Phase</h4>
            <p>Equilibrium raffinate at t = 0.5</p>
          </div>
        </div>
        <div class="metric-list">
          <div v-for="item in raffinateMetrics" :key="item.label" class="metric-row">
            <div class="metric-label">
              <span>{{ item.label }}</span>
              <strong>{{ item.value }}</strong>
            </div>
            <div class="metric-bar raffinate"><i :style="{ width: item.percent + '%' }"></i></div>
          </div>
        </div>
      </div>
    </div>

    <div class="molecule-row">
      <MoleculeCard
        v-for="component in resultData.components || []"
        :key="component.index"
        :component="component"
      />
    </div>

    <div class="table-panel">
      <div class="panel-title">Tie-line Compositions</div>
      <el-table :data="tieLineRows" border size="small" class="tie-line-table">
        <el-table-column prop="t" label="t" width="72" />
        <el-table-column label="Extract x1" prop="ex1" />
        <el-table-column label="Extract x2" prop="ex2" />
        <el-table-column label="Extract x3" prop="ex3" />
        <el-table-column label="Raffinate x1" prop="rx1" />
        <el-table-column label="Raffinate x2" prop="rx2" />
        <el-table-column label="Raffinate x3" prop="rx3" />
      </el-table>
    </div>

    <ExplainabilityPanel :explain="explain" :loading="explainLoading" embedded />
  </section>
</template>

<script>
import MoleculeCard from './MoleculeCard.vue'
import ExplainabilityPanel from './ExplainabilityPanel.vue'

export default {
  name: 'AnalysisSection',
  components: { MoleculeCard, ExplainabilityPanel },
  props: {
    resultData: Object,
    explain: Object,
    explainLoading: Boolean,
  },
  computed: {
    componentNames() {
      return this.resultData?.names || ['Component 1', 'Component 2', 'Component 3']
    },
    extractMetrics() {
      return this.buildMetrics(this.resultData?.e_phase, this.componentNames)
    },
    raffinateMetrics() {
      return this.buildMetrics(this.resultData?.r_phase, this.componentNames)
    },
    tieLineRows() {
      return (this.resultData?.tie_lines || []).map(line => ({
        t: this.formatValue(line.t, 3),
        ex1: this.formatValue(line.extract?.component1),
        ex2: this.formatValue(line.extract?.component2),
        ex3: this.formatValue(line.extract?.component3),
        rx1: this.formatValue(line.raffinate?.component1),
        rx2: this.formatValue(line.raffinate?.component2),
        rx3: this.formatValue(line.raffinate?.component3),
      }))
    },
  },
  methods: {
    buildMetrics(phase, names) {
      if (!phase) return []
      const values = [Number(phase.component1), Number(phase.component2), Number(phase.component3)]
      return values.map((value, index) => {
        const safe = Number.isFinite(value) ? value : 0
        return {
          label: names[index] || `Component ${index + 1}`,
          value: safe.toFixed(6),
          percent: Math.max(0, Math.min(100, safe * 100)),
        }
      })
    },
    formatValue(value, digits = 4) {
      const num = Number(value)
      return Number.isFinite(num) ? num.toFixed(digits) : '-'
    },
  },
}
</script>

<style scoped>
.analysis-section {
  margin-top: 16px;
  padding: 18px 20px 20px;
  border-radius: 16px;
  border: 1px solid rgba(133, 153, 181, 0.18);
  background: rgba(255, 255, 255, 0.92);
  box-shadow: 0 10px 28px rgba(32, 52, 84, 0.07);
}

.analysis-head h2 {
  margin: 4px 0 0;
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

.phase-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
  margin-top: 14px;
}

.phase-card {
  padding: 14px 16px;
  border-radius: 14px;
  border: 1px solid rgba(133, 153, 181, 0.14);
}

.extract-card {
  background: linear-gradient(180deg, rgba(47, 128, 237, 0.07), rgba(255, 255, 255, 0.96));
}

.raffinate-card {
  background: linear-gradient(180deg, rgba(245, 158, 11, 0.08), rgba(255, 255, 255, 0.96));
}

.phase-card-head {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  margin-bottom: 12px;
}

.phase-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border-radius: 10px;
  color: #fff;
  font-size: 13px;
  font-weight: 800;
  flex-shrink: 0;
}

.phase-badge.extract {
  background: linear-gradient(135deg, #2f80ed, #5aa9ff);
}

.phase-badge.raffinate {
  background: linear-gradient(135deg, #f59e0b, #fbbf24);
}

.phase-card-head h4 {
  margin: 0;
  font-size: 15px;
  color: #172033;
}

.phase-card-head p {
  margin: 3px 0 0;
  font-size: 12px;
  color: #667085;
}

.metric-list {
  display: grid;
  gap: 10px;
}

.metric-label {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 4px;
  font-size: 12px;
  color: #526071;
}

.metric-label strong {
  color: #172033;
  font-family: Consolas, monospace;
  font-variant-numeric: tabular-nums;
}

.metric-bar {
  height: 7px;
  border-radius: 999px;
  background: rgba(47, 128, 237, 0.12);
  overflow: hidden;
}

.metric-bar.raffinate {
  background: rgba(245, 158, 11, 0.14);
}

.metric-bar i {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #2f80ed, #14b8a6);
}

.metric-bar.raffinate i {
  background: linear-gradient(90deg, #f59e0b, #fb923c);
}

.molecule-row {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  margin-top: 16px;
}

.table-panel {
  margin-top: 16px;
}

.panel-title {
  margin-bottom: 10px;
  font-size: 14px;
  font-weight: 800;
  color: #172033;
}

.tie-line-table :deep(.el-table__cell) {
  font-variant-numeric: tabular-nums;
}

@media (max-width: 1100px) {
  .phase-row,
  .molecule-row {
    grid-template-columns: 1fr;
  }
}
</style>

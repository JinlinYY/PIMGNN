<template>
  <section class="explain-panel">
    <div class="explain-head">
      <div>
        <span class="section-kicker">Mechanistic View</span>
        <h4>What the Model Looks At</h4>
      </div>
      <el-button v-if="!embedded" link type="primary" @click="expanded = !expanded">
        {{ expanded ? 'Collapse' : 'Expand' }}
      </el-button>
    </div>

    <div v-if="loading" class="explain-loading">Loading model insights...</div>
    <div v-else-if="!explain" class="explain-loading">Model insights unavailable.</div>
    <div v-else-if="expanded" class="explain-grid">
      <div class="notes-card">
        <h5>Physics-informed interpretation</h5>
        <ul>
          <li v-for="note in explain.mechanism_notes" :key="note">{{ note }}</li>
        </ul>
      </div>
      <FeatureList title="Global molecular drivers" :features="explain.global_features" tone="blue" />
      <FeatureList title="Mixture compatibility signals" :features="explain.mixture_features" tone="teal" />
    </div>
  </section>
</template>

<script>
import FeatureList from './FeatureList.vue'

export default {
  name: 'ExplainabilityPanel',
  components: { FeatureList },
  props: {
    explain: {
      type: Object,
      default: null,
    },
    loading: {
      type: Boolean,
      default: false,
    },
    embedded: {
      type: Boolean,
      default: false,
    },
  },
  data() {
    return {
      expanded: true,
    }
  },
}
</script>

<style scoped>
.explain-panel {
  flex-shrink: 0;
  padding: 12px 16px;
  border-radius: 16px;
  border: 1px solid rgba(133, 153, 181, 0.16);
  background: rgba(255, 255, 255, 0.88);
  box-shadow: 0 8px 24px rgba(32, 52, 84, 0.06);
}

.explain-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.section-kicker {
  color: #7c3aed;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.explain-head h4 {
  margin: 2px 0 0;
  color: #172033;
  font-size: 16px;
}

.explain-loading {
  margin-top: 10px;
  color: #667085;
  font-size: 13px;
}

.explain-grid {
  display: grid;
  grid-template-columns: 1.1fr 1fr 1fr;
  gap: 12px;
  margin-top: 12px;
}

.notes-card {
  padding: 14px;
  border-radius: 14px;
  background: linear-gradient(180deg, rgba(124, 58, 237, 0.08), rgba(255, 255, 255, 0.96));
  border: 1px solid rgba(124, 58, 237, 0.12);
}

.notes-card h5,
.notes-card li {
  color: #344054;
}

.notes-card h5 {
  margin: 0 0 8px;
  font-size: 13px;
}

.notes-card ul {
  margin: 0;
  padding-left: 18px;
}

.notes-card li {
  font-size: 12px;
  line-height: 1.55;
}

.notes-card li + li {
  margin-top: 6px;
}

@media (max-width: 1200px) {
  .explain-grid {
    grid-template-columns: 1fr;
  }
}
</style>

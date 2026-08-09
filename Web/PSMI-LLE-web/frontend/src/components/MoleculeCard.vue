<template>
  <article class="molecule-card" :class="{ compact }">
    <div class="molecule-head">
      <span class="molecule-kicker">Comp {{ component.index }}</span>
      <h4>{{ component.label }}</h4>
      <p class="formula">{{ component.formula || '—' }}</p>
    </div>
    <div class="molecule-svg" v-if="component.svg" v-html="component.svg"></div>
    <dl v-if="!compact" class="descriptor-grid">
      <div v-for="item in topDescriptors" :key="item.name">
        <dt>{{ item.name }}</dt>
        <dd>{{ formatDescriptor(item) }}</dd>
      </div>
    </dl>
    <code v-if="!compact" class="canonical-smiles">{{ component.canonical_smiles }}</code>
  </article>
</template>

<script>
export default {
  name: 'MoleculeCard',
  props: {
    component: {
      type: Object,
      required: true,
    },
    compact: {
      type: Boolean,
      default: false,
    },
  },
  computed: {
    topDescriptors() {
      return (this.component.descriptors || []).slice(0, 4)
    },
  },
  methods: {
    formatDescriptor(item) {
      const value = Number(item.value)
      const text = Number.isFinite(value) ? value.toFixed(2) : '-'
      return item.unit ? `${text} ${item.unit}` : text
    },
  },
}
</script>

<style scoped>
.molecule-card {
  display: grid;
  gap: 10px;
  padding: 14px;
  border-radius: 14px;
  border: 1px solid rgba(133, 153, 181, 0.14);
  background: rgba(255, 255, 255, 0.96);
}

.molecule-card:not(.compact) {
  padding: 16px;
}

.molecule-head h4 {
  margin: 3px 0 0;
  color: #172033;
  font-size: 14px;
  line-height: 1.3;
}

.formula {
  margin: 3px 0 0;
  color: #667085;
  font-size: 12px;
  font-weight: 700;
}

.molecule-svg {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 120px;
  border-radius: 10px;
  background: rgba(248, 251, 255, 0.92);
  border: 1px solid rgba(133, 153, 181, 0.1);
}

.molecule-card.compact .molecule-svg {
  min-height: 58px;
}

.molecule-svg :deep(svg) {
  width: 100%;
  max-height: 120px;
}

.molecule-kicker {
  color: #2f80ed;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.descriptor-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
  margin: 0;
}

.descriptor-grid div {
  padding: 8px 10px;
  border-radius: 8px;
  background: rgba(243, 247, 251, 0.92);
}

.descriptor-grid dt {
  margin: 0;
  color: #667085;
  font-size: 10px;
  font-weight: 700;
}

.descriptor-grid dd {
  margin: 2px 0 0;
  color: #172033;
  font-size: 12px;
  font-weight: 700;
}

.canonical-smiles {
  display: block;
  padding: 8px 10px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.92);
  border: 1px solid rgba(133, 153, 181, 0.1);
  color: #344054;
  font-size: 11px;
  line-height: 1.45;
  overflow-wrap: anywhere;
  font-family: Consolas, monospace;
}
</style>

<template>

  <el-card class="input-card">

    <template #header>

      <div class="card-header">

        <div>

          <span class="section-kicker">Input Parameters</span>

          <h3>Ternary System Setup</h3>

        </div>

      </div>

    </template>



    <div class="input-stack">

      <div class="preset-panel">

        <div class="preset-title">Literature Benchmarks</div>

        <el-select

          v-model="selectedPreset"

          class="preset-select"

          placeholder="Select a benchmark ternary system"

          teleported

          @change="applyPreset"

        >

          <el-option

            v-for="item in presets"

            :key="item.id"

            :label="item.shortLabel"

            :value="item.id"

          />

        </el-select>

        <p class="preset-subtitle">{{ activePresetDescription }}</p>

      </div>



      <el-form :model="form" :rules="rules" ref="formRef" label-position="top" class="smiles-form">

        <el-form-item

          v-for="index in 3"

          :key="index"

          :label="form[`name${index}`] || `Component ${index}`"

          :prop="`smiles${index}`"

        >

          <el-input

            v-model="form[`smiles${index}`]"

            placeholder="SMILES string"

            clearable

            class="smiles-input"

          />

        </el-form-item>



        <div class="inline-fields">

          <el-form-item label="Temperature (K)" prop="temperature" class="temp-field">

            <el-input-number

              v-model="form.temperature"

              :min="0.1"

              :step="0.1"

              :precision="1"

              controls-position="right"

              class="smiles-number"

            />

          </el-form-item>

          <el-form-item label="Tie-lines" prop="tie_lines_count" class="slider-field">

            <el-slider v-model="form.tie_lines_count" :min="1" :max="100" :step="1" show-input />

          </el-form-item>

        </div>

      </el-form>



      <div class="system-preview">
        <div class="preview-header">
          <span class="preview-kicker">System Preview</span>
          <div class="preview-meta">
            <span>T = {{ previewTemperature }} K</span>
            <span>{{ form.tie_lines_count }} tie-lines</span>
          </div>
        </div>

        <div class="molecule-grid">
          <div
            v-for="index in 3"
            :key="index"
            class="molecule-preview-card"
            :class="`tone-${index}`"
          >
            <div class="molecule-preview-text">
              <div class="molecule-name-row">
                <span class="component-dot" :class="`dot-${index}`" />
                <strong>{{ form[`name${index}`] || `Component ${index}` }}</strong>
              </div>
              <em>{{ moleculePreviews[index]?.formula || '—' }}</em>
            </div>
            <div
              class="molecule-preview-svg"
              v-loading="moleculePreviews[index]?.loading"
              element-loading-background="rgba(255,255,255,0.65)"
            >
              <div
                v-if="moleculePreviews[index]?.svg"
                class="svg-wrap"
                v-html="moleculePreviews[index].svg"
              />
              <span v-else-if="moleculePreviews[index]?.error" class="svg-placeholder invalid">!</span>
              <span v-else class="svg-placeholder">···</span>
            </div>
          </div>
        </div>
      </div>

    </div>



    <div class="form-actions">

      <el-button

        type="primary"

        class="action-btn predict-btn"

        @click="submitForm"

        :loading="loading"

        :disabled="!isFormValid"

      >

        Predict

      </el-button>

      <el-button class="action-btn reset-btn" @click="resetForm" :disabled="loading">

        Reset

      </el-button>

    </div>

  </el-card>

</template>



<script>

const defaultForm = {

  name1: '',

  name2: '',

  name3: '',

  smiles1: '',

  smiles2: '',

  smiles3: '',

  temperature: null,

  tie_lines_count: 14,

}



export default {

  name: 'SmilesInput',

  props: { loading: Boolean },

  data() {

    return {

      selectedPreset: 'system22',

      presets: [

        {

          id: 'sulfolane',

          shortLabel: 'Case I — n-Heptane / Toluene / Sulfolane',

          description: 'Paper Case I: aromatic extraction · n-Heptane + Toluene + Sulfolane · T = 348.15 K',

          form: {

            name1: 'n-Heptane', name2: 'Toluene', name3: 'Sulfolane',

            smiles1: 'CCCCCCC', smiles2: 'Cc1ccccc1', smiles3: 'O=S1(=O)CCCC1',

            temperature: 348.15, tie_lines_count: 14,

          },

        },

        {

          id: 'dem',

          shortLabel: 'Case II — Water / DEM / p-Xylene',

          description: 'Paper Case II: DEM recovery · Water + Diethoxymethane + p-Xylene · T = 303.15 K',

          form: {

            name1: 'Water', name2: 'Diethoxymethane', name3: 'p-Xylene',

            smiles1: 'O', smiles2: 'CCOCOCC', smiles3: 'Cc1ccc(C)cc1',

            temperature: 303.15, tie_lines_count: 14,

          },

        },

        {

          id: 'system22',

          shortLabel: 'Benchmark 22 — IL / Benzene / n-Decane',

          description: 'Benchmark ternary LLE · [C4MPyr][NTf2] + Benzene + n-Decane · T = 298.15 K',

          form: {

            name1: '[C4MPyr][NTf2]', name2: 'Benzene', name3: 'n-Decane',

            smiles1: 'CCCC[N+]1(C)CCCC1.O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F',

            smiles2: 'c1ccccc1', smiles3: 'CCCCCCCCCC',

            temperature: 298.15, tie_lines_count: 14,

          },

        },

      ],

      form: { ...defaultForm },

      moleculePreviews: {
        1: { svg: '', formula: '', loading: false, error: '' },
        2: { svg: '', formula: '', loading: false, error: '' },
        3: { svg: '', formula: '', loading: false, error: '' },
      },

      previewTimer: null,

      rules: {

        smiles1: [{ required: true, message: 'Required', trigger: 'blur' }],

        smiles2: [{ required: true, message: 'Required', trigger: 'blur' }],

        smiles3: [{ required: true, message: 'Required', trigger: 'blur' }],

        temperature: [{ required: true, message: 'Required', trigger: 'change' }],

        tie_lines_count: [{ required: true, message: 'Required', trigger: 'change' }],

      },

    }

  },

  computed: {

    isFormValid() {

      return this.form.smiles1 && this.form.smiles2 && this.form.smiles3 && this.form.temperature > 0

    },

    activePresetDescription() {

      const item = this.presets.find(p => p.id === this.selectedPreset)

      return item ? item.description : 'Select a literature benchmark to load SMILES and conditions.'

    },

    previewTemperature() {

      const value = Number(this.form.temperature)

      return Number.isFinite(value) ? value.toFixed(2) : '—'

    },

    smilesWatchKey() {

      return [this.form.smiles1, this.form.smiles2, this.form.smiles3].join('|')

    },

  },

  watch: {

    smilesWatchKey() {

      this.schedulePreviewRefresh()

    },

  },

  mounted() {

    this.applyPreset('system22')

  },

  beforeUnmount() {

    if (this.previewTimer) clearTimeout(this.previewTimer)

  },

  methods: {

    schedulePreviewRefresh() {

      if (this.previewTimer) clearTimeout(this.previewTimer)

      this.previewTimer = setTimeout(() => this.refreshMoleculePreviews(), 350)

    },

    async refreshMoleculePreviews() {

      for (let index = 1; index <= 3; index += 1) {

        const smiles = String(this.form[`smiles${index}`] || '').trim()

        if (!smiles) {

          this.moleculePreviews[index] = { svg: '', formula: '', loading: false, error: '' }

          continue

        }

        this.moleculePreviews[index] = {
          ...this.moleculePreviews[index],
          loading: true,
          error: '',
        }

        try {

          const response = await this.$api.validateSmiles(smiles)

          this.moleculePreviews[index] = {
            svg: response.data.svg,
            formula: response.data.formula,
            loading: false,
            error: '',
          }

        } catch (err) {

          this.moleculePreviews[index] = {
            svg: '',
            formula: '',
            loading: false,
            error: 'Invalid SMILES',
          }

        }

      }

    },

    applyPreset(id) {

      const preset = this.presets.find(item => item.id === id)

      if (!preset) return

      this.selectedPreset = id

      this.form = { ...preset.form }

      this.$nextTick(() => {
        this.$refs.formRef?.clearValidate()
        this.schedulePreviewRefresh()
      })

    },

    resetForm() {

      this.selectedPreset = ''

      this.form = { ...defaultForm }

      this.$nextTick(() => {
        this.$refs.formRef?.clearValidate()
        this.schedulePreviewRefresh()
      })

    },

    submitForm() {

      this.$refs.formRef.validate((valid) => {

        if (valid) this.$emit('predict', { ...this.form })

      })

    },

  },

}

</script>



<style scoped>

.input-card {

  height: 100%;

  display: flex;

  flex-direction: column;

  min-height: 0;

}



.input-card :deep(.el-card__body) {

  flex: 1;

  min-height: 0;

  display: flex;

  flex-direction: column;

  padding: 10px 14px 12px;

}



.input-card :deep(.el-card__header) {

  padding: 10px 14px;

}



.card-header h3 {

  margin: 2px 0 0;

  font-size: 18px;

  color: #172033;

}



.section-kicker {

  color: #0f8f8a;

  font-size: 11px;

  font-weight: 800;

  letter-spacing: 0.12em;

  text-transform: uppercase;

}



.input-stack {

  flex: 1;

  min-height: 0;

  display: flex;

  flex-direction: column;

  gap: 6px;

  overflow-y: auto;

}



.preset-panel {

  flex-shrink: 0;

  padding: 8px 10px;

  border: 1px solid rgba(20, 184, 166, 0.16);

  border-radius: 12px;

  background: linear-gradient(180deg, #f8fffd, #f1f8ff);

}



.preset-title {

  margin-bottom: 6px;

  font-size: 12px;

  font-weight: 800;

  color: #172033;

}



.preset-select {

  width: 100%;

}



.preset-select :deep(.el-input__wrapper) {

  min-height: 34px;

  border-radius: 10px;

  box-shadow: 0 0 0 1px rgba(47, 128, 237, 0.18) inset;

}



.preset-subtitle {

  margin: 6px 0 0;

  font-size: 11px;

  color: #667085;

  line-height: 1.35;

}



.smiles-form {

  flex-shrink: 0;

}



.smiles-form :deep(.el-form-item) {

  margin-bottom: 9px;

}



.smiles-form :deep(.el-form-item__label) {

  font-size: 12px;

  font-weight: 700;

  color: #344054;

  padding-bottom: 0;

  margin-bottom: 2px;

}



.smiles-input :deep(.el-input__wrapper) {

  font-family: Consolas, monospace;

  font-size: 12px;

  min-height: 34px;

}



.smiles-number :deep(.el-input__wrapper) {

  min-height: 34px;

}



.inline-fields {

  display: grid;

  grid-template-columns: 132px 1fr;

  gap: 8px;

  margin-bottom: 0;

}



.slider-field :deep(.el-form-item__content) {

  padding-top: 2px;

}



.system-preview {

  flex: 1;

  min-height: 0;

  display: flex;

  flex-direction: column;

  gap: 8px;

  padding: 10px;

  border: 1px dashed rgba(47, 128, 237, 0.22);

  border-radius: 12px;

  background:

    linear-gradient(180deg, rgba(248, 252, 255, 0.95), rgba(236, 245, 255, 0.72));

}



.preview-header {

  flex-shrink: 0;

  display: flex;

  align-items: center;

  justify-content: space-between;

  gap: 8px;

}



.preview-kicker {

  color: #2f80ed;

  font-size: 11px;

  font-weight: 800;

  letter-spacing: 0.1em;

  text-transform: uppercase;

}



.preview-meta {

  display: flex;

  flex-wrap: wrap;

  justify-content: flex-end;

  gap: 6px;

}



.preview-meta span {

  border: 1px solid rgba(47, 128, 237, 0.16);

  border-radius: 999px;

  padding: 3px 10px;

  background: rgba(255, 255, 255, 0.82);

  color: #265985;

  font-size: 10px;

  font-weight: 700;

}



.molecule-grid {

  flex: 1;

  min-height: 0;

  display: flex;

  flex-direction: column;

  gap: 6px;

  overflow: hidden;

}



.molecule-preview-card {

  flex: 1 1 0;

  min-height: 0;

  display: grid;

  grid-template-columns: minmax(0, 1fr) 64px;

  align-items: center;

  gap: 8px;

  padding: 6px 8px;

  border-radius: 10px;

  border: 1px solid rgba(133, 153, 181, 0.14);

  background: rgba(255, 255, 255, 0.88);

  overflow: hidden;

}



.molecule-preview-card.tone-1 {
  border-left: 3px solid #2f80ed;
}

.molecule-preview-card.tone-2 {
  border-left: 3px solid #14b8a6;
}

.molecule-preview-card.tone-3 {
  border-left: 3px solid #7c3aed;
}



.molecule-name-row {

  display: flex;

  align-items: center;

  gap: 6px;

  min-width: 0;

}



.component-dot {

  flex-shrink: 0;

  width: 7px;

  height: 7px;

  border-radius: 50%;

}



.dot-1 { background: #2f80ed; }

.dot-2 { background: #14b8a6; }

.dot-3 { background: #7c3aed; }



.molecule-preview-text {

  min-width: 0;

  display: flex;

  flex-direction: column;

  gap: 2px;

  overflow: hidden;

}



.molecule-preview-text strong {

  color: #172033;

  font-size: 11px;

  line-height: 1.25;

  white-space: nowrap;

  overflow: hidden;

  text-overflow: ellipsis;

}



.molecule-preview-text em {

  color: #667085;

  font-size: 10px;

  font-style: normal;

  font-weight: 700;

  letter-spacing: 0.02em;

  white-space: nowrap;

  overflow: hidden;

  text-overflow: ellipsis;

}



.molecule-preview-svg {

  width: 64px;

  height: 40px;

  flex-shrink: 0;

  display: flex;

  align-items: center;

  justify-content: center;

  border-radius: 6px;

  border: 1px solid rgba(133, 153, 181, 0.12);

  background: rgba(248, 251, 255, 0.95);

  overflow: hidden;

}



.molecule-preview-svg :deep(.el-loading-mask) {
  border-radius: 6px;
}



.molecule-preview-svg :deep(.el-loading-spinner) {
  margin-top: -12px;
}



.molecule-preview-svg :deep(.circular) {
  width: 20px;
  height: 20px;
}



.svg-wrap {

  width: 64px;

  height: 40px;

  display: flex;

  align-items: center;

  justify-content: center;

  overflow: hidden;

  pointer-events: none;

}



.svg-wrap :deep(svg) {

  width: 60px !important;

  height: 36px !important;

  max-width: 60px !important;

  max-height: 36px !important;

}



.svg-placeholder {

  color: #98a2b3;

  font-size: 11px;

  font-weight: 700;

  line-height: 1;

}



.svg-placeholder.invalid {

  color: #d14343;

}



.form-actions {

  flex-shrink: 0;

  display: flex;

  justify-content: center;

  align-items: center;

  gap: 12px;

  margin-top: 6px;

  padding-top: 8px;

  border-top: 1px solid rgba(133, 153, 181, 0.14);

}



.action-btn {

  min-width: 112px;

  height: 36px;

  border-radius: 10px;

  font-size: 13px;

  font-weight: 700;

  letter-spacing: 0.02em;

}



.predict-btn {

  border: 0;

  background: linear-gradient(135deg, #2f80ed 0%, #14b8a6 100%);

  box-shadow: 0 6px 16px rgba(47, 128, 237, 0.22);

}



.predict-btn:hover {

  background: linear-gradient(135deg, #2569c7 0%, #0fa396 100%);

}



.reset-btn {

  border: 1px solid rgba(133, 153, 181, 0.28);

  background: #fff;

  color: #475467;

}



.reset-btn:hover {

  border-color: rgba(47, 128, 237, 0.35);

  color: #2f80ed;

  background: #f8fbff;

}



.smiles-number {

  width: 100%;

}

@media (max-width: 960px) {
  .input-card {
    height: auto;
  }
}

</style>



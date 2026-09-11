<script setup lang="ts">
import { PipelineType } from "@/types/PipelineTypes";
import PvSelect from "@/components/common/pv-select.vue";
import PvSlider from "@/components/common/pv-slider.vue";
import PvSwitch from "@/components/common/pv-switch.vue";
import { computed } from "vue";
import { useStateStore } from "@/stores/StateStore";
import type { ActivePipelineSettings } from "@/types/PipelineTypes";
import { useCameraSettingsStore } from "@/stores/settings/CameraSettingsStore";
import { useDisplay } from "vuetify";

// TODO fix pipeline typing in order to fix this, the store settings call should be able to infer that only valid pipeline type settings are exposed based on pre-checks for the entire config section
// Defer reference to store access method
const currentPipelineSettings = computed<ActivePipelineSettings>(
  () => useCameraSettingsStore().currentPipelineSettings
);
const { mdAndDown } = useDisplay();
const interactiveCols = computed(() =>
  mdAndDown.value && (!useStateStore().sidebarFolded || useCameraSettingsStore().isDriverMode) ? 8 : 7
);
</script>

<template>
  <div v-if="currentPipelineSettings.pipelineType === PipelineType.AprilTag">
    <pv-select
      v-model="currentPipelineSettings.tagFamily"
      label="Target family"
      :items="['AprilTag 36h11 (6.5in)', 'AprilTag 16h5 (6in)']"
      :select-cols="interactiveCols"
      @update:modelValue="(value) => useCameraSettingsStore().changeCurrentPipelineSetting({ tagFamily: value }, false)"
    />
    <pv-slider
      v-model="currentPipelineSettings.decimate"
      :slider-cols="interactiveCols"
      label="Decimate"
      tooltip="Increases FPS at the expense of range by reducing image resolution initially"
      :min="1"
      :max="8"
      @update:modelValue="(value) => useCameraSettingsStore().changeCurrentPipelineSetting({ decimate: value }, false)"
    />
    <pv-slider
      v-model="currentPipelineSettings.blur"
      :slider-cols="interactiveCols"
      label="Blur"
      tooltip="Gaussian blur added to the image, high FPS cost for slightly decreased noise"
      :min="0"
      :max="5"
      :step="0.1"
      @update:modelValue="(value) => useCameraSettingsStore().changeCurrentPipelineSetting({ blur: value }, false)"
    />
    <pv-slider
      v-model="currentPipelineSettings.threads"
      :slider-cols="interactiveCols"
      label="Threads"
      tooltip="Number of threads spawned by the AprilTag detector"
      :min="1"
      :max="8"
      @update:modelValue="(value) => useCameraSettingsStore().changeCurrentPipelineSetting({ threads: value }, false)"
    />
    <pv-slider
      v-model="currentPipelineSettings.decisionMargin"
      :slider-cols="interactiveCols"
      label="Decision Margin Cutoff"
      tooltip="Tags with a 'margin' (decoding quality score) less than this wil be rejected. Increase this to reduce the number of false positive detections"
      :min="0"
      :max="250"
      @update:modelValue="
        (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ decisionMargin: value }, false)
      "
    />
    <pv-slider
      v-model="currentPipelineSettings.numIterations"
      :slider-cols="interactiveCols"
      label="Pose Estimation Iterations"
      tooltip="Number of iterations the pose estimation algorithm will run, 50-100 is a good starting point"
      :min="0"
      :max="500"
      @update:modelValue="
        (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ numIterations: value }, false)
      "
    />
    <pv-switch
      v-model="currentPipelineSettings.refineEdges"
      :switch-cols="interactiveCols"
      label="Refine Edges"
      tooltip="Further refines the AprilTag corner position initial estimate, suggested left on"
      @update:modelValue="
        (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ refineEdges: value }, false)
      "
    />
    <!-- Jetson: GPU detector and the far-field tiers -->
    <pv-switch
      v-model="currentPipelineSettings.gpuDetector"
      :switch-cols="interactiveCols"
      label="GPU detector"
      tooltip="Detect on the GPU (NVIDIA Jetson with photon-gpu installed). Quad finding runs at decimate 2 above 1024 px; corners are refined at full resolution. Falls back to the CPU detector if the library is not available."
      @update:modelValue="
        (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ gpuDetector: value }, false)
      "
    />
    <pv-switch
      v-model="currentPipelineSettings.farFieldEnabled"
      :switch-cols="interactiveCols"
      label="Far-field search"
      tooltip="Search a horizontal band of the frame at full resolution on a background thread, to find small (distant) tags the main detector misses. Found tags seed ROI tracking."
      @update:modelValue="
        (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ farFieldEnabled: value }, false)
      "
    />
    <template v-if="currentPipelineSettings.farFieldEnabled">
      <pv-slider
        v-model="currentPipelineSettings.farFieldRateHz"
        :slider-cols="interactiveCols"
        label="Far-field rate (Hz)"
        tooltip="How often the band is searched. Each search costs ~50-130 ms of one CPU core."
        :min="1"
        :max="22"
        :step="1"
        @update:modelValue="
          (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ farFieldRateHz: value }, false)
        "
      />
      <pv-slider
        v-model="currentPipelineSettings.farFieldUpsample"
        :slider-cols="interactiveCols"
        label="Far-field upsample"
        tooltip="Upsample the band before searching. 1.5 finds blurred tags about one size step smaller at 2.25x the cost."
        :min="1"
        :max="2"
        :step="0.5"
        @update:modelValue="
          (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ farFieldUpsample: value }, false)
        "
      />
      <pv-switch
        v-model="currentPipelineSettings.farFieldAutoBand"
        :switch-cols="interactiveCols"
        label="Band from mount pose"
        tooltip="Derive the band from the camera's calibration and its mount height/pitch (needs a calibration); otherwise use the fractions below."
        @update:modelValue="
          (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ farFieldAutoBand: value }, false)
        "
      />
      <template v-if="!currentPipelineSettings.farFieldAutoBand">
        <pv-slider
          v-model="currentPipelineSettings.farFieldBandTop"
          :slider-cols="interactiveCols"
          label="Band top"
          tooltip="Top of the band as a fraction of the image height"
          :min="0"
          :max="1"
          :step="0.01"
          @update:modelValue="
            (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ farFieldBandTop: value }, false)
          "
        />
        <pv-slider
          v-model="currentPipelineSettings.farFieldBandBottom"
          :slider-cols="interactiveCols"
          label="Band bottom"
          tooltip="Bottom of the band as a fraction of the image height"
          :min="0"
          :max="1"
          :step="0.01"
          @update:modelValue="
            (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ farFieldBandBottom: value }, false)
          "
        />
      </template>
      <template v-else>
        <pv-slider
          v-model="currentPipelineSettings.mountHeightMeters"
          :slider-cols="interactiveCols"
          label="Camera height (m)"
          tooltip="Camera height above the floor"
          :min="0"
          :max="2"
          :step="0.01"
          @update:modelValue="
            (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ mountHeightMeters: value }, false)
          "
        />
        <pv-slider
          v-model="currentPipelineSettings.mountPitchDegrees"
          :slider-cols="interactiveCols"
          label="Camera pitch (deg, up +)"
          tooltip="Camera pitch above horizontal"
          :min="-45"
          :max="45"
          :step="0.5"
          @update:modelValue="
            (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ mountPitchDegrees: value }, false)
          "
        />
        <pv-slider
          v-model="currentPipelineSettings.farFieldMinDistanceMeters"
          :slider-cols="interactiveCols"
          label="Far field starts at (m)"
          tooltip="Tags nearer than this are the main detector's job"
          :min="2"
          :max="15"
          :step="0.5"
          @update:modelValue="
            (value) =>
              useCameraSettingsStore().changeCurrentPipelineSetting({ farFieldMinDistanceMeters: value }, false)
          "
        />
      </template>
    </template>
    <pv-switch
      v-model="currentPipelineSettings.roiTrackEnabled"
      :switch-cols="interactiveCols"
      label="ROI tracking"
      tooltip="Every frame, re-detect at full resolution, in a small crop, tags the main detector did not see but that are known from the far-field search or earlier frames."
      @update:modelValue="
        (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ roiTrackEnabled: value }, false)
      "
    />
    <template v-if="currentPipelineSettings.roiTrackEnabled">
      <pv-slider
        v-model="currentPipelineSettings.roiUpsample"
        :slider-cols="interactiveCols"
        label="ROI upsample"
        tooltip="Upsample each ROI before re-detection"
        :min="1"
        :max="2"
        :step="0.5"
        @update:modelValue="
          (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ roiUpsample: value }, false)
        "
      />
      <pv-slider
        v-model="currentPipelineSettings.roiMaxCount"
        :slider-cols="interactiveCols"
        label="Max ROIs per frame"
        tooltip="Upper bound on ROI re-detections per frame (~2.6 ms each at full resolution)"
        :min="1"
        :max="16"
        :step="1"
        @update:modelValue="
          (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ roiMaxCount: value }, false)
        "
      />
    </template>
  </div>
</template>

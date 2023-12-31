# Reproducibility & Generation

Reproducibility settings are settings that change generation output. These changes can vary from small, to large, with small being a few lines look sharper

## Device

Changing the device to the correct one -- that being, your fastest available GPU -- can not only improve performance, but also change how the images look like. Something generated using DirectML on an AMD card won't EVER look the same as something generated with CUDA.

## Data type

Generally, changing data type to a lower precision (lower number) one, will improve performance, however, when taken to extreme degrees (volta doesn't have this implemented) image quality starts to get hammered. `16-bit float` or `16-bit bfloat` is generally the lowest people should need to go.

## Deterministic generation

PyTorch, and as such, Volta, is by design indeterministic, - that is, not 100% reproducible. This can raise a few issues: generations using the exact same parameters **MAY NOT** come out the same. Changing this to on, should fix these issues.

## SGM Noise Multiplier

SGM Noise multiplier changes how noise is calculated. This is only useful for reproducing already created images. From a more technical standpoint: this changes noising to mimic SDXL's noise creation. **Only useful on `SD1.x`.**

### On vs. off

<component v-if="dynamicComponent" :is="dynamicComponent">
<img
        slot="first"
        style="width: 100%"
        src="/static/settings/reproducibility/sgm_on.webp"
    />
<img
        slot="second"
        style="width: 100%"
        src="/static/settings/reproducibility/sgm_off.webp"
    />
</component>

## Quantization in KDiff samplers

Quantization in K-samplers helps the samplers to create more sharp and defined lines. This is another one of those _"small, but useful"_ changes.

### On vs. off

<component v-if="dynamicComponent" :is="dynamicComponent">
<img
        slot="first"
        style="width: 100%"
        src="/static/settings/reproducibility/quant_on.webp"
    />
<img
        slot="second"
        style="width: 100%"
        src="/static/settings/reproducibility/sgm_off.webp"
    />
</component>

## Generator

<script setup>
import { shallowRef } from 'vue'

let dynamicComponent = shallowRef(null)
if (!import.meta.env.SSR) {
    import('@img-comparison-slider/vue').then((module) => {
        dynamicComponent.value = module.ImgComparisonSlider
        console.log(dynamicComponent)
    });
}
</script>

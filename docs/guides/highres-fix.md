# Highres Fix

Method that allows you to generate high resolution images without duplicating subjects.

**Currently available on the `Text To Image` page.**

::: info
Generated using `UniPCMultistep (diffusers)`<br>
Hires scale `2x` in cases where hires is applied.<br>
Image Upscaler `RealESRGAN_x4plus_anime_6B`<br>
Latent Upscaler `Bislerp (Tortured, fast)`<br>
Steps: `30` base, `40` hires
Strength: `0.65`
:::

Usually, Latent mode tends to perform better than Image mode, but it will get blurry with strength under `0.6`.
It is usually further from the original image. See example below.

## Base VS Latent Upscale

<component v-if="dynamicComponent" :is="dynamicComponent">
<img
        slot="first"
        style="width: 100%"
        src="/static/getting-started/highres/without.webp"
    />
<img
        slot="second"
        style="width: 100%"
        src="/static/getting-started/highres/latent.webp"
    />
</component>

## Base VS Image Upscale

<component v-if="dynamicComponent" :is="dynamicComponent">
<img
        slot="first"
        style="width: 100%"
        src="/static/getting-started/highres/without.webp"
    />
<img
        slot="second"
        style="width: 100%"
        src="/static/getting-started/highres/image.webp"
    />
</component>

## Image Upscale VS Latent Upscale

<component v-if="dynamicComponent" :is="dynamicComponent">
<img
        slot="first"
        style="width: 100%"
        src="/static/getting-started/highres/image.webp"
    />
<img
        slot="second"
        style="width: 100%"
        src="/static/getting-started/highres/latent.webp"
    />
</component>

## Base VS High resolution base

Even though the model I use is trained on higher resolution images, it still duplicated the railings. This gets more noticeable with older models that were trained only on 512x512 images.

More recent models or SDXL might be totally fine with higher resolution images.

<ClientOnly>
<component v-if="dynamicComponent" :is="dynamicComponent">
<img
        slot="first"
        style="width: 100%"
        src="/static/getting-started/highres/without.webp"
    />
<img
        slot="second"
        style="width: 100%"
        src="/static/getting-started/highres/highres-base.webp"
    />
</component>
</ClientOnly>

## Workflow diagram

![Workflow diagram](/static/getting-started/highres/highres-diagram.webp)

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

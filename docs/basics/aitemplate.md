# AITemplate Acceleration

## Benefits

- About 1.7x faster generation

## Downsides

- No LoRA support
- No Textual Inversion support
- Locked down resolution (if you use static AITemplate)

## Acceleration

Make sure that you are one the correct page as can be seen in the image below.

![AITemplate Acceleration](../static/basics/aitemplate-accelerate.webp)

1. Width and Height - these cannot be changed after the model is compiled
2. Batch Size - this cannot be changed after the model is compiled as well - but Batch Count can - for this reason, I recommend setting it to 1
3. CPU Threads - Number of CPU threads that will be used for compilation - **MORE THREADS, MORE RAM NEEDED** - if you do not have enough RAM, lower this number down, **6-8 Threads is recommended (about 7GB free RAM needed)**
4. Model - model that will be accelerated
5. Accelerate Button - click this to start the acceleration process

::: warning
Due to extreme load on the CPU, connection between the browser and the server will be lost. If this happens, you need to click the reconnect button - acceleration should not be affected.

WE WOULD RECOMMEND RESTARTING VOLTA AFTER ACCELERATION IS DONE, AS WEBSOCKETS MIGHT BE TOTALLY BROKEN
:::

## How to use

1. Switch the Model tab to AITemplate
2. Click the `Load` button
3. Generate an image - only txt2img, img2img and ControlNet are supported

![AITemplate Acceleration](../static/basics/aitemplate-load.webp)

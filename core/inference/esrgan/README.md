# Fork of [BlueAmulet's fork](https://github.com/BlueAmulet/ESRGAN) of [ESRGAN by Xinntao](https://github.com/xinntao/ESRGAN)

## NOTE: This repo is no longer being actively maintained. I highly recommend using [chaiNNer](https://github.com/chaiNNer-org/chaiNNer) instead, as it is still actively maintained and supports a lot more than this repo does.

This fork ports features over from my ESRGAN-Bot repository and adds a few more. It natively allows:

-   In-memory splitting/merging functionality (fully seamless, recently revamped for the third time and no longer requires tile size)
-   Seamless texture preservation (both tiled and mirrored)
-   Model chaining
-   Transparency preservation (3 different modes)
-   1-bit transparency support (with half transparency as well)
-   Variations of the ESRGAN (RRDB) architecture, including normal ESRGAN, ESRGAN+, BSRGAN, RealSR, SPSR, Real-ESRGAN, as well as Real-ESRGANv2 (SRVGG) architecture
-   Any scale and most other internal model parameter settings
-   On-the-fly interpolation

**Tile size was recently removed! It is no longer needed for split/merge functionality!**

To set your textures to seamless, use the `--seamless` flag. For regular tiled seamless mode, use `--seamless tile`. For mirrored seamless mode, use `--seamless mirror`. You can also add pixel-replication padding using `--seamless replicate` and alpha padding using `--seamless alpha_pad`.

To chain models, simply put one model name after another with a `>` in between (you can also use `+` if using bash to avoid issues), such as `1xDeJpeg.pth>4xESRGAN.pth` **note: To use model chaining, model names must be the complete full name without the path included, and the models must be present in your `/models` folder. You can still use full model paths to upscale with a single model.**

For on-the-fly interpolation, you use this syntax: `<model1_name>:<##>&<model2_name>:<##>`, where the model name is the path to the model and ## is the numerical percentage to interpolate by. For example, `model1:50&model2:50` would interpolate model1 and model2 by 50 each. The numbers should add up to 100. If you have trouble using `:` or `&`, either try putting the interpolation string in quotes or use `@` or `|` respectively (`"model1@50|model2@50"`).

To use 1 bit binary alpha transparency, set the `--binary-alpha` flag to True. When using `--binary-alpha` transparency, provide the optional `--alpha-threshold` to specify the alpha transparency threshold. 1 bit binary transparency is useful when upscaling images that require that the end result has 1 bit transparency, e.g. PSX games. If you want to include half transparency, use `--ternary-alpha` instead, which allows you to set the `--alpha-boundary-offset` threshold.

The default alpha mode is now `none`. There are also now 3 other modes to choose from:

-   `--alpha-mode bg_difference`: Fills the alpha channel with both white and black and extracts the difference from each result.
-   `--alpha-mode separate`: Upscales the alpha channel by itself, as a fake 3 channel image (The IEU way) then combines with result.
-   `--alpha-mode swapping`: Shifts the channels so that it upscales the alpha channel along with other regular channels then combines with result.

To process images in reverse order, use `--reverse`. If needed, you can also skip existing files by using `--skip-existing`.

If you're upscaling images of the same size, you can do `--cache-max-split-depth` to only calculate the automatic tile size once to improve performance.

Examples:

-   `python upscale.py 4xBox.pth --seamless tile`
-   `python upscale.py 1xSSAntiAlias9x.pth>4xBox.pth`
-   `python upscale.py 4xBox.pth --binary-alpha --alpha-threshold .2`
-   `python upscale.py /models/4xBox.pth`
-   `python upscale.py "1x_model1.pth@50|1x_model2.pth@50>2x_model3.pth"`

## Dependencies and Installation

Via pip:

```
pip install --user -r requirements.txt
```

## GUIs
- [chaiNNer](https://github.com/chaiNNer-org/chaiNNer)
    - My GUI, has all these same features available plus more architecture support. Node based and very customizable. Multiplatform.
- [Cupscale](https://github.com/n00mkrad/cupscale/).
    - Windows only. Implements this fork as well as other utilities around it.

import{_ as e,o as i,c as a,Q as l}from"./chunks/framework.70b8ae0d.js";const p=JSON.parse('{"title":"Changelog","description":"","frontmatter":{},"headers":[],"relativePath":"changelog.md","filePath":"changelog.md","lastUpdated":1701102090000}'),t={name:"changelog.md"},o=l('<h1 id="changelog" tabindex="-1">Changelog <a class="header-anchor" href="#changelog" aria-label="Permalink to &quot;Changelog&quot;">​</a></h1><h2 id="v0-5-0" tabindex="-1">v0.5.0 <a class="header-anchor" href="#v0-5-0" aria-label="Permalink to &quot;v0.5.0&quot;">​</a></h2><ul><li>Highres fix can now use image upscalers (ESRGAN, RealSR, etc.) for the intermediate step</li><li>API converted to sync where applicable, this should resolve some issues with websockets and thread lockups</li><li>FreeU support</li><li>Diffusers version bump</li></ul><h2 id="v0-4-2" tabindex="-1">v0.4.2 <a class="header-anchor" href="#v0-4-2" aria-label="Permalink to &quot;v0.4.2&quot;">​</a></h2><h3 id="critical-fix" tabindex="-1">Critical fix <a class="header-anchor" href="#critical-fix" aria-label="Permalink to &quot;Critical fix&quot;">​</a></h3><ul><li>PyTorch will now download CUDA version instead of CPU version if available</li></ul><h2 id="v0-4-1" tabindex="-1">v0.4.1 <a class="header-anchor" href="#v0-4-1" aria-label="Permalink to &quot;v0.4.1&quot;">​</a></h2><h3 id="bug-fixes" tabindex="-1">Bug Fixes <a class="header-anchor" href="#bug-fixes" aria-label="Permalink to &quot;Bug Fixes&quot;">​</a></h3><ul><li>Fixed loras on latest version of diffusers</li><li>Fixed Karras sigmas</li><li>Fixed incorrect step count being displayed in the UI</li><li>Fixed CivitAI browser getting stuck in some scenarios</li></ul><h3 id="all-changes" tabindex="-1">All changes <a class="header-anchor" href="#all-changes" aria-label="Permalink to &quot;All changes&quot;">​</a></h3><ul><li>Added support for prompt expansion</li><li>Reorganized frontend code</li></ul><h2 id="v0-4-0" tabindex="-1">v0.4.0 <a class="header-anchor" href="#v0-4-0" aria-label="Permalink to &quot;v0.4.0&quot;">​</a></h2><h3 id="biggest-changes" tabindex="-1">Biggest Changes <a class="header-anchor" href="#biggest-changes" aria-label="Permalink to &quot;Biggest Changes&quot;">​</a></h3><ul><li>Hi-res fix for AITemplate</li><li>Model and VAE autoloading</li><li>Partial support for Kdiffusion samplers (might be broken in some cases - controlnet, hi-res...)</li><li>Hypertile support (<a href="https://github.com/tfernd/HyperTile" target="_blank" rel="noreferrer">https://github.com/tfernd/HyperTile</a>)</li><li>Theme overhaul</li></ul><h3 id="all-changes-1" tabindex="-1">All Changes <a class="header-anchor" href="#all-changes-1" aria-label="Permalink to &quot;All Changes&quot;">​</a></h3><ul><li>Added docs for Vast.ai</li><li>Better Settings UI</li><li>Updated Model Manager</li><li>Garbage Collection improvements</li><li>Hi-res fix for AITemplate</li><li>Added DPMSolverSDE diffusers sampler</li><li>Model autoloading</li><li>Partial support for Kdiffusion samplers (might be broken in some cases - controlnet, hi-res...)</li><li>Tag autofill</li><li>New SendTo UI that is context aware</li><li>Fixed symlink deletion bug for models</li><li>New documentation theme</li><li>New sigma types for Kdiffusion (exponential, polyexponential, VP)</li><li>Image upload should now display correct dimensions</li><li>Fixed WebP crashes in some cases</li><li>Remote LoRA support <code>&lt;lora:URL:weight&gt;</code></li><li>Fix some CORS issues</li><li>Hypertile support (<a href="https://github.com/tfernd/HyperTile" target="_blank" rel="noreferrer">https://github.com/tfernd/HyperTile</a>)</li><li>Fixed uvicorn logging issues</li><li>Fixed update checker</li><li>Added some extra tooltips to the UI</li><li>Sampler config override for people that hate their free time</li><li>Bumped dependency versions</li><li>Image browser entries should get sorted on server, removing the need for layout shift in the UI</li><li>Cleaned up some old documentation</li><li>Transfer project from PyLint to Ruff</li><li>Github Actions CI for Ruff linting</li><li>Theme overhaul</li><li>Fixed NaiveUI ThemeEditor</li><li>Sort models in Model Loader</li><li>Console logs now accessible in the UI</li><li>...and probably a lot more that I already forgot</li></ul><h3 id="contributors" tabindex="-1">Contributors <a class="header-anchor" href="#contributors" aria-label="Permalink to &quot;Contributors&quot;">​</a></h3><ul><li>gabe56f (<a href="https://github.com/gabe56f" target="_blank" rel="noreferrer">https://github.com/gabe56f</a>)</li><li>Stax124 (<a href="https://github.com/Stax124" target="_blank" rel="noreferrer">https://github.com/Stax124</a>)</li><li>Katehuuh (<a href="https://github.com/Katehuuh" target="_blank" rel="noreferrer">https://github.com/Katehuuh</a>)</li></ul><h3 id="additional-notes" tabindex="-1">Additional Notes <a class="header-anchor" href="#additional-notes" aria-label="Permalink to &quot;Additional Notes&quot;">​</a></h3><p>Thank you for 850 stars on GitHub and 500 Discord members ❤️</p>',20),r=[o];function s(n,h,d,c,u,g){return i(),a("div",null,r)}const m=e(t,[["render",s]]);export{p as __pageData,m as default};

import{_ as a,o,c as e,Q as t}from"./chunks/framework.70b8ae0d.js";const _=JSON.parse('{"title":"Autoload","description":"","frontmatter":{},"headers":[],"relativePath":"basics/autoload.md","filePath":"basics/autoload.md","lastUpdated":1704041333000}'),l={name:"basics/autoload.md"},s=t('<h1 id="autoload" tabindex="-1">Autoload <a class="header-anchor" href="#autoload" aria-label="Permalink to &quot;Autoload&quot;">​</a></h1><p>Volta can be configured to automatically load a model (or even multiple models at once), Textual Inversion or even a custom VAE for a specific model.</p><div class="tip custom-block"><p class="custom-block-title">TIP</p><p>To see autoload in action, save the settings and restart Volta. You should see the model loading automatically.</p></div><h2 id="how-to-use" tabindex="-1">How to use <a class="header-anchor" href="#how-to-use" aria-label="Permalink to &quot;How to use&quot;">​</a></h2><p>Navigate to <code>Settings &gt; API &gt; Autoload</code> and select a model that you would like to load at the startup. Feel free to select multiple models at once if you have enough GPU memory / Offload enabled.</p><div class="warning custom-block"><p class="custom-block-title">WARNING</p><p>To save settings, click either on the <code>Save settings</code> button or navigate to other tab. Notification will appear if the settings were saved successfully.</p></div><h2 id="autoloading-textual-inversion" tabindex="-1">Autoloading Textual Inversion <a class="header-anchor" href="#autoloading-textual-inversion" aria-label="Permalink to &quot;Autoloading Textual Inversion&quot;">​</a></h2><p>Autoloading Textual inversion will apply to all models. You can check the status in the Model Loader.</p><h2 id="autoloading-custom-vae" tabindex="-1">Autoloading custom VAE <a class="header-anchor" href="#autoloading-custom-vae" aria-label="Permalink to &quot;Autoloading custom VAE&quot;">​</a></h2><p>Custom VAEs are loaded depending on the model and should be applied automatically. You can check this behavior in the Model Loader.</p>',10),i=[s];function d(n,c,u,r,h,p){return o(),e("div",null,i)}const g=a(l,[["render",d]]);export{_ as __pageData,g as default};
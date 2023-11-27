import{E as h,V as m,o,c as u,l as i,b as n,w as r,I as l,e as c,Q as p,k as e,a}from"./chunks/framework.70b8ae0d.js";const f="/voltaML-fast-stable-diffusion/assets/sgm_on.0d34ed9b.webp",d="/voltaML-fast-stable-diffusion/assets/sgm_off.800fdbee.webp",g="/voltaML-fast-stable-diffusion/assets/quant_on.53fb9ca2.webp",_=p('<h1 id="reproducibility-generation" tabindex="-1">Reproducibility &amp; Generation <a class="header-anchor" href="#reproducibility-generation" aria-label="Permalink to &quot;Reproducibility &amp; Generation&quot;">​</a></h1><p>Reproducibility settings are settings that change generation output. These changes can vary from small, to large, with small being a few lines look sharper</p><h2 id="device" tabindex="-1">Device <a class="header-anchor" href="#device" aria-label="Permalink to &quot;Device&quot;">​</a></h2><p>Changing the device to the correct one -- that being, your fastest available GPU -- can not only improve performance, but also change how the images look like. Something generated using DirectML on an AMD card won&#39;t EVER look the same as something generated with CUDA.</p><h2 id="data-type" tabindex="-1">Data type <a class="header-anchor" href="#data-type" aria-label="Permalink to &quot;Data type&quot;">​</a></h2><p>Generally, changing data type to a lower precision (lower number) one, will improve performance, however, when taken to extreme degrees (volta doesn&#39;t have this implemented) image quality starts to get hammered. <code>16-bit float</code> or <code>16-bit bfloat</code> is generally the lowest people should need to go.</p><h2 id="deterministic-generation" tabindex="-1">Deterministic generation <a class="header-anchor" href="#deterministic-generation" aria-label="Permalink to &quot;Deterministic generation&quot;">​</a></h2><p>PyTorch, and as such, Volta, is by design indeterministic, - that is, not 100% reproducible. This can raise a few issues: generations using the exact same parameters <strong>MAY NOT</strong> come out the same. Changing this to on, should fix these issues.</p><h2 id="sgm-noise-multiplier" tabindex="-1">SGM Noise Multiplier <a class="header-anchor" href="#sgm-noise-multiplier" aria-label="Permalink to &quot;SGM Noise Multiplier&quot;">​</a></h2><p>SGM Noise multiplier changes how noise is calculated. This is only useful for reproducing already created images. From a more technical standpoint: this changes noising to mimic SDXL&#39;s noise creation. <strong>Only useful on <code>SD1.x</code>.</strong></p><h3 id="on-vs-off" tabindex="-1">On vs. off <a class="header-anchor" href="#on-vs-off" aria-label="Permalink to &quot;On vs. off&quot;">​</a></h3>',11),b=e("img",{slot:"first",style:{width:"100%"},src:f},null,-1),y=e("img",{slot:"second",style:{width:"100%"},src:d},null,-1),v=e("h2",{id:"quantization-in-kdiff-samplers",tabindex:"-1"},[a("Quantization in KDiff samplers "),e("a",{class:"header-anchor",href:"#quantization-in-kdiff-samplers","aria-label":'Permalink to "Quantization in KDiff samplers"'},"​")],-1),w=e("p",null,[a("Quantization in K-samplers helps the samplers to create more sharp and defined lines. This is another one of those "),e("em",null,'"small, but useful"'),a(" changes.")],-1),k=e("h3",{id:"on-vs-off-1",tabindex:"-1"},[a("On vs. off "),e("a",{class:"header-anchor",href:"#on-vs-off-1","aria-label":'Permalink to "On vs. off"'},"​")],-1),x=e("img",{slot:"first",style:{width:"100%"},src:g},null,-1),D=e("img",{slot:"second",style:{width:"100%"},src:d},null,-1),P=e("h2",{id:"generator",tabindex:"-1"},[a("Generator "),e("a",{class:"header-anchor",href:"#generator","aria-label":'Permalink to "Generator"'},"​")],-1),C=JSON.parse('{"title":"Reproducibility & Generation","description":"","frontmatter":{},"headers":[],"relativePath":"settings/reproducibility.md","filePath":"settings/reproducibility.md","lastUpdated":1701110036000}'),q={name:"settings/reproducibility.md"},G=Object.assign(q,{setup(T){let t=h(null);return m(()=>import("./chunks/index.esm.10496f3a.js"),["assets/chunks/index.esm.10496f3a.js","assets/chunks/framework.70b8ae0d.js"]).then(s=>{t.value=s.ImgComparisonSlider,console.log(t)}),(s,S)=>(o(),u("div",null,[_,i(t)?(o(),n(l(i(t)),{key:0},{default:r(()=>[b,y]),_:1})):c("",!0),v,w,k,i(t)?(o(),n(l(i(t)),{key:1},{default:r(()=>[x,D]),_:1})):c("",!0),P]))}});export{C as __pageData,G as default};

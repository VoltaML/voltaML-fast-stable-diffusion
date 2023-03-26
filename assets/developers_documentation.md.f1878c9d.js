import{_ as e,c as n,o as a,a as s}from"./app.1aa49d57.js";const b=JSON.parse('{"title":"Documentation","description":"","frontmatter":{},"headers":[{"level":2,"title":"Rules","slug":"rules","link":"#rules","children":[]},{"level":2,"title":"How to edit","slug":"how-to-edit","link":"#how-to-edit","children":[]},{"level":2,"title":"Running documentation locally","slug":"running-documentation-locally","link":"#running-documentation-locally","children":[]}],"relativePath":"developers/documentation.md","lastUpdated":1679861903000}'),l={name:"developers/documentation.md"},o=s(`<h1 id="documentation" tabindex="-1">Documentation <a class="header-anchor" href="#documentation" aria-hidden="true">#</a></h1><p>This is will show you how to edit our documentation and how to properly contribute while outlining some rules for us.</p><h2 id="rules" tabindex="-1">Rules <a class="header-anchor" href="#rules" aria-hidden="true">#</a></h2><div class="warning custom-block"><p class="custom-block-title">WARNING</p><p>Please read the rules before you start editing the documentation.</p></div><ul><li>All images will be in WEBP format with maximal 90% image quality</li><li>Images will be of sensible resolution (no 4k or higher resolution images)</li><li>English only</li><li>Grammarly correct when possible</li><li>Keep it simple</li></ul><h2 id="how-to-edit" tabindex="-1">How to edit <a class="header-anchor" href="#how-to-edit" aria-hidden="true">#</a></h2><p>All documentation is written in Markdown and is located in the <code>docs</code> folder. You can edit it directly on GitHub or you can clone the repository and edit it locally.</p><p>Edits on GitHub will create a Pull Request with the changes and they will be waiting for review.</p><p>Once the change is reviewed and approved it will be merged into the branch and will be deployed by our CI/CD pipeline.</p><h2 id="running-documentation-locally" tabindex="-1">Running documentation locally <a class="header-anchor" href="#running-documentation-locally" aria-hidden="true">#</a></h2><p>Clone the repository</p><div class="language-bash line-numbers-mode"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki one-dark-pro has-diff"><code><span class="line"><span style="color:#ABB2BF;">git </span><span style="color:#98C379;">clone</span><span style="color:#ABB2BF;"> </span><span style="color:#98C379;">https://github.com/VoltaML/voltaML-fast-stable-diffusion.git</span></span>
<span class="line"></span></code></pre><div class="line-numbers-wrapper" aria-hidden="true"><span class="line-number">1</span><br></div></div><p>Install dependencies</p><div class="language-bash line-numbers-mode"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki one-dark-pro"><code><span class="line"><span style="color:#ABB2BF;">yarn </span><span style="color:#98C379;">install</span></span>
<span class="line"></span></code></pre><div class="line-numbers-wrapper" aria-hidden="true"><span class="line-number">1</span><br></div></div><p>Run the documentation</p><div class="language-bash line-numbers-mode"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki one-dark-pro"><code><span class="line"><span style="color:#ABB2BF;">yarn </span><span style="color:#98C379;">docs:dev</span></span>
<span class="line"></span></code></pre><div class="line-numbers-wrapper" aria-hidden="true"><span class="line-number">1</span><br></div></div><p>You should now be able to access the documentation on <code>http://localhost:5173/voltaML-fast-stable-diffusion/</code></p>`,17),t=[o];function i(r,d,c,p,u,h){return a(),n("div",null,t)}const y=e(l,[["render",i]]);export{b as __pageData,y as default};
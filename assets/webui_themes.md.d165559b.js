import{_ as s,o as a,c as e,Q as n}from"./chunks/framework.70b8ae0d.js";const m=JSON.parse('{"title":"Themes","description":"","frontmatter":{},"headers":[],"relativePath":"webui/themes.md","filePath":"webui/themes.md","lastUpdated":1704041333000}'),t={name:"webui/themes.md"},o=n(`<h1 id="themes" tabindex="-1">Themes <a class="header-anchor" href="#themes" aria-label="Permalink to &quot;Themes&quot;">​</a></h1><p>Volta includes 4 themes as of the time of writing:</p><ul><li>Dark (<em>default</em>)</li><li>Dark Flat</li><li>Light</li><li>Light Flat</li></ul><p>Dark and Light have neon-ish vibes, while Flat themes are more minimalistic and lack a background.</p><h2 id="changing-the-theme" tabindex="-1">Changing the theme <a class="header-anchor" href="#changing-the-theme" aria-label="Permalink to &quot;Changing the theme&quot;">​</a></h2><p>Theme can be changed on the settings page: <code>Settings &gt; Theme &gt; Theme</code></p><h2 id="importing-themes" tabindex="-1">Importing themes <a class="header-anchor" href="#importing-themes" aria-label="Permalink to &quot;Importing themes&quot;">​</a></h2><p>Any themes that you download should be placed in the <code>data/themes</code> directory and they should be picked up automatically (refresh the UI page for them to show up there).</p><h2 id="creating-a-theme" tabindex="-1">Creating a theme <a class="header-anchor" href="#creating-a-theme" aria-label="Permalink to &quot;Creating a theme&quot;">​</a></h2><p>I would recommend setting <code>Settings &gt; Theme &gt; Enable Theme Editor</code> to <code>true</code> to make the process of creating a theme easier. This should enable the theme editor inside the UI, you should be able to see it in the bottom right corner of the screen.</p><p>Changes are cached, so I would recommend pressing the <code>Clear All Variables</code> button first to just be sure.</p><p>Now, you can start changing the variables and see the changes in real time. Once you are happy with the result, you can press the <code>Export</code> button and save the theme to a file.</p><p>Then, open either <code>data/themes/dark.json</code> or <code>data/themes/light.json</code> and copy the <code>volta</code> object to your theme file.</p><div class="language-json line-numbers-mode"><button title="Copy Code" class="copy"></button><span class="lang">json</span><pre class="shiki one-dark-pro has-highlighted-lines has-diff"><code><span class="line"><span style="color:#ABB2BF;">{</span></span>
<span class="line"><span style="color:#ABB2BF;">	</span><span style="color:#7F848E;font-style:italic;">// Feel free to change these settings once you copy them!</span></span>
<span class="line highlighted"><span style="color:#ABB2BF;">	</span><span style="color:#E06C75;">&quot;volta&quot;</span><span style="color:#ABB2BF;">: {</span></span>
<span class="line highlighted"><span style="color:#ABB2BF;">		</span><span style="color:#E06C75;">&quot;base&quot;</span><span style="color:#ABB2BF;">: </span><span style="color:#98C379;">&quot;dark&quot;</span><span style="color:#ABB2BF;">,</span></span>
<span class="line highlighted"><span style="color:#ABB2BF;">		</span><span style="color:#E06C75;">&quot;blur&quot;</span><span style="color:#ABB2BF;">: </span><span style="color:#98C379;">&quot;6px&quot;</span><span style="color:#ABB2BF;">,</span></span>
<span class="line highlighted"><span style="color:#ABB2BF;">		</span><span style="color:#E06C75;">&quot;backgroundImage&quot;</span><span style="color:#ABB2BF;">: </span><span style="color:#98C379;">&quot;https://raw.githubusercontent.com/VoltaML/voltaML-fast-stable-diffusion/2cf7a8abf1e5035a0dc57a67cd13505653c492f6/static/volta-dark-background.svg&quot;</span></span>
<span class="line highlighted"><span style="color:#ABB2BF;">	},</span></span>
<span class="line"><span style="color:#ABB2BF;">	</span><span style="color:#E06C75;">&quot;common&quot;</span><span style="color:#ABB2BF;">: {</span></span>
<span class="line"><span style="color:#ABB2BF;">		</span><span style="color:#E06C75;">&quot;fontSize&quot;</span><span style="color:#ABB2BF;">: </span><span style="color:#98C379;">&quot;15px&quot;</span><span style="color:#ABB2BF;">,</span></span>
<span class="line"><span style="color:#ABB2BF;">		</span><span style="color:#E06C75;">&quot;fontWeight&quot;</span><span style="color:#ABB2BF;">: </span><span style="color:#98C379;">&quot;600&quot;</span></span>
<span class="line"><span style="color:#ABB2BF;">	},</span></span>
<span class="line"><span style="color:#ABB2BF;">	</span><span style="color:#E06C75;">&quot;Card&quot;</span><span style="color:#ABB2BF;">: {</span></span>
<span class="line"><span style="color:#ABB2BF;">		</span><span style="color:#E06C75;">&quot;color&quot;</span><span style="color:#ABB2BF;">: </span><span style="color:#98C379;">&quot;rgba(24, 24, 28, 0.6)&quot;</span></span>
<span class="line"><span style="color:#ABB2BF;">	},</span></span>
<span class="line"><span style="color:#ABB2BF;">	</span><span style="color:#E06C75;">&quot;Layout&quot;</span><span style="color:#ABB2BF;">: {</span></span>
<span class="line"><span style="color:#ABB2BF;">		</span><span style="color:#E06C75;">&quot;color&quot;</span><span style="color:#ABB2BF;">: </span><span style="color:#98C379;">&quot;rgba(16, 16, 20, 0.6)&quot;</span><span style="color:#ABB2BF;">,</span></span>
<span class="line"><span style="color:#ABB2BF;">		</span><span style="color:#E06C75;">&quot;siderColor&quot;</span><span style="color:#ABB2BF;">: </span><span style="color:#98C379;">&quot;rgba(24, 24, 28, 0)&quot;</span></span>
<span class="line"><span style="color:#ABB2BF;">	},</span></span>
<span class="line"><span style="color:#ABB2BF;">	</span><span style="color:#E06C75;">&quot;Tabs&quot;</span><span style="color:#ABB2BF;">: {</span></span>
<span class="line"><span style="color:#ABB2BF;">		</span><span style="color:#E06C75;">&quot;colorSegment&quot;</span><span style="color:#ABB2BF;">: </span><span style="color:#98C379;">&quot;rgba(24, 24, 28, 0.6)&quot;</span></span>
<span class="line"><span style="color:#ABB2BF;">	}</span></span>
<span class="line"><span style="color:#ABB2BF;">}</span></span></code></pre><div class="line-numbers-wrapper" aria-hidden="true"><span class="line-number">1</span><br><span class="line-number">2</span><br><span class="line-number">3</span><br><span class="line-number">4</span><br><span class="line-number">5</span><br><span class="line-number">6</span><br><span class="line-number">7</span><br><span class="line-number">8</span><br><span class="line-number">9</span><br><span class="line-number">10</span><br><span class="line-number">11</span><br><span class="line-number">12</span><br><span class="line-number">13</span><br><span class="line-number">14</span><br><span class="line-number">15</span><br><span class="line-number">16</span><br><span class="line-number">17</span><br><span class="line-number">18</span><br><span class="line-number">19</span><br><span class="line-number">20</span><br><span class="line-number">21</span><br><span class="line-number">22</span><br></div></div>`,14),l=[o];function p(r,c,i,h,u,B){return a(),e("div",null,l)}const b=s(t,[["render",p]]);export{m as __pageData,b as default};
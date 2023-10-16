import { defineConfig } from "vitepress";

// https://vitepress.dev/reference/site-config
export default defineConfig({
	title: "VoltaML",
	description: "Easy to use, yet feature-rich WebUI",
	lang: "en-US",
	appearance: "dark",
	lastUpdated: true,
	base: "/voltaML-fast-stable-diffusion/",
	locales: {
		root: {
			lang: "en-US",
			label: "English",
		},
	},
	head: [
		[
			"link",
			{
				rel: "shortcut icon",
				type: "image/x-icon",
				href: "/voltaML-fast-stable-diffusion/favicon.ico",
			},
		],
		[
			"meta",
			{
				property: "og:image",
				content: "/voltaML-fast-stable-diffusion/volta-og-image.webp",
			},
		],
		[
			"script",
			{
				async: "",
				src: "https://www.googletagmanager.com/gtag/js?id=G-WZPQL8HDP0",
			},
		],
		[
			"script",
			{},
			"window.dataLayer = window.dataLayer || [];\nfunction gtag(){dataLayer.push(arguments);}\ngtag('js', new Date());\ngtag('config', 'G-WZPQL8HDP0');",
		],
	],
	themeConfig: {
		search: {
			provider: "local",
		},
		nav: [
			{ text: "Home", link: "/" },
			{ text: "Docs", link: "/getting-started/introduction" },
			{ text: "Changelog", link: "/changelog" },
		],
		editLink: {
			pattern:
				"https://github.com/VoltaML/voltaML-fast-stable-diffusion/edit/main/docs/:path",
		},
		logo: "/volta-rounded.webp",
		socialLinks: [
			{
				icon: "github",
				link: "https://github.com/voltaML/voltaML-fast-stable-diffusion",
			},
			{
				icon: "discord",
				link: "https://discord.gg/pY5SVyHmWm",
			},
			{
				icon: "linkedin",
				link: "https://www.linkedin.com/in/tom%C3%A1%C5%A1-nov%C3%A1k-5a163321b/",
			},
			{
				icon: "linkedin",
				link: "https://www.linkedin.com/in/márton-kissik/",
			},
		],
		sidebar: [
			{
				text: "Introduction",
				items: [
					{ text: "Introduction", link: "/getting-started/introduction" },
				],
				collapsed: false,
			},
			{
				text: "Installation",
				items: [
					{
						text: "Local",
						items: [
							{ text: "Windows", link: "/installation/windows" },
							{ text: "Linux", link: "/installation/linux" },
							{ text: "WSL", link: "/installation/wsl" },
							{ text: "Docker", link: "/installation/docker" },
						],
					},
					{
						text: "Cloud",
						items: [{ text: "Vast.ai", link: "/installation/vast" }],
					},
					{
						text: "Extra",
						items: [
							{ text: "Old", link: "/installation/old" },
							{ text: "xFormers", link: "/installation/xformers" },
						],
					},
				],
				collapsed: false,
			},
			{
				text: "Guides",
				items: [{ text: "First image", link: "/guides/first-image" }],
				collapsed: false,
			},
			{
				text: "Basics",
				items: [
					{ text: "Models", link: "/basics/models" },
					{ text: "LoRA", link: "/basics/lora" },
					{ text: "Textual Inversion", link: "/basics/textual-inversion" },
					{
						text: "AITemplate Acceleration",
						link: "/basics/aitemplate",
					},
					{ text: "Autoload", link: "/basics/autoload" },
				],
				collapsed: false,
			},
			{
				text: "WebUI",
				items: [
					{ text: "Image to Image", link: "/webui/img2img" },
					{ text: "Downloading Models", link: "/webui/download" },
					{ text: "Image browser", link: "/webui/imagebrowser" },
				],
				collapsed: false,
			},
			{
				text: "Bot",
				items: [
					{ text: "Setup", link: "/bot/setup" },
					{
						text: "Commands",
						link: "/bot/commands",
					},
				],
				collapsed: false,
			},
			{
				text: "Settings",
				items: [{ text: "Settings", link: "/settings/settings" }],
				collapsed: false,
			},
			{
				text: "Experimental",
				items: [],
				collapsed: false,
			},
			{
				text: "API",
				items: [{ text: "API", link: "/api/" }],
				collapsed: true,
			},
			{
				text: "Developers",
				items: [
					{ text: "API", link: "/developers/api" },
					{
						text: "Frontend",
						link: "/developers/frontend",
					},
					{
						text: "Documentation",
						link: "/developers/documentation",
					},
					{
						text: "Testing",
						link: "/developers/testing",
					},
				],
				collapsed: true,
			},
			{
				text: "Troubleshooting",
				items: [
					{ text: "Linux", link: "/troubleshooting/linux" },
					{
						text: "Windows",
						link: "/troubleshooting/windows",
					},
					{
						text: "Docker",
						link: "/troubleshooting/docker",
					},
				],
				collapsed: true,
			},
		],
	},
	cleanUrls: true,
	ignoreDeadLinks: "localhostLinks",
	markdown: {
		theme: "one-dark-pro",
		lineNumbers: true,
	},
});

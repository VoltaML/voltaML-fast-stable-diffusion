# Themes

Volta includes 4 themes as of the time of writing:

- Dark (_default_)
- Dark Flat
- Light
- Light Flat

Dark and Light have neon-ish vibes, while Flat themes are more minimalistic and lack a background.

## Changing the theme

Theme can be changed on the settings page: `Settings > Theme > Theme`

## Importing themes

Any themes that you download should be placed in the `data/themes` directory and they should be picked up automatically (refresh the UI page for them to show up there).

## Creating a theme

I would recommend setting `Settings > Theme > Enable Theme Editor` to `true` to make the process of creating a theme easier.
This should enable the theme editor inside the UI, you should be able to see it in the bottom right corner of the screen.

Changes are cached, so I would recommend pressing the `Clear All Variables` button first to just be sure.

Now, you can start changing the variables and see the changes in real time. Once you are happy with the result, you can press the `Export` button and save the theme to a file.

Then, open either `data/themes/dark.json` or `data/themes/light.json` and copy the `volta` object to your theme file.

```json{3-7}
{
	// Feel free to change these settings once you copy them!
	"volta": {
		"base": "dark",
		"blur": "6px",
		"backgroundImage": "https://raw.githubusercontent.com/VoltaML/voltaML-fast-stable-diffusion/2cf7a8abf1e5035a0dc57a67cd13505653c492f6/static/volta-dark-background.svg"
	},
	"common": {
		"fontSize": "15px",
		"fontWeight": "600"
	},
	"Card": {
		"color": "rgba(24, 24, 28, 0.6)"
	},
	"Layout": {
		"color": "rgba(16, 16, 20, 0.6)",
		"siderColor": "rgba(24, 24, 28, 0)"
	},
	"Tabs": {
		"colorSegment": "rgba(24, 24, 28, 0.6)"
	}
}
```

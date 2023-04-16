# Settings

This is an experimental feature that is available in both the UI and the API. It can be fund in the `Settings` tab in the UI.

Settings are split into multiple categories:

- `Frontend` - UI settings (usually default values for the UI)
- `API` - API settings
- `Bot` - Settings for the Discord bot
- `General` - General settings

All settings will be stored as a json file on your local machine (server).
This file is named `settings.json` and can be found in the `data` directory of the project.

::: warning
Settings are not automatically saved. You need to click the `Save Settings` button to save the settings.
:::

::: warning
When resetting the settings, the settings will be reset to the default values, but will once again need to be saved manully.
:::

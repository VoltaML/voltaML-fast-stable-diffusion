# Autoload

Volta can be configured to automatically load a model (or even multiple models at once), Textual Inversion or even a custom VAE for a specific model.

::: tip
To see autoload in action, save the settings and restart Volta. You should see the model loading automatically.
:::

## How to use

Navigate to `Settings > API > Autoload` and select a model that you would like to load at the startup. Feel free to select multiple models at once if you have enough GPU memory / Offload enabled.

::: warning
To save settings, click either on the `Save settings` button or navigate to other tab. Notification will appear if the settings were saved successfully.
:::

## Autoloading Textual Inversion

Autoloading Textual inversion will apply to all models. You can check the status in the Model Loader.

## Autoloading custom VAE

Custom VAEs are loaded depending on the model and should be applied automatically. You can check this behavior in the Model Loader.

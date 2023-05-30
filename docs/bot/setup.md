# Setup

Discord bot is already setup and prepared for you with the [Docker](/installation/docker) installation. You will need to setup the bot yourself if you are using the manual installation.

## Requirements

::: warning READ THIS CAREFULLY
Scopes: `Bot`, `applications.commands` <br>
Permissions: `Administrator` <br>
Privileged Gateway Intents: `All` of them for ease of use
:::

Discord bot requires a bot token which you can get from the [Discord Developer Portal](https://discord.com/developers/applications). You will also need to create a bot user and invite it to your server. You can find a guide on how to do that [here](https://discordpy.readthedocs.io/en/latest/discord.html).

## Configuration

You need to pass the Token to the container via the `.env` file. It is called `DISCORD_BOT_TOKEN` and there should already be a placeholder for it in the `.env` file.

Then you need to append the `--bot` argument to the `EXTRA_ARGS` variable in the `.env` file. It should look like this:

```bash{11,14}
# Hugging Face Token (https://huggingface.co/settings/tokens)
HUGGINGFACE_TOKEN=

# Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=INFO

# [Optional] Analytics (https://my-api-analytics.vercel.app/generate) (https://my-api-analytics.vercel.app/dashboard)
FASTAPI_ANALYTICS_KEY=

# [Optional] Discord Bot Token (https://discord.com/developers/applications)
DISCORD_BOT_TOKEN=YOUR_TOKEN_HERE

# [Optional] Extra arguments for the API
EXTRA_ARGS="--bot"
```

All you need to do is **paste** your token, set the extra argument and **restart the container**.

## First time sync

::: tip
This process may take a while, Discord needs to inform all the servers about the new commands. Please be patient.
:::

The first time you start the bot, bot needs to sync all the commands with Discord. You need to do this manually by running the following command:

```
!sync
```

This will sync all the commands and you should be good to go.

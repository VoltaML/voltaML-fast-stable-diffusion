from bot.helper import get_available_models, get_loaded_models


class BotModelManager:
    "Class for internally managing models"

    def __init__(self):
        self._cached_models = []
        self._loaded_models = []

    async def cached_available_models(self):
        "List all available models"

        if not self._cached_models:
            self._cached_models, code = await get_available_models()
        else:
            code = 200

        return self._cached_models, code

    def set_cached_available_models(self, value):
        "Set the internal cached models"

        self._cached_models = value

    async def cached_loaded_models(self):
        "List all loaded models"

        if not self._loaded_models:
            self._loaded_models, code = await get_loaded_models()
        else:
            code = 200

        return self._loaded_models, code

    def set_cached_loaded_models(self, value):
        "Set the internal loaded models"

        self._loaded_models = value

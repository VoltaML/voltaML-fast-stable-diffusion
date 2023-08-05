from dataclasses import dataclass

from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers


@dataclass
class BotConfig:
    "Configuration for the bot"

    default_scheduler: KarrasDiffusionSchedulers = (
        KarrasDiffusionSchedulers.DPMSolverSinglestepScheduler
    )
    verbose: bool = False
    use_default_negative_prompt: bool = True
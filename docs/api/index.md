# API

Documentation can be found on http://localhost:5003/api/docs

## Basic structure

API endpoints are divided into few categories:

- Generate (inference and model conversions)
- Hardware (hardware statistics)
- Models (model management)
- Output (generated images)
- General (generally useful endpoints)
- Default (non categorized endpoints)

## Schedulers (Samplers)

```py
class KarrasDiffusionSchedulers(Enum):
    DDIMScheduler = 1
    DDPMScheduler = 2
    PNDMScheduler = 3
    LMSDiscreteScheduler = 4
    EulerDiscreteScheduler = 5
    HeunDiscreteScheduler = 6
    EulerAncestralDiscreteScheduler = 7
    DPMSolverMultistepScheduler = 8
    DPMSolverSinglestepScheduler = 9
    KDPM2DiscreteScheduler = 10
    KDPM2AncestralDiscreteScheduler = 11
    DEISMultistepScheduler = 12
    UniPCMultistepScheduler = 13
```

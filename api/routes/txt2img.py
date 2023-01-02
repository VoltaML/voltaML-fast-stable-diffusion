from fastapi import APIRouter, HTTPException

from api.shared import state
from api.types import Txt2imgJob
from core.inference.pytorch import infer_pt

router = APIRouter()


@router.post("/interupt")
async def stop():
    state.interrupt = True
    return {"message": "Interupted"}


@router.post("/voltaml/job")
def txt2img_job(job: Txt2imgJob):
    # Create directory to save images if it does not exist

    if job.backend == "PyTorch":
        pipeline_time = infer_pt(job)
    elif job.backend == "TensorRT":
        pipeline_time = 0
        # pipeline_time = infer_trt(job)
    else:
        raise HTTPException(status_code=400, detail="Invalid backend")

    return {"message": "Job completed", "time": pipeline_time}

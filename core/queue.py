import asyncio
from typing import List, Optional


class Queue:
    "A queue for handling jobs"

    def __init__(self) -> None:
        self.jobs: List[str] = []
        self.current_job: Optional[str] = None

    def mark_finished(self):
        "Mark the current job as finished"

        self.current_job = None

    async def wait_for_turn(self, job_id):
        "Wait until the job can be processed"

        self.jobs.append(job_id)
        if not self.current_job:
            self.current_job = job_id

        while self.current_job != job_id:
            await asyncio.sleep(0.1)

        return

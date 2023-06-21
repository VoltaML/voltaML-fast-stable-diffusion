import asyncio
import logging
from typing import List

from core.config import config

logger = logging.getLogger(__name__)


class Queue:
    "A queue for handling jobs"

    def __init__(self) -> None:
        self.jobs: List[str] = []
        self.lock = asyncio.Lock()
        self.condition = asyncio.Condition(self.lock)
        self.concurrent_jobs = config.api.concurrent_jobs

    async def mark_finished(self, job_id: str):
        "Mark the current job as finished"

        async with self.lock:
            try:
                self.jobs.remove(job_id)
            except ValueError:
                logger.warning(
                    f"Job {job_id} was not in the queue, assuming queue was cleared manually by 3rd party"
                )

            self.condition.notify_all()
            logger.info(f"Job {job_id} has been processed")

    async def wait_for_turn(self, job_id: str):
        "Wait until the job can be processed"

        async with self.lock:
            self.jobs.append(job_id)

            while job_id not in self.jobs[: self.concurrent_jobs]:
                await self.condition.wait()

            logger.info(f"Job {job_id} is now being processed")
            return

    def clear(self):
        "Clear the queue"

        self.jobs = []
        logger.info("Queue has been cleared")

    async def trigger_condition(self):
        "Trigger the condition manually, this will cause all jobs to check if they can be processed"

        async with self.lock:
            self.condition.notify_all()
            logger.info("Queue Condition has been triggered manually")

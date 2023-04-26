import asyncio
import logging
from asyncio import AbstractEventLoop
from typing import Coroutine, List, Optional

from fastapi import WebSocket
from psutil import NoSuchProcess

from api.websockets.data import Data
from core.config import config
from core.shared_dependent import gpu

logger = logging.getLogger(__name__)


class WebSocketManager:
    "Manages active websocket connections"

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.loop: Optional[AbstractEventLoop] = None
        self.to_run: List[Coroutine] = []

    async def sync_loop(self):
        "Infinite loop that runs all coroutines in the to_run list"

        while True:
            for task in self.to_run:
                await task
                self.to_run.remove(task)

            await asyncio.sleep(config.api.websocket_sync_interval)

    async def perf_loop(self):
        "Infinite loop that sends performance data to all active websocket connections"

        amd = False
        gpus = []
        try:
            from gpustat.core import GPUStatCollection

            gpus = [i.entry for i in GPUStatCollection.new_query().gpus]
        except Exception:  # pylint: disable=broad-exception-caught
            logger.info("GPUStat failed to initialize - probably not an NVIDIA GPU")
            logger.info("Trying pyamdgpuinfo...")
            try:
                import pyamdgpuinfo as amdgpu  # pylint: disable=shadowed-import

                if amdgpu.detect_gpus() == 0:
                    raise Exception(  # pylint: disable=broad-exception-raised,raise-missing-from
                        "hello"
                    )
                else:
                    gpus = [amdgpu.get_gpu(x) for x in range(amdgpu.detect_gpus())]
                    for stat in gpus:
                        stat.start_utilisation_polling()
                amd = True
            except Exception:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "User doesn't have an AMD nor an NVIDIA card. GPU info will be unavailable."
                )
                return

        while True:
            data = []
            if amd:
                for stat in gpus:
                    data.append(
                        {
                            "index": stat.gpu_id,
                            "uuid": "",
                            "name": stat.name if stat.name else "Generic AMD GPU",
                            "temperature": stat.query_temperature(),
                            "fan_speed": 0,
                            "utilization": stat.query_load(),
                            "power_draw": stat.query_power(),
                            "power_limit": stat.query_power(),
                            "memory_used": stat.query_vram_usage(),
                            "memory_total": stat.memory_info["vram_size"]
                            if stat.memory_info["vram_size"]
                            else 1024,
                            "memory_usage": int(
                                stat.query_vram_usage()
                                / (
                                    stat.memory_info["vram_size"]
                                    if stat.memory_info["vram_size"]
                                    else 1024
                                )
                                * 100
                            ),
                        }
                    )
            else:
                try:
                    for stat in gpus:
                        data.append(
                            {
                                "index": stat["index"],
                                "uuid": stat["uuid"],
                                "name": stat["name"],
                                "temperature": stat["temperature.gpu"],
                                "fan_speed": stat["fan.speed"],
                                "utilization": stat["utilization.gpu"],
                                "power_draw": stat["power.draw"],
                                "power_limit": stat["enforced.power.limit"],
                                "memory_used": stat["memory.used"],
                                "memory_total": stat["memory.total"],
                                "memory_usage": int(
                                    stat["memory.used"] / stat["memory.total"] * 100
                                ),
                            }
                        )

                except NoSuchProcess:
                    logger.debug("HW Stat - No such process, sleeping...")
            await self.broadcast(Data(data_type="cluster_stats", data=data))
            await asyncio.sleep(config.api.websocket_perf_interval)

    async def connect(self, websocket: WebSocket):
        "Accepts a new websocket connection and adds it to the list of active connections"

        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        "Removes a websocket connection from the list of active connections"

        if (
            len(self.active_connections) == 0
            and config.api.clear_memory_policy == "after_disconnect"
        ):
            gpu.memory_cleanup()

        self.active_connections.remove(websocket)

    async def send_personal_message(self, data: Data, websocket: WebSocket):
        "Sends a data message to a specific websocket connection"

        await websocket.send_json(data.to_json())

    async def broadcast(self, data: Data):
        "Broadcasts data message to all active websocket connections"

        for connection in self.active_connections:
            await connection.send_json(data.to_json())

    def broadcast_sync(self, data: Data):
        "Broadcasts data message to all active websocket connections synchronously"

        for connection in self.active_connections:
            self.to_run.append(connection.send_json(data.to_json()))

    async def close_all(self):
        "Closes all active websocket connections"

        for connection in self.active_connections:
            await connection.close(reason="Server shutdown")

        if config.api.clear_memory_policy == "after_disconnect":
            gpu.memory_cleanup()

        self.active_connections = []

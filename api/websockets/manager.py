import asyncio
import logging
from asyncio import AbstractEventLoop
from typing import List, Optional

import torch
from fastapi import WebSocket
from psutil import NoSuchProcess

from api.websockets.data import Data
from core import shared
from core.config import config

logger = logging.getLogger(__name__)


class WebSocketManager:
    "Manages active websocket connections"

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.loop: Optional[AbstractEventLoop] = None

    async def perf_loop(self):
        "Infinite loop that sends performance data to all active websocket connections"
        try:
            from gpustat.core import GPUStatCollection

            shared.all_gpus = [i.entry for i in GPUStatCollection.new_query().gpus]
        except Exception as e:
            logger.info(
                f"GPUStat failed to initialize - probably not an NVIDIA GPU: {e}"
            )
            logger.debug("Trying pyamdgpuinfo...")
            try:
                import pyamdgpuinfo

                if pyamdgpuinfo.detect_gpus() == 0:
                    raise ImportError("User doesn't have an AMD gpu")
                shared.all_gpus = [
                    pyamdgpuinfo.get_gpu(x) for x in range(pyamdgpuinfo.detect_gpus())
                ]
                shared.amd = True
            except Exception:
                logger.warning(
                    "User doesn't have an AMD nor an NVIDIA card. GPU info will be unavailable."
                )
                return

        while True:
            data = []
            if shared.amd:
                for stat in shared.all_gpus:
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
                            "memory_used": stat.query_vram_usage() / 1024**2,
                            "memory_total": stat.memory_info["vram_size"] / 1024**2
                            if stat.memory_info["vram_size"]
                            else 1024,
                            "memory_usage": int(
                                stat.query_vram_usage()
                                / stat.memory_info["vram_size"]
                                * 100
                            ),
                        }
                    )
            else:
                try:
                    from gpustat.core import GPUStatCollection

                    shared.all_gpus = [
                        i.entry for i in GPUStatCollection.new_query().gpus
                    ]
                    for stat in shared.all_gpus:
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
            if torch.cuda.is_available():
                logger.debug(f"Cleaning up GPU memory: {config.api.device}")

                with torch.device(config.api.device):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

        self.active_connections.remove(websocket)

    async def send_personal_message(self, data: Data, websocket: WebSocket):
        "Sends a data message to a specific websocket connection"

        await websocket.send_json(data.to_json())

    async def broadcast(self, data: Data):
        "Broadcasts data message to all active websocket connections"

        for connection in self.active_connections:
            if connection.application_state.CONNECTED:
                try:
                    await connection.send_json(data.to_json())
                except RuntimeError:
                    logger.debug("RuntimeError, removing connection")
                    try:
                        await connection.close()
                    except RuntimeError:
                        pass
                    self.active_connections.remove(connection)
            else:
                self.active_connections.remove(connection)

    def broadcast_sync(self, data: Data):
        "Broadcasts data message to all active websocket connections synchronously"

        loop_error_message = "No event loop found, please inject it in the code"

        try:
            assert self.loop is not None, loop_error_message
            asyncio.get_event_loop()
        except RuntimeError:
            assert self.loop is not None  # For type safety
            asyncio.set_event_loop(self.loop)
        except AssertionError:
            return

        for connection in self.active_connections:
            if connection.application_state.CONNECTED:
                asyncio.run_coroutine_threadsafe(
                    connection.send_json(data.to_json()), self.loop
                )
            else:
                self.active_connections.remove(connection)

    async def close_all(self):
        "Closes all active websocket connections"

        for connection in self.active_connections:
            await connection.close(reason="Server shutdown")

        if config.api.clear_memory_policy == "after_disconnect":
            if torch.cuda.is_available():
                logger.debug(f"Cleaning up GPU memory: {config.api.device}")

                with torch.cuda.device(config.api.device):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

        self.active_connections = []

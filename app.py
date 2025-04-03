import asyncio
import os
import threading
import tomllib
import uuid
import webbrowser
from datetime import datetime
from functools import partial
from json import dumps
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Task(BaseModel):
    id: str
    prompt: str
    created_at: datetime
    status: str
    steps: list = []
    max_step: int = 0  # 新增字段

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        data["created_at"] = self.created_at.isoformat()
        data["task_id"] = self.id  # 确保 task_id 字段存在
        return data

class TaskManager:
    MAX_HISTORY_SIZE = 100  # 最大历史任务数量

    def __init__(self):
        self.tasks = {}
        self.queues = {}
        self.history = []

    def create_task(self, prompt: str) -> Task:
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            prompt=prompt,
            created_at=datetime.now(),
            status="pending",
        )
        self.tasks[task_id] = task
        self.queues[task_id] = asyncio.Queue()
        return task

    async def require_confirmation(self, task_id: str, step_id: int):
        if task_id in self.tasks:
            task = self.tasks[task_id]
            for step in task.steps:
                if step.get("step") == step_id:
                    step["confirmation_required"] = True
                    break
            else:
                raise HTTPException(status_code=404, detail=f"Step {step_id} not found in task {task_id}")
            await self.queues[task_id].put(
                {"type": "status", "status": task.status, "steps": task.steps, "max_step": task.max_step}
            )

    async def confirm_task(self, task_id: str, step_id: int):
        if task_id in self.tasks:
            task = self.tasks[task_id]
            for step in task.steps:
                if step.get("step") == step_id:
                    if step.get("confirmation_required"):
                        step["confirmation_required"] = False
                        task.status = "running"
                        await self.queues[task_id].put(
                            {"type": "status", "status": task.status, "steps": task.steps, "max_step": task.max_step}
                        )
                        return
                    else:
                        raise HTTPException(status_code=400, detail=f"Step {step_id} does not require confirmation")
            else:
                raise HTTPException(status_code=404, detail=f"Step {step_id} not found in task {task_id}")
        else:
            raise HTTPException(status_code=404, detail="Task not found")

    async def update_task_step(
        self, task_id: str, step: int, result: str, step_type: str = "step"
    ):
        if task_id in self.tasks:
            task = self.tasks[task_id]
            current_step = step
            max_steps = task.max_step

            if "Executing step" in result:
                try:
                    parts = result.split("Executing step", 1)[1].strip().split("/")
                    current_step = int(parts[0])
                    max_steps = int(parts[1])
                    task.max_step = max_steps
                except Exception as e:
                    print(f"Error parsing step information: {e}")

            new_step = {
                "step": current_step,
                "result": result,
                "type": step_type,
                "confirmation_required": False
            }

            task.steps.append(new_step)
            await self.queues[task_id].put(
                {"type": step_type, "step": current_step, "result": result, "max_step": max_steps}
            )
            await self.queues[task_id].put(
                {"type": "status", "status": task.status, "steps": task.steps, "max_step": max_steps}
            )

    async def complete_task(self, task_id: str):
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = "completed"
            await self.queues[task_id].put(
                {"type": "status", "status": task.status, "steps": task.steps, "max_step": task.max_step}
            )
            await self.queues[task_id].put({"type": "complete"})
            self.history.append(task)
            del self.tasks[task_id]
            # 如果历史任务超过最大限制，则移除最旧的任务
            if len(self.history) > self.MAX_HISTORY_SIZE:
                self.history.pop(0)

    async def fail_task(self, task_id: str, error: str):
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = f"failed: {error}"
            await self.queues[task_id].put({"type": "error", "message": error, "max_step": task.max_step})
            self.history.append(task)
            del self.tasks[task_id]
            # 如果历史任务超过最大限制，则移除最旧的任务
            if len(self.history) > self.MAX_HISTORY_SIZE:
                self.history.pop(0)


task_manager = TaskManager()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/download")
async def download_file(file_path: str):
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, filename=os.path.basename(file_path))


@app.post("/tasks")
async def create_task(prompt: str = Body(..., embed=True)):
    task = task_manager.create_task(prompt)
    asyncio.create_task(run_task(task.id, prompt))
    return {"task_id": task.id}


@app.get("/history")
async def get_history():
    history_tasks = task_manager.history
    return JSONResponse(
        content=[task.model_dump() for task in history_tasks],  # 包含 task_id
        headers={"Content-Type": "application/json"},
    )

@app.post("/tasks/{task_id}/step/{step_id}/run")
async def confirm_task(task_id: str, step_id: int, confirmed: dict = Body(...)):
    if not confirmed.get("confirmed"):
        raise HTTPException(status_code=400, detail="Confirmation required")

    await task_manager.confirm_task(task_id, step_id)
    return {"message": f"Step {step_id} of task {task_id} confirmed and resumed"}


from app.agent.manus import Manus

async def run_task(task_id: str, prompt: str):
    try:
        task_manager.tasks[task_id].status = "running"

        agent = Manus(
            name="Manus",
            description="A versatile agent that can solve various tasks using multiple tools",
        )

        async def on_think(thought):
            await task_manager.update_task_step(task_id, 0, thought, "think")

        async def on_tool_execute(tool, input):
            if tool == "obdiag":  # obdiag 是需要确认的工具
                await task_manager.require_confirmation(task_id, 0)
                while any(step.get("confirmation_required") for step in task_manager.tasks[task_id].steps):
                    await asyncio.sleep(1)  # 等待用户确认
            await task_manager.update_task_step(
                task_id, 0, f"Executing tool: {tool}\nInput: {input}", "tool"
            )

        async def on_action(action):
            await task_manager.update_task_step(
                task_id, 0, f"Executing action: {action}", "act"
            )

        async def on_run(step, result):
            await task_manager.update_task_step(task_id, step, result, "run")

        from app.logger import logger

        class SSELogHandler:
            def __init__(self, task_id):
                self.task_id = task_id

            async def __call__(self, message):
                import re

                # Extract - Subsequent Content
                cleaned_message = re.sub(r"^.*? - ", "", message)

                event_type = "log"
                if "✨ Manus's thoughts:" in cleaned_message:
                    event_type = "think"
                elif "🛠️ Manus selected" in cleaned_message:
                    event_type = "tool"
                elif "🎯 Tool" in cleaned_message:
                    event_type = "act"
                elif "📝 Oops!" in cleaned_message:
                    event_type = "error"
                elif "🏁 Special tool" in cleaned_message:
                    event_type = "complete"

                await task_manager.update_task_step(
                    self.task_id, 0, cleaned_message, event_type
                )

        sse_handler = SSELogHandler(task_id)
        logger.add(sse_handler)
        result = await agent.run(prompt)
        await task_manager.update_task_step(task_id, 1, result, "result")
        await task_manager.complete_task(task_id)
    except Exception as e:
        await task_manager.fail_task(task_id, str(e))


@app.get("/tasks/{task_id}/events")
async def task_events(task_id: str):
    async def event_generator():
        if task_id not in task_manager.queues:
            yield f"event: error\ndata: {dumps({'message': 'Task not found', 'task_id': task_id})}\n\n"
            return

        queue = task_manager.queues[task_id]

        while True:
            try:
                event = await queue.get()
                event["task_id"] = task_id  # 添加 task_id 字段
                formatted_event = dumps(event)

                yield ": heartbeat\n\n"

                if event["type"] == "complete":
                    yield f"event: complete\ndata: {formatted_event}\n\n"
                    break
                elif event["type"] == "error":
                    yield f"event: error\ndata: {formatted_event}\n\n"
                elif event["type"] == "status":
                    yield f"event: status\ndata: {formatted_event}\n\n"
                else:
                    yield f"event: {event['type']}\ndata: {formatted_event}\n\n"

            except asyncio.CancelledError:
                print(f"Client disconnected for task {task_id}")
                break
            except Exception as e:
                print(f"Error in event stream: {str(e)}")
                yield f"event: error\ndata: {dumps({'message': str(e), 'task_id': task_id})}\n\n"
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/tasks")
async def get_tasks():
    sorted_tasks = sorted(
        task_manager.tasks.values(), key=lambda task: task.created_at, reverse=True
    )
    return JSONResponse(
        content=[task.model_dump() for task in sorted_tasks],  # 包含 task_id
        headers={"Content-Type": "application/json"},
    )

@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    if task_id not in task_manager.tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    task = task_manager.tasks[task_id]
    return JSONResponse(
        content=task.model_dump(),  # 包含 task_id
        headers={"Content-Type": "application/json"},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500, content={"message": f"Server error: {str(exc)}"}
    )


def open_local_browser(config):
    webbrowser.open_new_tab(f"http://{config['host']}:{config['port']}")


def load_config():
    try:
        config_path = Path(__file__).parent / "config" / "config.toml"

        with open(config_path, "rb") as f:
            config = tomllib.load(f)

        return {"host": config["server"]["host"], "port": config["server"]["port"]}
    except FileNotFoundError:
        raise RuntimeError(
            "Configuration file not found, please check if config/config.toml exists"
        )
    except KeyError as e:
        raise RuntimeError(
            f"The configuration file is missing necessary fields: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    config = load_config()
    open_with_config = partial(open_local_browser, config)
    threading.Timer(3, open_with_config).start()
    uvicorn.run(app, host=config["host"], port=config["port"])

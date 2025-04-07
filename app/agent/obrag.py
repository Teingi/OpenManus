from typing import List

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.prompt.obrag import SYSTEM_PROMPT, NEXT_STEP_PROMPT
from app.tool import Bash, Terminate, ToolCollection, OBRAG


class OBRAGAgent(ToolCallAgent):
    """An agent that implements the OBRAGAgent paradigm for executing obdiag command and natural conversations."""

    name: str = "obrag"
    description: str = "an autonomous AI programmer that interacts directly with the computer to solve obdiag tasks."

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    available_tools: ToolCollection = ToolCollection(
        Bash(), OBRAG(), Terminate()
    )
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])

    max_steps: int = 30

    bash: Bash = Field(default_factory=Bash)
    working_dir: str = "."

    async def think(self) -> bool:
        """Process current state and decide next action"""
        # Update working directory
        result = await self.bash.execute("pwd")
        self.working_dir = result.output
        self.next_step_prompt = self.next_step_prompt.format(
            current_dir=self.working_dir
        )

        return await super().think()

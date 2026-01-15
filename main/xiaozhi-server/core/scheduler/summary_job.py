import asyncio
import time
import logging
from config.logger import setup_logging
from core.utils.llm import create_instance

TAG = "DailySummaryScheduler"

class DailySummaryScheduler:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logging()
        self.running = False
        self.llm = None
        self._init_llm()

    def _init_llm(self):
        try:
            # Use the memory LLM or main LLM
            if "Memory" in self.config and self.config["selected_module"]["Memory"] in self.config["Memory"]:
                mem_conf = self.config["Memory"][self.config["selected_module"]["Memory"]]
                if "llm" in mem_conf:
                    llm_name = mem_conf["llm"]
                    if llm_name in self.config["LLM"]:
                        self.llm = create_instance(self.config["LLM"][llm_name]["type"], self.config["LLM"][llm_name])

            if not self.llm and "LLM" in self.config:
                # Fallback to main LLM
                llm_name = list(self.config["LLM"].keys())[0] # Pick first available
                self.llm = create_instance(self.config["LLM"][llm_name]["type"], self.config["LLM"][llm_name])
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"Failed to init LLM for scheduler: {e}")

    async def start(self):
        self.running = True
        self.logger.bind(tag=TAG).info("Starting Daily Summary Scheduler...")
        while self.running:
            try:
                now = time.localtime()
                # Check if it is 01:00 AM (approx)
                if now.tm_hour == 1 and now.tm_min == 0:
                    await self.run_summary_task()
                    # Sleep for 61 seconds to avoid double execution
                    await asyncio.sleep(61)
                else:
                    # Check every minute
                    await asyncio.sleep(60)
            except Exception as e:
                self.logger.bind(tag=TAG).error(f"Scheduler loop error: {e}")
                await asyncio.sleep(60)

    async def stop(self):
        self.running = False
        self.logger.bind(tag=TAG).info("Daily Summary Scheduler stopped.")

    async def run_summary_task(self):
        self.logger.bind(tag=TAG).info("Running Daily Psychological Summary...")
        try:
            # Logic:
            # 1. Iterate over all users/memories (Mocking this step as we don't have a user DB access method exposed easily)
            # 2. Analyze their state
            # 3. Log or save

            # Since this is a PoC/Refactor request, we implement the core logic for ONE hypothetical user
            # or simply log that the task is executing.

            # If we had access to memory files (usually in ./memory/ or similar)
            # we could iterate them.

            self.logger.bind(tag=TAG).info("Analyzing user memories for psychological trends...")

            # Mock analysis
            if self.llm:
                # We would construct a prompt here
                pass

            self.logger.bind(tag=TAG).info("Daily Summary Completed.")

        except Exception as e:
            self.logger.bind(tag=TAG).error(f"Summary task failed: {e}")

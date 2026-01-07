from langgraph.store.memory import InMemoryStore
from typing import Dict, Any, List
import uuid
import datetime

# Singleton store for demonstration purposes
# In production, this would be backed by Postgres or Redis
_global_store = InMemoryStore()

def get_store():
    return _global_store

class SummaryManager:
    """
    Manages the periodic summary of user interactions.
    """
    def __init__(self, store: InMemoryStore):
        self.store = store
        self.namespace_prefix = "user_profile"

    async def update_user_profile(self, user_id: str, new_summary: str):
        """
        Updates the user's profile in the store.
        """
        namespace = (self.namespace_prefix, user_id)
        # We can store the raw summary or a structured object
        await self.store.aput(namespace, "summary", {"content": new_summary, "updated_at": str(datetime.datetime.now())})

    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieves the user's profile.
        """
        namespace = (self.namespace_prefix, user_id)
        item = await self.store.aget(namespace, "summary")
        if item:
            return item.value
        return {}

# --- Periodic Task Simulation ---

async def run_daily_summary(user_id: str, conversation_history: List[str]):
    """
    Simulates the daily summary task.
    This would be triggered by a Cron job.
    """
    store = get_store()
    manager = SummaryManager(store)

    # In a real app, we would use an LLM to summarize the history
    # For now, we simulate the summary generation
    print(f"Running daily summary for user {user_id}...")

    summary_text = f"User discussed: {', '.join(conversation_history[:3])}..."
    await manager.update_user_profile(user_id, summary_text)

    print(f"Updated profile for {user_id}: {summary_text}")

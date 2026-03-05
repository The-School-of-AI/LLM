from llm.logger import Logger


class NoOpLogger(Logger):
    def log_step(
        self,
        step: int,
        metrics: dict,
        context: dict | None = None,
    ):
        pass

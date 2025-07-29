from opentelemetry import trace

from agents.analyzer import AnalyzerAgent, AnalyzerAgentConfig
from utils.repo import get_repo_version

from .base_handler import BaseHandler, BaseHandlerConfig


class AnalyzeHandlerConfig(BaseHandlerConfig, AnalyzerAgentConfig):
    pass


class AnalyzeHandler(BaseHandler):
    def __init__(self, config: AnalyzeHandlerConfig):
        super().__init__(config)

        self.agent = AnalyzerAgent(config)

    async def handle(self):
        with trace.get_tracer("analyzer").start_as_current_span("Analyzer Agent") as span:
            span.set_attributes(
                {
                    "repo_path": str(self.config.repo_path),
                    "repo_version": get_repo_version(self.config.repo_path),
                    "exclude_code_structure": self.config.exclude_code_structure,
                    "exclude_data_flow": self.config.exclude_data_flow,
                    "exclude_dependencies": self.config.exclude_dependencies,
                    "exclude_request_flow": self.config.exclude_request_flow,
                    "exclude_api_analysis": self.config.exclude_api_analysis,
                    "exclude_business_logic": self.config.exclude_business_logic,
                    "input": str(self.config.repo_path),
                }
            )
            result = await self.agent.run()

            return result

import json

# Agent Engine imports
from agent_engine.utils import get_relative_path_from_current_file
from agent_engine.agent_logger import AgentLogger

logger = AgentLogger(__name__)

class RisksTable:
    def __init__(self):
        with open(get_relative_path_from_current_file('risks_table.json'), "r", encoding="utf-8") as f:
            self.table = json.load(f)
        self.categories = self._build_categories()
        self.risks = self._build_risks()

    def _build_categories(self) -> list[str]:
        categories = []
        for _system, _categories in self.table.items():
            categories.extend(_categories.keys())
        return categories

    def _build_risks(self) -> list[str]:
        risks = []
        for _system, _categories in self.table.items():
            for _category, _risks in _categories.items():
                risks.extend(_risks)
        return risks

    def get_categories(self, systems: list[str]) -> list[str]:
        categories = []
        for system in systems:
            categories.extend(self.table[system].keys())
        return categories

    def get_risks(self, categories: list[str]) -> list[str]:
        risks = []
        for category in categories:
            for _system, _categories in self.table.items():
                if category in _categories:
                    risks.extend(_categories[category])
        return risks


if __name__ == "__main__":
    risks_table = RisksTable()
    print(len(risks_table.categories))
    print(len(risks_table.risks))
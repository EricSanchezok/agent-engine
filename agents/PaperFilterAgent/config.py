import json

# A2A framework imports 
from a2a.types import AgentCard, AgentSkill, AgentCapabilities, AgentProvider

# AgentEngine imports
from agent_engine.utils import get_local_ip


AGENT_NAME = 'PaperFilterAgent'
AGENT_DESCRIPTION = '接收一个论文元数据列表，并根据更高级的语义标准或用户特定指令进行智能筛选、排序和推荐，最终输出一个精简的论文 ID 列表。'
AGENT_VERSION = '0.1.0'
PROVIDER = AgentProvider(
    organization='EricSanchez',
    url='https://github.com/EricSanchezok'
)

AGENT_HOST = get_local_ip()
AGENT_PORT = 9901
AGENT_URL = f'http://{AGENT_HOST}:{AGENT_PORT}/'

INPUT_MODES = ['application/json'] 
OUTPUT_MODES = ['application/json']

CAPABILITIES = AgentCapabilities(
    streaming=False, 
    pushNotifications=False,
    stateTransitionHistory=False
)

AGENT_SKILLS = [
    AgentSkill(
        id='filter_and_recommend',
        name='论文筛选与推荐',
        name_en='Filter and Recommend Papers',
        description=(
            '接收一个JSON对象作为输入，其中包含 `arxiv_ids` (ArXiv ID的列表)。'
            '代理内部会根据这些ID获取完整的论文元数据，并基于预设的内部逻辑进行智能筛选和排序。'
            '最终返回一个JSON格式的列表，其中包含推荐论文的ArXiv ID，列表的顺序即为推荐顺序。'
            '可以额外传入 `max_recommendations` (整数) 来限制推荐数量，若不提供则默认为16。'
        ),
        description_en=(
            'Receives a JSON object as input, containing `paper_ids` (a list of ArXiv IDs). '
            'The agent will internally fetch the full paper metadata for these IDs and perform intelligent '
            'filtering and ranking based on its preset internal logic. '
            'It returns a JSON-formatted list of recommended ArXiv IDs, and the order of the list represents the recommendation order. '
            'An optional `max_recommendations` (integer) can be passed to limit the number of recommendations (defaults to 16).'
        ),
        tags=['filter', 'recommend', 'ranking', '筛选', '推荐', '排序', '智能分析'],
        examples=[
            json.dumps({
                "arxiv_ids": [
                    "2508.12345",
                    "2507.67890",
                    "2506.11223"
                ],
                "max_recommendations": 2
            }, indent=4, ensure_ascii=False)
        ],
    )
]

AGENT_CARD = AgentCard(
    name=AGENT_NAME,
    description=AGENT_DESCRIPTION,
    version=AGENT_VERSION,
    url=AGENT_URL,
    defaultInputModes=INPUT_MODES,
    defaultOutputModes=OUTPUT_MODES,
    capabilities=CAPABILITIES,
    skills=AGENT_SKILLS,
    provider=PROVIDER
)

__all__ = [
    'AGENT_NAME',
    'AGENT_DESCRIPTION',
    'AGENT_VERSION',
    'AGENT_HOST',
    'AGENT_PORT',
    'AGENT_URL',
    'INPUT_MODES',
    'OUTPUT_MODES',
    'AGENT_SKILLS',
    'AGENT_CARD',
]

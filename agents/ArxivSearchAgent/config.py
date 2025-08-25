import json

# A2A framework imports 
from a2a.types import AgentCard, AgentSkill, AgentCapabilities, AgentProvider

# AgentEngine imports
from agent_engine.utils import get_local_ip, get_current_file_dir

LOG_DIR = get_current_file_dir() / 'logs'

AGENT_NAME = 'ArxivSearchAgent'
AGENT_DESCRIPTION = '根据结构化查询 (JSON) 或自然语言指令 (Text)，在 Arxiv 数据库上执行文献检索，并返回原始的论文元数据列表。'
AGENT_VERSION = '0.1.1' # Incremented version
PROVIDER = AgentProvider(
    organization='EricSanchez',
    url='https://github.com/EricSanchezok'
)

AGENT_HOST = get_local_ip()
AGENT_PORT = 9900
AGENT_URL = f'http://{AGENT_HOST}:{AGENT_PORT}/'

INPUT_MODES = ['text/plain'] 
OUTPUT_MODES = ['application/json']

CAPABILITIES = AgentCapabilities(
    streaming=False, 
    pushNotifications=False,
    stateTransitionHistory=False
)

AGENT_SKILLS = [
    AgentSkill(
        id='search_papers_with_query_string',
        name='ArXiv原生字符串搜索',
        name_en='Search Papers with ArXiv Query String',
        description=(
            '通过ArXiv API支持的原生查询字符串执行高级文献检索。这允许使用布尔运算符（AND, OR, ANDNOT）、'
            '字段限定符（如 ti: 表示标题, cat: 表示分类）和日期范围（submittedDate:[YYYYMMDD TO YYMMDD]）进行复杂查询。'
            '此技能适用于需要构建精细、复杂查询逻辑的场景。'
            '执行成功后，返回一个JSON格式的字符串，其中包含所有检索到的论文元数据（标题、作者、摘要等）的列表。'
        ),
        description_en=(
            'Performs an advanced literature search using a native query string supported by the ArXiv API. '
            'This allows for complex queries using Boolean operators (AND, OR, ANDNOT), field prefixes '
            '(e.g., ti: for title, cat: for category), and date ranges (submittedDate:[YYYYMMDD TO YYYYMMDD]). '
            'This skill is suitable for scenarios requiring the construction of fine-grained, complex query logic. '
            'Upon successful execution, it returns a JSON-formatted string containing a list of all retrieved paper metadata (title, authors, summary, etc.).'
        ),
        tags=['arxiv', 'query string', 'search', 'advanced search', '学术', '论文检索', '高级搜索'],
        examples=[
            '(ti:"artificial intelligence" OR cat:cs.AI) AND submittedDate:[20250802 TO 20250803]'
        ],
    ),
    AgentSkill(
        id='search_papers_with_text',
        name='自然语言论文搜索',
        name_en='Search Papers with Text',
        description=(
            '接收一句自然语言指令（如 "昨天的人工智能论文"），在内部利用自然语言处理技术将其解析为一个结构化的JSON查询对象，'
            '然后执行搜索。此技能非常适合直接与终端用户交互或处理非结构化文本输入的场景。'
            '执行成功后，返回一个JSON格式的字符串，其中包含所有检索到的论文元数据（标题、作者、摘要等）的列表。'
        ),
        description_en=(
            'Receives a natural language command (e.g., "AI papers from yesterday"), internally parses it into a '
            'structured JSON query object using NLP techniques, and then executes the search. This skill is ideal '
            'for direct user interaction or handling unstructured text input. '
            'Upon successful execution, it returns a JSON-formatted string containing a list of all retrieved paper metadata (title, authors, summary, etc.).'
        ),
        tags=['arxiv', 'text', 'nlp', 'natural language', '学术', '论文检索', '模糊搜索'],
        examples=[
            '昨天人工智能领域的论文',
            'find me recent papers on multi-agent systems',
            '查找2025年8月关于多智能体系统的论文'
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

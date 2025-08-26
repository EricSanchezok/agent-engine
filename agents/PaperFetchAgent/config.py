import json
import base64

# A2A framework imports 
from a2a.types import AgentCard, AgentSkill, AgentCapabilities, AgentProvider

# AgentEngine imports
from agent_engine.utils import get_local_ip, get_current_file_dir

LOG_DIR = get_current_file_dir() / 'logs'

AGENT_NAME = 'PaperFetchAgent'
AGENT_DESCRIPTION = '根据输入的论文 ID 列表，获取并下载这些论文的 PDF 实体文件。该代理专注于处理 I/O 操作。'
AGENT_VERSION = '0.1.0'
PROVIDER = AgentProvider(
    organization='EricSanchez',
    url='https://github.com/EricSanchezok'
)

AGENT_HOST = get_local_ip()
AGENT_PORT = 9902
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
        id='fetch_papers_as_pdf',
        name='获取论文PDF',
        name_en='Fetch Papers as PDF',
        description=(
            '接收一个JSON对象作为输入，其中包含 `arxiv_ids` (ArXiv ID的列表)。'
            '代理会下载列表中每篇论文的PDF文件，并将其内容进行Base64编码。'
            '最终返回一个JSON格式的列表，其中每一项都是对应论文PDF文件的Base64编码字符串。'
        ),
        description_en=(
            'Receives a JSON object as input, containing `arxiv_ids` (a list of ArXiv IDs). '
            'The agent downloads the PDF file for each paper in the list and encodes its content into Base64. '
            'It returns a JSON-formatted list where each item is the Base64 encoded string of the corresponding paper\'s PDF file.'
        ),
        tags=['fetch', 'download', 'pdf', 'base64', '获取', '下载', '文件', 'IO'],
        examples=[
            json.dumps({
                "arxiv_ids": [
                    "2508.12345",
                    "2507.67890"
                ]
            }, indent=4, ensure_ascii=False)
        ],
    ),
    AgentSkill(
        id='extract_and_fetch_pdfs',
        name='从文本中提取ID并获取PDF',
        name_en='Extract IDs and Fetch Papers as PDF from Text',
        description=(
            '接收一段包含一个或多个ArXiv论文ID的自然语言文本。'
            '代理会自动从文本中提取所有有效的ArXiv ID，下载对应论文的PDF文件，并将其内容进行Base64编码。'
            '最终返回一个JSON格式的列表，其中每一项都是对应论文PDF文件的Base64编码字符串。'
        ),
        description_en=(
            'Receives a natural language text string containing one or more ArXiv paper IDs. '
            'The agent automatically extracts all valid ArXiv IDs from the text, downloads the PDF file for each paper, and encodes its content into Base64. '
            'It returns a JSON-formatted list where each item is the Base64 encoded string of the corresponding paper\'s PDF file.'
        ),
        tags=['fetch', 'download', 'pdf', 'base64', '获取', '下载', '文件', 'IO', 'extract', '提取', 'NLP'],
        examples=[
            "你好，能帮我下载一下 2508.12345 和 2507.67890 这两篇论文的PDF文件吗？"
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

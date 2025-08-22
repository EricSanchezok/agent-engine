import json

# A2A framework imports 
from a2a.types import AgentCard, AgentSkill, AgentCapabilities, AgentProvider

# AgentEngine imports
from agent_engine.utils import get_local_ip


AGENT_NAME = 'PaperAnalysisAgent'
AGENT_DESCRIPTION = '对输入的论文 PDF 文件进行深度分析和内容总结，根据用户要求生成结构化的分析报告。'
AGENT_VERSION = '0.1.0'
PROVIDER = AgentProvider(
    organization='EricSanchez',
    url='https://github.com/EricSanchezok'
)

AGENT_HOST = get_local_ip()
AGENT_PORT = 9903
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
        id='generate_analysis_report',
        name='生成论文解读报告',
        name_en='Generate Paper Analysis Report',
        description=(
            '接收一个JSON对象作为输入，其中包含 `papers_base64` (一个由PDF文件的Base64编码字符串组成的列表)。'
            '代理会对列表中的每一份PDF进行深度内容分析和总结。'
            '最终返回一个JSON格式的列表，其中每一项都是对应论文的Markdown格式解读报告。'
        ),
        description_en=(
            'Receives a JSON object as input, containing `papers_base64` (a list of Base64 encoded strings from PDF files). '
            'The agent performs an in-depth content analysis and summarization for each PDF in the list. '
            'It returns a JSON-formatted list where each item is the analysis report in Markdown format for the corresponding paper.'
        ),
        tags=['analysis', 'summary', 'markdown', 'pdf', '解读', '总结', '报告', 'NLP'],
        examples=[
            json.dumps({
                "papers_base64": [
                    "JVBERi0xLjcKJeLjz9MKMSAwIG9iago8PC9UeXBlL0NhdGFsb2cvUGFnZXMgMiAwIFIvTGF...",
                    "JVBERi0xLjcKJeLjz9MKMSAwIG9iago8PC9UeXBlL0NhdGFsb2cvUGFnZXMgMiAwIFIvTGF..."
                ]
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

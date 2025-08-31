import json

# A2A framework imports 
from a2a.types import AgentCard, AgentSkill, AgentCapabilities, AgentProvider

# AgentEngine imports
from agent_engine.utils import get_local_ip

AGENT_NAME = 'PaperAnalysisAgent'
AGENT_DESCRIPTION = '可根据用户提供的 ArXiv 论文 ID 或直接分析消息中附加的 PDF 文件，进行深度内容分析和总结，并生成结构化的解读报告。'
AGENT_VERSION = '0.1.0'
PROVIDER = AgentProvider(
    organization='EricSanchez',
    url='https://github.com/EricSanchezok'
)

AGENT_HOST = get_local_ip()
AGENT_PORT = 9903
AGENT_URL = f'http://{AGENT_HOST}:{AGENT_PORT}/'

INPUT_MODES = ['text/plain', 'application/json'] 
OUTPUT_MODES = ['application/json']

CAPABILITIES = AgentCapabilities(
    streaming=False, 
    pushNotifications=False,
    stateTransitionHistory=False
)

AGENT_SKILLS = [
    AgentSkill(
        id='analyze_by_arxiv_ids',
        name='通过ArXiv ID解读论文',
        name_en='Analyze Papers by ArXiv IDs',
        description=(
            '接收一个包含ArXiv论文ID列表的JSON对象，专门用于解读ArXiv平台上的论文。'
            '代理会根据ID自动获取对应的论文并进行深度分析和总结。'
            '最终返回一个JSON格式的列表，其中每一项都是对应论文的Markdown格式解读报告。'
        ),
        description_en=(
            'Receives a JSON object containing a list of ArXiv paper IDs, specialized for analyzing papers from the ArXiv platform. '
            'The agent automatically fetches the corresponding papers based on the IDs and performs an in-depth analysis and summarization. '
            'It returns a JSON-formatted list where each item is the analysis report in Markdown format for the corresponding paper.'
        ),
        tags=['analysis', 'summary', 'markdown', 'pdf', 'arxiv', '解读', '总结', '报告', 'NLP', '论文'],
        examples=[
            json.dumps({
                "arxiv_ids": [
                    "2402.19473",
                    "2305.10601"
                ]
            }, indent=4, ensure_ascii=False)
        ],
    ),
    AgentSkill(
        id='extract_and_analyze_arxiv_ids',
        name='从文本中提取并解读ArXiv论文',
        name_en='Extract and Analyze ArXiv Papers from Text',
        description=(
            '接收一段包含一个或多个ArXiv论文ID的文本。'
            '代理会自动从文本中提取所有有效的ArXiv ID，获取对应的论文并进行深度分析和总结。'
            '最终返回一个JSON格式的列表，其中每一项都是对应论文的Markdown格式解读报告。'
            '这个工具非常适合当您在一段话中提及多篇论文时使用。'
        ),
        description_en=(
            'Receives a text string containing one or more ArXiv paper IDs. '
            'The agent automatically extracts all valid ArXiv IDs from the text, fetches the corresponding papers, and performs an in-depth analysis and summarization. '
            'It returns a JSON-formatted list where each item is the analysis report in Markdown format for the corresponding paper. '
            'This tool is ideal for when you mention multiple papers within a block of text.'
        ),
        tags=['analysis', 'summary', 'markdown', 'pdf', 'arxiv', '解读', '总结', '报告', 'NLP', '论文', 'extract', '提取'],
        examples=[
            "你好，请帮我深入分析一下这几篇重要的AI论文：2402.19473 (关于Jamba模型) 和 2305.10601，谢谢！"
        ],
    ),
    AgentSkill(
        id='analyze_from_message_files',
        name='解读附件中的论文PDF',
        name_en='Analyze Paper PDFs from Message Attachments',
        description=(
            '此技能用于解读直接附加在A2A消息filepart中的论文PDF文件。'
            '用户需要在消息中附加一个或多个论文PDF，并发送一条简单的文本指令。'
            '代理会读取附件中的所有论文文件，进行深度内容分析和总结，并返回一个JSON格式的列表，其中每一项都是对应论文的Markdown格式解读报告。'
        ),
        description_en=(
            'This skill analyzes paper PDF files attached directly in the filepart of an A2A message. '
            'The user needs to attach one or more paper PDFs to the message and send a simple text instruction. '
            'The agent will read all attached paper files, perform an in-depth content analysis and summarization, and return a JSON-formatted list where each item is the analysis report in Markdown format for the corresponding paper.'
        ),
        tags=['analysis', 'summary', 'markdown', 'pdf', 'file', 'attachment', '解读', '总结', '报告', 'NLP', '论文'],
        examples=[
            '帮我分析一下附件里的这几篇论文',
            'Summarize the attached papers',
            '对这几个论文文件进行内容总结和分析'
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

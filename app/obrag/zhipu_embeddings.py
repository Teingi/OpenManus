import os
import dotenv

from app.config import config

dotenv.load_dotenv()
from langchain_openai import OpenAIEmbeddings


tr = {
    "en": {
        "hello": "Hello, what can I do for you?",
        "title": "💬 Intelligent Q&A Assistant",
        "caption": "🚀 An intelligent Q&A robot built using OceanBase vector retrieval features and large language model capabilities",
        "setting": "🔧 Settings",
        "table_name_input": "Table Name",
        "table_name_help": "The table name used to store document and vector data, set using the environment variable TABLE_NAME",
        "llm_model": "LLM",
        "llm_model_help": "The large language model used for response generation, set using the environment variable LLM_MODEL",
        "chat_history_len": "Chat History Length",
        "chat_history_len_help": "The length of chat history used for context understanding",
        "search_docs": "Search Documents",
        "search_docs_help": "Retrieve documents to provide more information, otherwise only use the large language model to answer questions",
        "oceanbase_only": "OceanBase Only",
        "oceanbase_only_help": "If checked, the robot will only answer questions related to OceanBase",
        "rerank": "Rerank Documents",
        "rerank_help": "Rerank retrieved documents using BGE-M3 to improve the quality of search results, this is a slow process, only use when necessary",
        "time_elapse": " (elapsed {:.2f}s)",
        "analyzing_intent": "Analyzing the intent of the question...",
        "no_oceanbase": "This question is not related to OceanBase.",
        "analyzing_components": "Analyzing the related OceanBase components of the question...",
        "list_related_components": "The related OceanBase components are: {}",
        "embedding_query": "Embedding the question into vector with deep learning model...",
        "searching_docs_for": "Searching for related documents of {} with OceanBase...",
        "searching_docs": "Searching for related documents with OceanBase...",
        "reranking_docs": "Reranking the retrieved documents with BGE-M3...",
        "llm_thinking": "The LLM is thinking...",
        "processing": "Processing...",
        "finish_thinking": "Finish thinking!",
        "ref_tips": "Retrieved documents are listed below,",
        "chat_placeholder": "Input your question here...",
        "lang_input": "Language",
        "lang_help": "The language used for LLM prompts and the UI, set using the environment variable UI_LANG, currently supports 'en' and 'zh'",
        "show_refs": "Show References",
    },
    "zh": {
        "hello": "您好，请问有什么可以帮助您的吗？",
        "title": "💬 智能问答助手",
        "caption": "🚀 使用 OceanBase 向量检索特性和大语言模型能力构建的智能问答机器人",
        "setting": "🔧 设置",
        "table_name_input": "表名",
        "table_name_help": "用于存放文档及其向量数据的表名，用环境变量 TABLE_NAME 进行设置",
        "llm_model": "大语言模型",
        "llm_model_help": "用于回答问题的大语言模型，用环境变量 LLM_MODEL 进行设置",
        "chat_history_len": "聊天历史长度",
        "chat_history_len_help": "用于上下文理解的聊天历史长度",
        "search_docs": "进行文档检索",
        "search_docs_help": "检索文档以提供更多信息，否则仅使用大语言模型回答问题",
        "oceanbase_only": "仅限 OceanBase 相关问题",
        "oceanbase_only_help": "如果选中，机器人将仅回答与 OceanBase 相关的问题",
        "rerank": "对文档进行重新排序",
        "rerank_help": "使用 BGE-M3 对检索到的文档进行重新排序以提高搜索结果的质量，这是一个缓慢的过程，仅在必要时使用",
        "time_elapse": "（耗时 {:.2f} 秒）",
        "analyzing_intent": "正在分析问题的意图...",
        "no_oceanbase": "这个问题与 OceanBase 无关。",
        "analyzing_components": "正在分析问题涉及的 OceanBase 组件...",
        "list_related_components": "问题涉及的 OceanBase 组件有：{}",
        "embedding_query": "正在使用深度学习模型将提问内容嵌入为向量...",
        "searching_docs_for": "正在使用 OceanBase 检索 {} 的相关文档...",
        "searching_docs": "正在使用 OceanBase 检索相关文档...",
        "reranking_docs": "正在使用 BGE-M3 对检索到的文档进行重新排序...",
        "llm_thinking": "大语言模型正在思考...",
        "processing": "处理中...",
        "finish_thinking": "思考完成！",
        "ref_tips": "根据向量相似性匹配检索到的相关文档如下:",
        "chat_placeholder": "请输入您想咨询的问题...",
        "lang_input": "语言",
        "lang_help": "大模型提示词和用户界面的语言，用环境变量 UI_LANG 进行设置，支持 en 和 zh 两种语言",
        "show_refs": "显示参考文档",
    },
}


def t(key: str, lang="en", *args) -> str:
    if len(args) > 0:
        return tr[lang].get(key, "").format(*args)
    return tr[lang].get(key, "")
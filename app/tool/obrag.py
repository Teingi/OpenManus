
from langchain_core.messages import AIMessageChunk

from app.tool import BaseTool


from app.obrag.doc_rag import doc_rag_stream
from app.tool.base import ToolResult, CLIResult


class OBRAG(BaseTool):
    name = "OBRAG"
    description = "从OceanBase向量库中检索文档"

    def query_qe(self,query):
        History = []
        oceanbase_only = True
        rerank = False
        search_docs = True
        lang = "zh"
        show_refs = True
        llm_model = "deepseek-v3"
        res = doc_rag_stream(
            query=query,
            chat_history=History,
            universal_rag=not oceanbase_only,
            rerank=rerank,
            llm_model=llm_model,
            search_docs=search_docs,
            lang=lang,
            show_refs=show_refs,
        )
        for chunk in res:
            if isinstance(chunk, AIMessageChunk):
                yield chunk.content, None
            else:
                yield None, chunk

    async def execute(self, query: str) -> ToolResult:
        # 调用doc_rag进行检索
        try:
            info = self.query_qe("obd的使用方法")
            chunk_all = ""
            content_all = ""
            for i in info:
                content = i[0]
                chunk = i[1]
                if content is not None:
                    content_all += content
                if chunk is not None:
                    chunk_all += chunk
            return CLIResult(system=chunk_all, output=content_all)
        except Exception as e:
            return ToolResult(error=f"Error executing tool: {str(e)}")




if __name__ == "__main__":
    rag = OBRAG()
    result=rag.query_qe("obdiag的使用方法")
    chunk_all = ""
    content_all = ""
    for i in result:
        content = i[0]
        chunk = i[1]
        if content is not None:
            content_all += content
        if chunk is not None:
            chunk_all += chunk
    print("chunk_all:", chunk_all)
    print("content_all:", content_all)



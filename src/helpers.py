import asyncio
import httpx
from typing import TYPE_CHECKING

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

if TYPE_CHECKING:
    from ..main import RAGFlowAdapterPlugin
    from astrbot.api.provider import ProviderRequest


def mask_sensitive_info(info: str, keep_last: int = 6) -> str:
    """隐藏敏感信息，只显示最后几位。"""
    if not isinstance(info, str) or len(info) <= keep_last:
        return info
    return f"******{info[-keep_last:]}"


def inject_content_into_request(
    plugin: "RAGFlowAdapterPlugin", req: "ProviderRequest", content: str
):
    """
    根据配置，将指定内容注入到 ProviderRequest 对象中。
    """
    if not content:
        return

    # 统一的 RAG 内容模板
    rag_prompt_template = (
        f"--- 以下是参考资料 ---\n{content}\n--- 请根据以上资料回答问题，并在回答中注明参考的来源 ---"
    )

    if plugin.rag_injection_method == "user_prompt":
        req.prompt = f"{rag_prompt_template}\n\n{req.prompt}"
        logger.debug("RAG content injected into user_prompt.")
    elif plugin.rag_injection_method == "insert_system_prompt":
        # 插入到倒数第二的位置，确保在用户最新消息之前
        req.contexts.insert(-1, {"role": "system", "content": rag_prompt_template})
        logger.debug("RAG content inserted as a new system message.")
    else:  # 默认为 system_prompt
        if req.system_prompt:
            req.system_prompt = f"{req.system_prompt}\n\n{rag_prompt_template}"
        else:
            req.system_prompt = rag_prompt_template
        logger.debug("RAG content injected into system_prompt.")


async def query_ragflow(plugin: "RAGFlowAdapterPlugin", query: str) -> str:
    """
    使用给定的查询与 RAGFlow API 进行交互，并返回拼接好的上下文。
    """
    if not all(
        [plugin.ragflow_base_url, plugin.ragflow_api_key, plugin.ragflow_kb_ids]
    ):
        logger.warning("RAGFlow 未完全配置，跳过检索。")
        return ""

    url = f"{plugin.ragflow_base_url.rstrip('/')}/api/v1/retrieval"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {plugin.ragflow_api_key}",
    }
    data = {
        "question": query,
        "dataset_ids": plugin.ragflow_kb_ids,
        "top_k": 1024,
        "similarity_threshold": 0.3,
        "vector_similarity_weight": 0.3,
        "cross_languages": plugin.ragflow_cross_lang,
        "keyword": True,
        "highlight": True,
    }

    if plugin.ragflow_rerank_model:
        data["rerank_id"] = plugin.ragflow_rerank_model

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data, timeout=30.0)
            response.raise_for_status()

            api_data = response.json()
            if api_data.get("code") != 0:
                logger.error(f"RAGFlow API 返回错误: {api_data}")
                return ""

            chunks = api_data.get("data", {}).get("chunks", [])
            if not chunks:
                logger.info("RAGFlow 未检索到相关内容。")
                return ""

            # 提取内容并附上来源标注
            # RAGFlow API 返回的 chunk 中文档名称可能是 document_keyword 或 document_name
            content_with_sources = []
            seen_sources = set()
            for chunk in chunks:
                content = chunk.get("content", "")
                # 尝试获取文档名称（兼容不同 API 响应格式）
                source_name = (
                    chunk.get("document_keyword")
                    or chunk.get("document_name")
                    or chunk.get("document_id", "未知来源")
                )
                if not source_name:
                    source_name = "未知来源"

                # 构建带来源的内容
                content_with_sources.append(f"[来源: {source_name}]\n{content}")
                seen_sources.add(source_name)

            retrieved_content = "\n\n".join(content_with_sources)
            logger.info(f"成功从 RAGFlow 检索到 {len(chunks)} 条内容，来源: {list(seen_sources)}")
            logger.debug(f"检索到的内容: \n{retrieved_content}")
            return retrieved_content

    except httpx.RequestError as e:
        logger.error(f"请求 RAGFlow API 时出错: {e}", exc_info=True)
        return ""
    except Exception as e:
        logger.error(f"处理 RAGFlow 响应时发生未知错误: {e}", exc_info=True)
        return ""


async def archive_conversation(plugin: "RAGFlowAdapterPlugin", event: AstrMessageEvent):
    """
    将当前会话的近期对话历史进行归档。
    """
    logger.info(f"触发了会话 {event.get_session_id()} 的归档流程。")

    try:
        # 获取会话ID
        session_id = event.get_session_id()

        # 获取当前对话ID
        conversation_id = (
            await plugin.context.conversation_manager.get_curr_conversation_id(
                session_id
            )
        )
        if not conversation_id:
            logger.warning(f"会话 {session_id} 没有找到对应的对话ID，跳过归档。")
            return

        # 获取对话对象
        conversation = await plugin.context.conversation_manager.get_conversation(
            session_id, conversation_id
        )
        if not conversation:
            logger.warning(
                f"会话 {session_id} 的对话 {conversation_id} 不存在，跳过归档。"
            )
            return

        # 解析历史记录
        import json

        history = json.loads(conversation.history)

        # 获取最近的 rag_archive_threshold 条消息
        threshold = plugin.rag_archive_threshold
        recent_messages = history[-threshold:] if len(history) > threshold else history

        logger.info(
            f"会话 {session_id} 准备归档最近 {len(recent_messages)} 条消息（阈值: {threshold}）："
        )

        # 打印每条消息
        for i, msg in enumerate(recent_messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # 截断过长的内容以便日志显示
            display_content = content[:200] + "..." if len(content) > 200 else content
            logger.info(f"  [{i + 1}] {role}: {display_content}")

    except Exception as e:
        logger.error(
            f"获取会话 {event.get_session_id()} 的对话历史时发生错误: {e}",
            exc_info=True,
        )

    await asyncio.sleep(1)  # 模拟异步操作

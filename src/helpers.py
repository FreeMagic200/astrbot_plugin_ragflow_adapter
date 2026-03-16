import asyncio
import hashlib
import httpx
from typing import TYPE_CHECKING

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

if TYPE_CHECKING:
    from ..main import RAGFlowAdapterPlugin
    from astrbot.api.provider import ProviderRequest

# 哨兵值：RAGFlow API 调用成功但未检索到相关内容
RETRIEVAL_NO_RESULTS = "__RETRIEVAL_NO_RESULTS__"


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
        f"--- 以下是从知识库检索到的参考资料 ---\n{content}\n--- 参考资料结束 ---\n\n"
        "请根据以上参考资料回答用户问题。注意：\n"
        "1. 优先引用参考资料中的原文内容作答，并在回答中注明来源文档名称\n"
        "2. 如参考资料未覆盖某知识点，请明确注明【该内容未在知识库中检索到，以下基于通用知识作答】\n"
        "3. 不要杜撰或添加参考资料中没有的内容"
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


async def query_ragflow(
    plugin: "RAGFlowAdapterPlugin",
    query: str,
    query_label: str = "",
    seen_chunk_hashes: set | None = None,
) -> str:
    """
    使用给定的查询与 RAGFlow API 进行交互，并返回拼接好的上下文。

    Args:
        query: 发送给 RAGFlow 的检索词组
        query_label: 用于在结果中标注来源检索词的标签（通常与 query 相同）
        seen_chunk_hashes: 跨多次调用共享的已见 chunk 哈希集合，用于去重
    """
    if not all(
        [plugin.ragflow_base_url, plugin.ragflow_api_key, plugin.ragflow_kb_ids]
    ):
        logger.warning("RAGFlow 未完全配置，跳过检索。")
        return ""

    if seen_chunk_hashes is None:
        seen_chunk_hashes = set()

    url = f"{plugin.ragflow_base_url.rstrip('/')}/api/v1/retrieval"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {plugin.ragflow_api_key}",
    }
    data = {
        "question": query,
        "dataset_ids": plugin.ragflow_kb_ids,
        "top_k": 1024,
        "similarity_threshold": plugin.rag_similarity_threshold,
        "vector_similarity_weight": 0.3,
        "cross_languages": plugin.ragflow_cross_lang,
        "keyword": True,
        "highlight": True,
    }

    if plugin.ragflow_rerank_model:
        data["rerank_id"] = plugin.ragflow_rerank_model

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data, timeout=plugin.ragflow_request_timeout)
            response.raise_for_status()

            api_data = response.json()
            if api_data.get("code") != 0:
                logger.error(f"RAGFlow API 返回错误: {api_data}")
                return ""

            chunks = api_data.get("data", {}).get("chunks", [])
            if not chunks:
                logger.info(f"RAGFlow 未检索到相关内容（检索词：{query_label or query}）。")
                return RETRIEVAL_NO_RESULTS

            # 提取内容并附上来源标注，跨 query 去重
            content_with_sources = []
            seen_sources = set()
            for chunk in chunks:
                content = chunk.get("content", "")
                if not content:
                    continue

                # 去重：同一内容只保留一次
                content_hash = hashlib.md5(content.encode()).hexdigest()
                if content_hash in seen_chunk_hashes:
                    continue
                seen_chunk_hashes.add(content_hash)

                # 兼容不同 API 响应格式的文档名称字段
                source_name = (
                    chunk.get("document_keyword")
                    or chunk.get("document_name")
                    or chunk.get("document_id", "未知来源")
                )
                if not source_name:
                    source_name = "未知来源"

                content_with_sources.append(f"【来源：{source_name}】\n原文：{content}")
                seen_sources.add(source_name)

            if not content_with_sources:
                # 所有 chunk 均已在其他 query 中出现，视为无新增内容
                logger.info(f"检索词「{query_label or query}」的所有结果均为重复 chunk，已跳过。")
                return RETRIEVAL_NO_RESULTS

            chunks_text = "\n\n".join(content_with_sources)
            label = query_label or query
            # 用检索词组标题包裹本次检索结果
            retrieved_content = f"=== 检索词组：{label} ===\n{chunks_text}"

            logger.info(f"成功从 RAGFlow 检索到 {len(content_with_sources)} 条内容（检索词：{label}），来源: {list(seen_sources)}")
            logger.debug(f"检索到的内容:\n{retrieved_content}")
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

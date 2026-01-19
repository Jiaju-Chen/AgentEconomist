#!/usr/bin/env python3
"""
将远程 Qdrant 服务中的 collection 导出到本地文件存储

用法:
    python export_collections_to_local.py [collection_names...]
    
示例:
    # 导出所有 collection
    python export_collections_to_local.py
    
    # 只导出指定 collection
    python export_collections_to_local.py academic_papers part_products
"""

import sys
import os
from pathlib import Path
import logging
from typing import List, Optional

# 添加项目路径
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from database.knowledge_base.config import QdrantConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def export_collection(
    remote_client: QdrantClient,
    local_client: QdrantClient,
    collection_name: str,
    batch_size: int = 100,
) -> int:
    """
    从远程 Qdrant 导出 collection 到本地
    
    Args:
        remote_client: 远程 Qdrant 客户端
        local_client: 本地 Qdrant 客户端
        collection_name: Collection 名称
        batch_size: 批处理大小
        
    Returns:
        导出的点数
    """
    logger.info(f"📦 开始导出 collection: {collection_name}")
    
    # 检查远程 collection 是否存在
    try:
        remote_info = remote_client.get_collection(collection_name)
    except Exception as e:
        logger.error(f"❌ 远程 collection '{collection_name}' 不存在或无法访问: {e}")
        return 0
    
    logger.info(f"   远程 collection 信息:")
    logger.info(f"   - 点数: {remote_info.points_count}")
    logger.info(f"   - 向量维度: {remote_info.config.params.vectors.size}")
    logger.info(f"   - 距离度量: {remote_info.config.params.vectors.distance}")
    
    # 获取向量配置
    vector_size = remote_info.config.params.vectors.size
    distance = remote_info.config.params.vectors.distance
    
    # 检查本地 collection 是否存在
    local_collections = local_client.get_collections().collections
    local_collection_names = [c.name for c in local_collections]
    
    if collection_name in local_collection_names:
        logger.warning(f"⚠️  本地 collection '{collection_name}' 已存在，将删除后重建")
        local_client.delete_collection(collection_name)
    
    # 创建本地 collection
    local_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=distance,
        ),
    )
    logger.info(f"✅ 已创建本地 collection: {collection_name}")
    
    # 从远程读取所有数据
    total_exported = 0
    offset = None
    
    while True:
        # 使用 scroll 分批读取
        result = remote_client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        
        points, next_offset = result
        
        if not points:
            break
        
        # 转换为 PointStruct 并写入本地
        point_structs = []
        for point in points:
            point_structs.append(
                PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=point.payload,
                )
            )
        
        # 批量写入本地
        local_client.upsert(
            collection_name=collection_name,
            points=point_structs,
        )
        
        total_exported += len(points)
        logger.info(f"   已导出: {total_exported}/{remote_info.points_count} 个点")
        
        if next_offset is None:
            break
        
        offset = next_offset
    
    # 验证
    local_info = local_client.get_collection(collection_name)
    logger.info(f"✅ 导出完成!")
    logger.info(f"   远程点数: {remote_info.points_count}")
    logger.info(f"   本地点数: {local_info.points_count}")
    
    if local_info.points_count != remote_info.points_count:
        logger.warning(f"⚠️  点数不匹配！可能导出不完整")
    
    return total_exported


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="将远程 Qdrant collection 导出到本地文件存储"
    )
    parser.add_argument(
        "collections",
        nargs="*",
        help="要导出的 collection 名称（不指定则导出所有）",
    )
    parser.add_argument(
        "--remote-host",
        default="localhost",
        help="远程 Qdrant 主机地址（默认: localhost）",
    )
    parser.add_argument(
        "--remote-port",
        type=int,
        default=6333,
        help="远程 Qdrant 端口（默认: 6333）",
    )
    parser.add_argument(
        "--local-path",
        type=str,
        default=str(_PROJECT_ROOT / "database" / "qdrant_data"),
        help="本地存储路径（默认: database/qdrant_data）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="批处理大小（默认: 100）",
    )
    
    args = parser.parse_args()
    
    # 创建远程客户端
    logger.info(f"🔗 连接到远程 Qdrant: {args.remote_host}:{args.remote_port}")
    try:
        remote_client = QdrantClient(host=args.remote_host, port=args.remote_port)
        remote_collections = remote_client.get_collections().collections
        remote_collection_names = [c.name for c in remote_collections]
        logger.info(f"✅ 远程连接成功，找到 {len(remote_collection_names)} 个 collection")
    except Exception as e:
        logger.error(f"❌ 无法连接到远程 Qdrant: {e}")
        return 1
    
    # 创建本地客户端
    local_path = Path(args.local_path)
    local_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 本地存储路径: {local_path}")
    
    # 检查是否有锁定文件
    lock_file = local_path / ".lock"
    if lock_file.exists():
        logger.warning(f"⚠️  检测到锁定文件: {lock_file}")
        logger.warning(f"   可能有其他进程正在使用本地 Qdrant")
        # 使用临时路径
        import tempfile
        temp_path = Path(tempfile.mkdtemp(prefix="qdrant_export_"))
        logger.info(f"   使用临时路径: {temp_path}")
        logger.info(f"   导出完成后，请手动将数据复制到: {local_path}")
        local_path = temp_path
    
    try:
        local_client = QdrantClient(path=str(local_path))
        logger.info(f"✅ 本地 Qdrant 客户端初始化成功")
    except Exception as e:
        if "already accessed" in str(e):
            logger.error(f"❌ 本地 Qdrant 文件被锁定")
            logger.error(f"   请先停止使用本地 Qdrant 的进程（如 Streamlit）")
            logger.error(f"   或使用 --local-path 指定其他路径")
        else:
            logger.error(f"❌ 无法初始化本地 Qdrant: {e}")
        return 1
    
    # 确定要导出的 collection
    if args.collections:
        collections_to_export = args.collections
    else:
        collections_to_export = remote_collection_names
        logger.info(f"📋 未指定 collection，将导出所有: {collections_to_export}")
    
    # 验证 collection 是否存在
    invalid_collections = [
        name for name in collections_to_export if name not in remote_collection_names
    ]
    if invalid_collections:
        logger.error(f"❌ 以下 collection 在远程不存在: {invalid_collections}")
        return 1
    
    # 导出每个 collection
    total_exported = 0
    for collection_name in collections_to_export:
        try:
            count = export_collection(
                remote_client=remote_client,
                local_client=local_client,
                collection_name=collection_name,
                batch_size=args.batch_size,
            )
            total_exported += count
            logger.info("")
        except Exception as e:
            logger.error(f"❌ 导出 collection '{collection_name}' 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info("=" * 60)
    logger.info(f"✅ 全部导出完成！共导出 {total_exported} 个点")
    logger.info(f"📁 本地存储位置: {local_path}")
    logger.info("")
    logger.info("💡 提示: 现在可以将 KB_QDRANT_MODE 设置为 'local' 使用本地存储")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


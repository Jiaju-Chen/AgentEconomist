#!/usr/bin/env bash
# 简化版导出脚本 - 将 Docker Qdrant 的 collection 导出到本地

set -e

REMOTE_HOST="${REMOTE_HOST:-localhost}"
REMOTE_PORT="${REMOTE_PORT:-6333}"
LOCAL_PATH="${LOCAL_PATH:-/root/project/agentsociety-ecosim/database/qdrant_data_backup}"
COLLECTIONS="${COLLECTIONS:-academic_papers part_products}"

echo "=========================================="
echo "Qdrant Collection 导出工具"
echo "=========================================="
echo ""
echo "远程 Qdrant: ${REMOTE_HOST}:${REMOTE_PORT}"
echo "本地路径: ${LOCAL_PATH}"
echo "要导出的 Collections: ${COLLECTIONS}"
echo ""

# 创建本地目录
mkdir -p "${LOCAL_PATH}"

# 使用 Python 脚本导出
cd /root/project/agentsociety-ecosim
conda run -n ecosim python <<EOF
import sys
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REMOTE_HOST = "${REMOTE_HOST}"
REMOTE_PORT = int("${REMOTE_PORT}")
LOCAL_PATH = Path("${LOCAL_PATH}")
COLLECTIONS = "${COLLECTIONS}".split()

# 连接远程
logger.info(f"连接到远程 Qdrant: {REMOTE_HOST}:{REMOTE_PORT}")
remote_client = QdrantClient(host=REMOTE_HOST, port=REMOTE_PORT)
remote_collections = remote_client.get_collections().collections
logger.info(f"找到 {len(remote_collections)} 个 collection")

# 连接本地（使用临时路径避免锁定）
import tempfile
temp_path = Path(tempfile.mkdtemp(prefix="qdrant_export_"))
logger.info(f"使用临时路径: {temp_path}")
local_client = QdrantClient(path=str(temp_path))

for collection_name in COLLECTIONS:
    if collection_name not in [c.name for c in remote_collections]:
        logger.warning(f"Collection '{collection_name}' 不存在，跳过")
        continue
    
    logger.info(f"\\n开始导出: {collection_name}")
    
    # 获取远程信息
    remote_info = remote_client.get_collection(collection_name)
    logger.info(f"  远程: {remote_info.points_count} 个点, 维度: {remote_info.config.params.vectors.size}")
    
    # 创建本地 collection
    if collection_name in [c.name for c in local_client.get_collections().collections]:
        local_client.delete_collection(collection_name)
    
    local_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=remote_info.config.params.vectors.size,
            distance=remote_info.config.params.vectors.distance,
        ),
    )
    
    # 导出数据
    total = 0
    offset = None
    batch_size = 100
    
    while True:
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
        
        point_structs = [PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points]
        local_client.upsert(collection_name=collection_name, points=point_structs)
        
        total += len(points)
        if total % 1000 == 0:
            logger.info(f"  已导出: {total} 个点")
        
        if next_offset is None:
            break
        offset = next_offset
    
    local_info = local_client.get_collection(collection_name)
    logger.info(f"✅ {collection_name}: 远程 {remote_info.points_count} -> 本地 {local_info.points_count}")

logger.info(f"\\n✅ 导出完成！")
logger.info(f"临时路径: {temp_path}")
logger.info(f"请手动复制到: {LOCAL_PATH}")
logger.info(f"命令: cp -r {temp_path}/* {LOCAL_PATH}/")

EOF

echo ""
echo "✅ 脚本执行完成！"



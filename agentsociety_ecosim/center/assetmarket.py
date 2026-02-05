# TODO: 商品/资产市场
# @夏栩
from typing import List, Optional, Dict, Any
from .model import Product
import ray
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModel
import torch
import re
import pandas as pd  # 添加pandas导入

from agentsociety_ecosim.utils.data_loader import load_processed_products
from agentsociety_ecosim.utils.embedding import embedding
from agentsociety_ecosim.utils.log_utils import setup_global_logger
from agentsociety_ecosim.utils.product_attribute_loader import (
    inject_product_attributes,
    get_product_attributes,
)
from transformers import AutoTokenizer, AutoModel

# 使用第二张GPU卡（避免使用第一张卡）
import os

# 为 MCP 服务器设置：如果环境变量 MCP_MODE 存在，强制使用 CPU
if os.getenv('MCP_MODE'):
    device = "cpu"
    num_gpus = 0
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '6' 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus = 1

logger = setup_global_logger(name="productmarket")

@ray.remote(num_gpus=num_gpus)
class ProductMarket:
    def __init__(self):
        self.products: List[Product] = []
        self.tokenizer = AutoTokenizer.from_pretrained(os.getenv("MODEL_PATH"))
        self.model = AutoModel.from_pretrained(os.getenv("MODEL_PATH")).to(device)
        # 删除self.df，改为从动态的self.products中搜索
        # self.df = load_processed_products()  # 不再需要静态数据
        self.collection_name = "part_products"
        
        # Qdrant 客户端：优先使用本地模式（不需要 Docker）
        qdrant_url = os.getenv("QDRANT_URL")
        if qdrant_url:
            self.client = QdrantClient(url=qdrant_url)
            logger.info(f"Using remote Qdrant: {qdrant_url}")
        else:
            # 本地模式：存储在项目目录下（与 simulation.py 共享同一个 Qdrant 实例）
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            qdrant_path = os.path.join(project_root, "agentsociety_ecosim", "data", "qdrant_data")
            self.client = QdrantClient(path=qdrant_path)
            logger.info(f"Using local Qdrant storage: {qdrant_path} (collection: {self.collection_name})")
        
        # 创建 collection（如果不存在）
        from qdrant_client.models import VectorParams, Distance
        try:
            self.client.get_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} already exists")
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            logger.info(f"Created collection {self.collection_name}")
        
        logger.info("ProductMarket initialized.")
        print("ProductMarket initialized.")

    def batch_load_to_qdrant(self, product_list: List[Product]):
        """
        批量加载商品向量到 Qdrant（在 ProductMarket Actor 内部执行，避免文件锁冲突）
        """
        from agentsociety_ecosim.utils.embedding import batch_embedding
        from qdrant_client.models import PointStruct
        from uuid import uuid5, NAMESPACE_DNS
        
        # 🚀 批量处理：先收集所有文本
        texts = []
        for product in product_list:
            text = ' '.join([product.name, product.brand, product.description or '', product.classification])
            texts.append(text)
        
        # 🚀 批量计算所有向量（加速 5-10 倍）
        vectors = batch_embedding(texts, self.tokenizer, self.model, batch_size=32)
        
        # 构建 Qdrant points
        points = []
        for product, vector in zip(product_list, vectors):
            payload = {
                "name": product.name,
                "Uniq Id": product.product_id,
                "description": product.description,
                "classification": product.classification,
                "price": product.price,
                "owner_id": product.owner_id,
                "description": product.description or ""  # 确保 description 不为 None
            }
            
            # 🔥 使用复合ID确保竞争模式下同一商品的不同供应商都能存储
            composite_string = f"{product.product_id}@{product.owner_id}"
            unique_id = str(uuid5(NAMESPACE_DNS, composite_string))
            points.append(PointStruct(id=unique_id, vector=vector, payload=payload))
        
        # 批量插入 Qdrant
        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"[Qdrant] 批量插入 {len(points)} 个商品向量")
        return len(points)

    
    def publish_product(self, product: Product):

        if not getattr(product, "attributes", None) and getattr(product, "product_id", None):
            attrs = get_product_attributes(product.product_id)
            if attrs:
                product.attributes = attrs
                if product.is_food is None:
                    product.is_food = attrs.get("is_food")
                if product.nutrition_supply is None:
                    product.nutrition_supply = attrs.get("nutrition_supply")
                if product.satisfaction_attributes is None:
                    product.satisfaction_attributes = attrs.get("satisfaction_attributes")
                if product.duration_months is None:
                    product.duration_months = attrs.get("duration_months")

        self.products.append(product)

    def search_by_vector(self, query: str, top_k: int = 20, must_contain: Optional[str] = None) -> List[Product]:
        """
        使用向量搜索匹配商品（从 Qdrant 搜索后匹配到 self.products）
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            must_contain: 必须包含的分类关键词
        
        Returns:
            匹配的商品列表（包含实时库存信息）
        """
        query_vec = embedding(query, self.tokenizer, self.model)
        
        # 🔧 增加搜索数量以补偿库存过滤
        search_limit = top_k * 3
        
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            limit=search_limit
        ).points
        
        # 🔥 构建 self.products 的快速查找索引 (product_id, owner_id) -> Product
        products_index = {}
        for p in self.products:
            key = (p.product_id, p.owner_id)
            products_index[key] = p
        
        results = []
        for hit in hits:
            payload = hit.payload
            if must_contain and must_contain.lower() not in (payload.get("classification") or '').lower():
                continue
            
            owner_id = payload.get("owner_id", "default_firm")
            product_id = payload.get("Uniq Id")
            
            # 🔥 从 self.products 中查找对应商品（获取实时库存）
            key = (product_id, owner_id)
            if key in products_index:
                product = products_index[key]
                # 只返回有库存的商品
                if product.amount > 0:
                    results.append(product)
            else:
                # 如果在 self.products 中找不到，使用 payload 信息（降级）
                logger.debug(f"商品 {product_id}@{owner_id} 不在 self.products 中，使用 Qdrant payload")
                product_kwargs = dict(
                    name=payload["name"],
                    description=payload.get("description"),
                    classification=payload.get("classification"),
                    price=payload.get("price"),
                    amount=1.0,  # 降级时使用默认库存
                    owner_id=owner_id,
                    asset_type="products",
                    product_id=product_id,
                )
                product_kwargs = inject_product_attributes(product_kwargs, product_id)
                matched = Product(**product_kwargs)
                results.append(matched)
            
            # 达到需要的数量就停止
            if len(results) >= top_k:
                break

        return results

    async def search_products(self, query: str = "", max_price: Optional[float] = None, top_k: int = 20, must_contain: Optional[str] = None, economic_center=None) -> List[Product]:
        """
        从self.products中搜索商品，从EconomicCenter获取实时库存数量
        先进行关键词匹配，如果找不到结果则用Qdrant语义搜索
        """
        results = []

        # 🔄 如果提供了economic_center，先批量更新所有商品的实时库存到 self.products
        if economic_center:
            try:
                # 🚀 一次性获取所有商品的库存信息
                inventory_dict = await economic_center.get_all_product_inventory.remote()
                
                # 批量更新 self.products 中的库存
                updated_count = 0
                for product in self.products:
                    key = (product.product_id, product.owner_id)
                    if key in inventory_dict:
                        product.amount = inventory_dict[key]
                        updated_count += 1
                
                logger.info(f"✅ 批量更新了 {updated_count}/{len(self.products)} 个商品的实时库存")
            except Exception as e:
                logger.warning(f"批量更新商品库存失败: {e}")
        
        # 筛选有库存的商品
        available_products = [p for p in self.products if p.amount > 0]
        
        is_valid_query = bool(re.search(r"[\u4e00-\u9fa5\w]{8,}", query))
        if is_valid_query:
            # 使用向量搜索（self.products已更新，search_by_vector会从中获取库存）
            results = self.search_by_vector(query, top_k=top_k, must_contain=must_contain)    
        else:
            # 从self.products中进行关键词匹配
            for product in available_products:
                # 检查商品名称、分类、品牌是否包含查询关键词
                matches_name = query.lower() in (product.name or "").lower()
                matches_classification = query.lower() in (product.classification or "").lower()
                matches_brand = query.lower() in (product.brand or "").lower()
                matches_description = query.lower() in (product.description or "").lower()
                
                if matches_name or matches_classification or matches_brand or matches_description:
                    # 检查价格限制
                    if max_price is not None and product.price > max_price:
                        continue
                    
                    # 检查must_contain条件
                    if must_contain and must_contain.lower() not in (product.classification or "").lower():
                        continue
                    
                    # 检查价格是否有效
                    if not product.price or product.price <= 0:
                        continue
                    
                    # 创建搜索结果，保持原有数量信息
                    product_kwargs = dict(
                        name=product.name,
                        amount=product.amount,  # 使用实际库存数量
                        price=product.price,
                        owner_id=product.owner_id,
                        classification=product.classification,
                        brand=product.brand,
                        product_id=product.product_id,
                        description=product.description,
                        attributes=product.attributes,
                        is_food=product.is_food,
                        nutrition_supply=product.nutrition_supply,
                        satisfaction_attributes=product.satisfaction_attributes,
                        duration_months=product.duration_months
                    )
                    product_kwargs = inject_product_attributes(product_kwargs, product.product_id)
                    result_product = Product.create(**product_kwargs)
                    results.append(result_product)

        if len(results) < top_k:
            # 如果关键词匹配没有找到足够结果，则使用语义搜索
            # 但只在有库存的商品中搜索
            if available_products:
                vector_results = self.search_by_vector(query, top_k=top_k - len(results), must_contain=must_contain)
                # 过滤语义搜索结果，只返回有库存的商品
                filtered_vector_results = []
                for vector_result in vector_results:
                    # 查找对应的实际商品以获取库存信息
                    actual_product = next((p for p in self.products if p.product_id == vector_result.product_id), None)
                    if actual_product and actual_product.amount > 0:
                        # 使用实际库存数量
                        vector_result.amount = actual_product.amount
                        filtered_vector_results.append(vector_result)
                        # 更新商品实际价格
                        vector_result.price = actual_product.price
                results.extend(filtered_vector_results)
        return results[:top_k]

    def get_current_prices(self, name: str) -> List[float]:
        prices = []
        for product in self.products:
            if product.name == name:
                prices.append(product.price)
        return prices
    
    
    async def get_product_stock(self, product_id: str, economic_center) -> float:
        """
        获取商品的当前库存数量，从EconomicCenter获取实时数据
        """
        # 查找商品的owner_id
        owner_id = None
        for product in self.products:
            if product.product_id == product_id:
                owner_id = product.owner_id
                break
        
        if not owner_id:
            return 0.0
            
        try:
            # 从EconomicCenter获取实时库存
            stock = await economic_center.get_product_inventory.remote(owner_id, product_id)
            return stock
        except Exception as e:
            logger.error(f"获取库存失败: {e}")
            return 0.0
    
    
    async def get_all_listings(self, economic_center=None) -> List[Product]:
        """
        Returns all active product listings with positive amount and a defined price.
        如果提供了economic_center，则返回实时库存信息
        """
        if economic_center:
            # 从EconomicCenter获取实时库存信息
            updated_products = []
            for product in self.products:
                try:
                    real_stock = await economic_center.get_product_inventory.remote(product.owner_id, product.product_id)
                    if real_stock and real_stock > 0 and product.price is not None:
                        # 创建包含实时库存的产品副本
                        product_kwargs = dict(
                            name=product.name,
                            amount=real_stock,
                            price=product.price,
                            owner_id=product.owner_id,
                            classification=product.classification,
                            brand=product.brand,
                            product_id=product.product_id,
                            description=product.description,
                            attributes=product.attributes,
                            is_food=product.is_food,
                            nutrition_supply=product.nutrition_supply,
                            satisfaction_attributes=product.satisfaction_attributes,
                            duration_months=product.duration_months
                        )
                        product_kwargs = inject_product_attributes(product_kwargs, product.product_id)
                        updated_product = Product.create(**product_kwargs)
                        updated_products.append(updated_product)
                except Exception as e:
                    logger.warning(f"获取商品 {product.product_id} 实时库存失败: {e}")
                    # 使用本地库存作为备选
                    if product.amount > 0 and product.price is not None:
                        updated_products.append(product)
            return updated_products
        else:
            # 使用本地库存信息
            return [p for p in self.products if p.amount > 0 and p.price is not None]
    
    def get_avg_price(self) -> float:
        """
        获取商品平均价格
        """
        return sum([p.price for p in self.products if p.price is not None]) / len([p for p in self.products if p.price is not None])
    
    def update_products_from_economic_center(self, products: List[Product]):
        """
        从EconomicCenter接收更新的商品列表
        这个方法用于保持ProductMarket的商品信息与EconomicCenter同步
        """
        try:
            # 创建产品ID到产品的映射
            product_map = {p.product_id: p for p in products}
            
            # 更新现有商品的库存信息
            for i, local_product in enumerate(self.products):
                if local_product.product_id in product_map:
                    updated_product = product_map[local_product.product_id]
                    # 更新库存数量，保持其他信息不变
                    self.products[i].amount = updated_product.amount
            
            logger.info(f"已从EconomicCenter更新 {len(products)} 个商品的库存信息")
        except Exception as e:
            logger.error(f"更新商品库存信息失败: {e}")
    
    def update_product_prices(self, price_changes: Dict[str, float]) -> bool:
        """
        更新商品价格
        price_changes: {product_id: new_price}
        """
        try:
            updated_count = 0
            for product in self.products:
                if product.product_id in price_changes:
                    old_price = product.price
                    new_price = price_changes[product.product_id]
                    product.price = new_price
                    updated_count += 1
                    logger.info(f"ProductMarket: 商品 {product.name} 价格更新 ${old_price:.2f} -> ${new_price:.2f}")
            
            logger.info(f"ProductMarket: 已更新 {updated_count} 个商品的价格")
            return True
        except Exception as e:
            logger.error(f"ProductMarket: 更新商品价格失败: {e}")
            return False
    
    def get_price_statistics(self) -> Dict[str, Any]:
        """
        获取价格统计信息
        """
        try:
            prices = [p.price for p in self.products if p.price is not None and p.price > 0]
            if not prices:
                return {"count": 0, "avg_price": 0, "min_price": 0, "max_price": 0}
            
            return {
                "count": len(prices),
                "avg_price": sum(prices) / len(prices),
                "min_price": min(prices),
                "max_price": max(prices),
                "price_range": max(prices) - min(prices)
            }
        except Exception as e:
            logger.error(f"获取价格统计失败: {e}")
            return {"count": 0, "avg_price": 0, "min_price": 0, "max_price": 0}



from enum import Enum


class EmbeddingType(Enum):
    TEXT_EMBEDDING = "Text Embedding"
    IMAGE_EMBEDDING = "Image Embedding"
    AUDIO_EMBEDDING = "Audio Embedding"
    VIDEO_EMBEDDING = "Video Embedding"


class SearchBalancer:
    DEFAULT_NUM_CANDIDATES: int = 50
    DEFAULT_LIMIT_PER_GROUP: int = 50
    DEFAULT_LIMIT:  int = 50
    STOP_INDEX: int = 32
    BATCH_SIZE: int = 1024
    THRESHOLD: float = 0.0

class SearchStrategy(Enum):
    FILTER = None
    ORDER_BY = None

class SearchArgs(Enum):
    ARTICLES = {
        "database_name" : "market",
        "collection_name" : "articles",
        "path" : "content_embeddings",
        "index" : "article_index",
        "num_candidates" : SearchBalancer.DEFAULT_NUM_CANDIDATES,
        "limit" : SearchBalancer.DEFAULT_LIMIT_PER_GROUP,
    }

    TICKERS = {
        "database_name" : "market",
        "collection_name" : "tickers",
        "path" : "name_embeddings",
        "index" : "ticker_index",
        "num_candidates" : SearchBalancer.DEFAULT_NUM_CANDIDATES,
        "limit" : SearchBalancer.DEFAULT_LIMIT_PER_GROUP,
    }

    FOREX = {
        "database_name" : "market",
        "collection_name" : "forex",
        "path" : "price_embeddings",
        "index" : "forex_index",
        "num_candidates" : SearchBalancer.DEFAULT_NUM_CANDIDATES,
        "limit" : SearchBalancer.DEFAULT_LIMIT_PER_GROUP,
    }

    CRYPTOS = {
        "database_name" : "market",
        "collection_name" : "cryptos",
        "path" : "price_embeddings",
        "index" : "crypto_index",
        "num_candidates" : SearchBalancer.DEFAULT_NUM_CANDIDATES,
        "limit" : SearchBalancer.DEFAULT_LIMIT_PER_GROUP,
    }
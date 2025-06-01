import os
from typing import List, Optional
from openai import OpenAI
from langchain_core.embeddings import Embeddings


class CustomOpenAIEmbeddings(Embeddings):
    """CustomEmbeddings embedding model integration.

    Key init args — completion params:
        model: str
            Name of model to use.
        base_url: str
            Base URL for API requests. Only specify if using a proxy or service
            emulator.
        api_key: Optional[str]
            API key. If not passed in will be read from env var OPENAI_API_KEY.
    """

    def __init__(self, model: str, base_url: str, api_key: Optional[str]):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        self._openai_clinet = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embeddings = self._openai_clinet.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [embedding.embedding for embedding in embeddings.data]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]
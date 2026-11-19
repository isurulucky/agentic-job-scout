from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


class NavigationStatusVectorStore:
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "navigation_page_states",
        embedding_model: str = "intfloat/multilingual-e5-base",
    ):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )

    @staticmethod
    def _page_state_to_text(page_state: dict) -> str:
        elements = page_state.get("elements", [])

        element_descriptions = [
            f"{element.get('type', 'element')}: {element.get('name', '')}"
            for element in elements
            if element.get("name")
        ]

        return "\n".join([
            f"URL: {page_state.get('url', '')}",
            f"Title: {page_state.get('title', '')}",
            "Visible elements:",
            *element_descriptions,
        ])

    def add(
        self,
        experience_uuid: str,
        framework: str,
        page_states: list[dict],
    ) -> None:
        """Add page states associated with a navigation experience."""

        documents = []
        ids = []

        for state in page_states:
            page_state = state["page_state"]
            page_state_uuid = state["uuid"]
            step_index = state["step_index"]

            documents.append(
                Document(
                    page_content=self._page_state_to_text(page_state),
                    metadata={
                        "experience_uuid": experience_uuid,
                        "page_state_uuid": page_state_uuid,
                        "step_index": step_index,
                        "framework": framework,
                    },
                )
            )

            ids.append(page_state_uuid)

        if documents:
            self.vector_store.add_documents(
                documents=documents,
                ids=ids,
            )

    def retrieve(
        self,
        page_state: dict,
        framework: str,
        k: int = 5,
        relevance_score: float = 0.9
    ) -> list[tuple[Document, float]]:
        """Retrieve similar page states, preferring the same framework."""

        query = self._page_state_to_text(page_state)

        # First, search within the same framework.
        results = self.vector_store.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            filter={"framework": framework},
        )

        # If no matching page states exist, search across frameworks.
        if not results:
            results = self.vector_store.similarity_search_with_relevance_scores(
                query=query,
                k=k,
            )

        results = [
                (document, score)
                for document, score in results
                if score >= relevance_score
            ]

        return results

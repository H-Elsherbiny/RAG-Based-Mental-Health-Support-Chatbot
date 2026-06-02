from groq import Groq
from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, RrfQuery, Prefetch
from fastembed import TextEmbedding, SparseTextEmbedding

from app.core.config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, GROQ_API_KEY


class RAGPipeline:
    """
    Advanced Search Pipeline for Hybrid RAG using Qdrant.
    Handles user query rewrite, dense/sparse two-stage hybrid search, and cross-encoder re-ranking.
    """
    def __init__(self,
                 model_name: str = "openai/gpt-oss-20b", 
                 temperature: float = 0.0, 
                 max_tokens: int = 100):
        
        # 1. Validate Configurations
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set in environment or config.")
        if not QDRANT_URL or not QDRANT_API_KEY:
            raise ValueError("QDRANT_URL or QDRANT_API_KEY is not set in environment or config.")
        
        self.collection_name = COLLECTION_NAME
        
        # 2. Initialize Clients
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        self.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        # 3. Initialize Embedding and Reranking Models
        self.dense_model = TextEmbedding(model_name="BAAI/bge-base-en-v1.5")
        self.sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # 4. Store LLM Generation Parameters
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        
    def rewrite_user_query(self, raw_user_prompt: str) -> str:
        """
        Uses a fast Groq model to strip conversational noise and output
        a semantically dense search string optimized for vector database lookup.
        """
        system_instruction = (
            "You are an advanced search query optimization engine for a mental health knowledge base. "
            "Your task is to transform raw, conversational user prompts into a detailed, descriptive, "
            "and semantically rich search paragraph (approximately 2 to 3 sentences long).\n\n"
            "Guidelines:\n"
            "1. Strip away all conversational fluff, greetings, casual sign-offs, and superficial narratives.\n"
            "2. Synthesize and expand upon the core psychological distress markers, emotional states, "
            "physical somatic symptoms, and situational triggers mentioned.\n"
            "3. Write the output as a fully articulated, dense description of a clinical scenario. "
            "Use clear, precise, and expressive terminology that mimics how a detailed patient case study "
            "or thorough counseling log is written.\n"
            "4. Output ONLY the optimized descriptive paragraph. Do not include introductory remarks, "
            "explanations, headers, or quotes."
        )

        response = self.groq_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Optimize this prompt for vector search: {raw_user_prompt}"}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        optimized_query = response.choices[0].message.content.strip()
        return optimized_query

    def _rerank_results(self, search_query: str, candidate_payloads: list) -> list:
        """
        Performs deep attention-based cross-encoding on candidate pairs
        and re-sorts them by absolute contextual relevance.
        """
        if not candidate_payloads:
            return []

        pairs = []
        for item in candidate_payloads:
            context_block = (
                f"Problem: {item['historical_patient_problem']}\n"
                f"Advice: {item['matched_counselor_advice']}"
            )
            pairs.append([search_query, context_block])

        # Compute cross-attention relevance scores
        scores = self.cross_encoder.predict(pairs)

        # Attach scores to payloads and sort them in descending order
        for idx, score in enumerate(scores):
            candidate_payloads[idx]["cross_encoder_score"] = float(score)

        reranked_payloads = sorted(
            candidate_payloads,
            key=lambda x: x["cross_encoder_score"],
            reverse=True
        )

        return reranked_payloads

    def two_stage_hybrid_search(self, raw_user_query: str,
                                top_k_scenarios: int = 3,
                                top_k_advice: int = 3,
                                top_n_final: int = 3) -> dict:
        """
        Executes the two-stage hybrid search across dense and sparse vectors, 
        followed by a final reranking stage.
        """
        # ==========================================
        # 1. Query Parsing / Rewriting Layer
        # ==========================================
        optimized_search_query = self.rewrite_user_query(raw_user_query)

        # ==========================================
        # 2. Embed the Optimized Query
        # ==========================================
        query_dense = list(self.dense_model.embed([optimized_search_query]))[0].tolist()

        sparse_res = list(self.sparse_model.embed([optimized_search_query]))[0]
        query_sparse = models.SparseVector(
            indices=sparse_res.indices.tolist(),
            values=sparse_res.values.tolist()
        )

        hybrid_prefetch = [
            Prefetch(query=query_dense, using="dense", limit=top_k_scenarios * 4),
            Prefetch(query=query_sparse, using="sparse", limit=top_k_scenarios * 4)
        ]

        # ==========================================
        # STAGE 1: Find Distinct Patient Scenarios
        # ==========================================
        stage_1_results = self.qdrant_client.query_points_groups(
            collection_name=self.collection_name,
            prefetch=hybrid_prefetch,
            query=RrfQuery(rrf=models.Rrf()),
            group_by="parent_id",
            group_size=1,
            limit=top_k_scenarios,
            query_filter=Filter(
                must=[FieldCondition(key="doc_type", match=MatchValue(value="patient_problem"))]
            )
        )

        if not stage_1_results.groups or len(stage_1_results.groups) == 0:
            return {"rag_status": "MISS", "context_payload": []}

        winning_parent_ids = []
        parent_problem_map = {}

        for group in stage_1_results.groups:
            if group.hits:
                p_id = group.id
                winning_parent_ids.append(p_id)
                parent_problem_map[p_id] = group.hits[0].payload.get("text", "")

        # ==========================================
        # STAGE 2: Isolate Advice Within Those Specific Scenarios
        # ==========================================
        stage_2_results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            prefetch=hybrid_prefetch,
            query=RrfQuery(rrf=models.Rrf()),
            query_filter=Filter(
                must=[
                    FieldCondition(key="doc_type", match=MatchValue(value="counselor_advice")),
                    FieldCondition(key="parent_id", match=MatchAny(any=winning_parent_ids))
                ]
            ),
            limit=top_k_advice
        )

        # Build raw un-ranked candidates array
        initial_candidates = []
        for point in stage_2_results.points:
            p_id = point.payload["parent_id"]
            initial_candidates.append({
                "scenario_id": p_id,
                "historical_patient_problem": parent_problem_map.get(p_id, ""),
                "matched_counselor_advice": point.payload["text"]
            })

        # ==========================================
        # STAGE 3: Re-ranking
        # ==========================================
        reranked_context = self._rerank_results(optimized_search_query, initial_candidates)
        final_context = reranked_context[:top_n_final]

        return {
            "rag_status": "HIT",
            "optimized_query": optimized_search_query,
            "context_payload": final_context
        }

    def format_retrieved_context(self, search_output: dict) -> str:
        """
        Safely unpacks the search output dictionary, handling both RAG HITs
        and MISSes, formatting them nicely for prompt injection.
        """
        if search_output.get("rag_status") == "MISS":
            return (
                "CRITICAL: No matching historical clinical scenarios were found in the database. "
                "Rely entirely on safe, generic parametric knowledge. Do not reference historical case studies, "
                "do not make explicit medical diagnoses, and offer universally safe grounding strategies."
            )

        payload = search_output.get("context_payload", [])
        if not payload:
            return "No specific counselor advice found for this topic."

        context_string = ""
        for i, doc in enumerate(payload):
            context_string += f"--- Verified Counselor Advice Fragment {i+1} ---\n{doc['matched_counselor_advice']}\n\n"

        return context_string.strip()

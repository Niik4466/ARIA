import time
from collections import deque
from typing import Optional, Tuple, List
from src.agent import (
    call_ollama,
    clean_think_tags,
    get_insight_extraction_prompt,
    get_intent_extraction_prompt,
    get_query_enrichment_prompt
)
from src.utils import Config

class MemoryManager:
    def __init__(self, rag_manager):
        self.rag_manager = rag_manager
        self.config = Config()
        self.small_model = self.config.get("DECISOR_MODEL")
        
        # Immediate STM
        self.immediate_stm = deque(maxlen=2)
        self.last_interaction_time = 0.0
        self.immediate_stm_ttl = float(self.config.get("IMMEDIATE_STM_TTL", 30.0))
        
        # Extended STM
        self.extended_stm: Optional[str] = None
        self.extended_stm_time = 0.0
        self.extended_stm_ttl = float(self.config.get("EXTENDED_STM_TTL", 120.0))
        
        # Configuration
        self.insight_threshold = float(self.config.get("INSIGHT_THRESHOLD", 0.35))  # Similarity threshold for insights

    def _cleanup_stm(self):
        """Clears STM if TTL has expired."""
        now = time.time()
        if now - self.last_interaction_time > self.immediate_stm_ttl:
            self.immediate_stm.clear()
            
        if self.extended_stm and (now - self.extended_stm_time > self.extended_stm_ttl):
            self.extended_stm = None

    def handle_dual_query(self, query: str) -> Optional[str]:
        """
        Evaluates the existing Extended STM. 
        If valid intent exists, enriches the query (Q2).
        """
        self._cleanup_stm()
        if not self.extended_stm:
            return None
            
        prompt = get_query_enrichment_prompt(query, self.extended_stm)
        try:
            q2 = call_ollama(prompt="Enrich query", model=self.small_model, system_prompt=prompt, temperature=0.2)
            q2 = clean_think_tags(q2).strip()
            print(f"[🧠 Memory] Q2 Generated: {q2}")
            return q2
        except Exception as e:
            print(f"[🧠 Memory] Failed to generate Q2: {e}")
            return None

    def retrieve_relevant_insights(self, query: str) -> str:
        """
        Fetches insights from LTM via RAGManager, filters by cosine similarity threshold.
        Returns formatted insights to be injected into the response prompt.
        """
        insights = self.rag_manager.get_insights(query, k=3)
        valid_insights = []
        for ins in insights:
            if ins.get("similarity", 0) >= self.insight_threshold:
                valid_insights.append(ins["text"])
                
        if not valid_insights:
            return ""
            
        formatted = "\n".join([f"- {i}" for i in valid_insights])
        print(f"[🧠 Memory] Retrieved Insights:\n{formatted}")
        return formatted

    def update_after_interaction(self, user_text: str, agent_response: str):
        """
        Updates LTM (Insights & Conversation History) and STM asynchronously 
        or synchronously after the response has been generated.
        """
        now = time.time()
        
        # Update Immediate STM
        self.immediate_stm.append({"user": user_text, "agent": agent_response})
        self.last_interaction_time = now
        
        # 1. Update Conversational History in LTM
        print(f"[✨ Integrated] Registering conversation in RAG history...")
        self.rag_manager.add_to_history(user_text, agent_response)
        
        # 2. Extract and Update Extended STM Intent
        intent_prompt = get_intent_extraction_prompt(user_text)
        try:
            intent_res = call_ollama(prompt="Extract intent", model=self.small_model, system_prompt=intent_prompt, temperature=0.2)
            intent_res = clean_think_tags(intent_res).strip()
            self.extended_stm = intent_res
            self.extended_stm_time = now
            print(f"[🧠 Memory] Extended STM updated: \n{self.extended_stm}")
        except Exception as e:
            print(f"[🧠 Memory] Intent extraction failed: {e}")

        # 3. Extract and Update LTM Insights
        insight_prompt = get_insight_extraction_prompt(user_text)
        try:
            insight_res = call_ollama(prompt="Extract insights", model=self.small_model, system_prompt=insight_prompt, temperature=0.1)
            insight_res = clean_think_tags(insight_res).strip()
            
            if "no insights" not in insight_res.lower() and len(insight_res) > 5:
                print(f"[🧠 Memory] New Insights generated:\n{insight_res}")
                # Parse bullet points and store them individually without duplicates
                lines = insight_res.split('\n')
                for line in lines:
                    cleaned_line = line.strip(" *-•\t")
                    if len(cleaned_line) < 3:
                        continue
                        
                    # Check for semantic duplicates
                    existing = self.rag_manager.get_insights(cleaned_line, k=1)
                    is_duplicate = False
                    if existing:
                        top_sim = existing[0].get("similarity", 0)
                        if top_sim > 0.85:
                            is_duplicate = True
                            print(f"   [🧠 Memory] Skipping duplicate insight: '{cleaned_line}' (Sim: {top_sim:.2f})")
                            
                    if not is_duplicate:
                        print(f"   [🧠 Memory] Storing unique insight: '{cleaned_line}'")
                        self.rag_manager.write_insight(cleaned_line)
                        
        except Exception as e:
            print(f"[🧠 Memory] Insight extraction failed: {e}")

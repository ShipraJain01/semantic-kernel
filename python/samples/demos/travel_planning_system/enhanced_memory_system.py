# Copyright (c) Microsoft. All rights reserved.
import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
from opentelemetry import trace, context, baggage
from opentelemetry.context import attach, detach
from pydantic import BaseModel, Field

from semantic_kernel.memory.memory_record import MemoryRecord
from semantic_kernel.memory.volatile_memory_store import VolatileMemoryStore
from semantic_kernel.connectors.ai.embedding_generator_base import EmbeddingGeneratorBase

logger = logging.getLogger(__name__)

# ✅ Custom Event Names following pattern from telemetry
STATE_MANAGEMENT_EVENT = "gen_ai.agent.state.management"
AGENT_INTERACTION_EVENT = "gen_ai.agent.to.agent.interaction"
SYSTEM = "semantic-kernel"


@dataclass
class AgentMemoryContext:
    """Context information for agent memory operations."""
    agent_id: str
    conversation_id: str
    trace_id: str
    span_id: str
    session_metadata: Dict[str, Any] = field(default_factory=dict)
    
    
@dataclass
class SharedMemoryEvent:
    """Event for shared memory operations."""
    event_type: str  # 'agent_transition', 'knowledge_enrichment', 'memory_update'
    source_agent: str
    target_agent: Optional[str]
    context_data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class EnhancedVolatileMemoryStore(VolatileMemoryStore):
    """Enhanced memory store with telemetry and context tracking."""
    
    def __init__(self, agent_id: str = None):
        super().__init__()
        self.agent_id = agent_id or "unknown_agent"
        
    async def upsert_with_telemetry(
        self, 
        collection_name: str, 
        record: MemoryRecord,
        memory_context: AgentMemoryContext = None
    ) -> str:
        """Upsert with telemetry events."""
        
        try:
            result = await super().upsert(collection_name, record)
            
            # ✅ Emit state management event for memory operations
            if memory_context:
                memory_data = {
                    "type": "local_agent_memory",
                    "agent_id": self.agent_id,
                    "collection": collection_name,
                    "record_id": record._id,
                    "is_reference": record._is_reference,
                    "external_source": record._external_source_name or "",
                    "conversation_id": memory_context.conversation_id,
                    "trace_id": memory_context.trace_id,
                    "span_id": memory_context.span_id,
                    "timestamp": datetime.now().isoformat(),
                    "operation": "upsert",
                    "success": True
                }
                
                # ✅ Log event using logger pattern from telemetry
                logger.info(
                    json.dumps(memory_data),
                    extra={
                        "event.name": STATE_MANAGEMENT_EVENT,
                        "gen_ai.system": SYSTEM,
                    }
                )
            
            return result
        except Exception as e:
            # ✅ Log error event
            if memory_context:
                error_data = {
                    "type": "local_agent_memory",
                    "agent_id": self.agent_id,
                    "collection": collection_name,
                    "operation": "upsert",
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.info(
                    json.dumps(error_data),
                    extra={
                        "event.name": STATE_MANAGEMENT_EVENT,
                        "gen_ai.system": SYSTEM,
                    }
                )
            raise


class SharedMemoryManager:
    """Manages shared memory across multiple agents with context propagation."""
    
    def __init__(self, embedding_generator: EmbeddingGeneratorBase = None):
        self.shared_store = EnhancedVolatileMemoryStore("shared_memory")
        self.agent_stores: Dict[str, EnhancedVolatileMemoryStore] = {}
        self.embedding_generator = embedding_generator
        self.memory_events: List[SharedMemoryEvent] = []
        
        # Store memory data as list of JSON objects
        self.state_management_data: List[Dict[str, Any]] = []
        
        # Initialize shared collections
        asyncio.create_task(self._initialize_shared_collections())
    
    async def _initialize_shared_collections(self):
        """Initialize shared memory collections."""
        collections = [
            "conversation_context",
            "agent_interactions", 
            "rag_documents",
            "planning_decisions",
            "booking_history"
        ]
        
        for collection in collections:
            await self.shared_store.create_collection(collection)
    
    def get_agent_memory(self, agent_id: str) -> EnhancedVolatileMemoryStore:
        """Get or create agent-specific memory store."""
        if agent_id not in self.agent_stores:
            self.agent_stores[agent_id] = EnhancedVolatileMemoryStore(agent_id)
            # Initialize agent-specific collections
            asyncio.create_task(self._initialize_agent_collections(agent_id))
        return self.agent_stores[agent_id]
    
    async def _initialize_agent_collections(self, agent_id: str):
        """Initialize agent-specific memory collections."""
        agent_store = self.agent_stores[agent_id]
        collections = [
            f"{agent_id}_working_memory",
            f"{agent_id}_learned_patterns",
            f"{agent_id}_interaction_history",
            f"{agent_id}_rag_knowledge"
        ]
        
        for collection in collections:
            await agent_store.create_collection(collection)
    
    async def propagate_context_to_agent(
        self, 
        source_agent: str, 
        target_agent: str, 
        context_data: Dict[str, Any],
        memory_context: AgentMemoryContext,
        chat_history: List[Dict[str, Any]] = None
    ) -> None:
        """Propagate context from one agent to another using OTEL Baggage."""
        
        # Create context propagation record
        transition_record = MemoryRecord(
            is_reference=False,
            external_source_name="agent_transition",
            id=f"transition_{uuid.uuid4()}",
            description=f"Context transition from {source_agent} to {target_agent}",
            text=json.dumps(context_data),
            additional_metadata=json.dumps({
                "source_agent": source_agent,
                "target_agent": target_agent,
                "conversation_id": memory_context.conversation_id,
                "trace_id": memory_context.trace_id,
                "span_id": memory_context.span_id,
            }),
            embedding=np.zeros(1536) if self.embedding_generator else np.array([0.0])
        )
        
        # Store in shared memory
        await self.shared_store.upsert_with_telemetry(
            "agent_interactions", 
            transition_record, 
            memory_context
        )
        
        # ✅ Add Context as Baggage Key-Value pair
        baggage_data = {
            "source_agent": source_agent,
            "target_agent": target_agent,
            "conversation_id": memory_context.conversation_id,
            "trace_id": memory_context.trace_id,
            "span_id": memory_context.span_id,
            "context_data": json.dumps(context_data)[:500],  # Truncate for baggage limits
            "transition_timestamp": datetime.now().isoformat()
        }
        
        # Set multiple baggage items for comprehensive context
        current_context = context.get_current()
        for key, value in baggage_data.items():
            current_context = baggage.set_baggage(f"agent_transition_{key}", str(value), current_context)
        
        # Attach the context
        token = attach(current_context)
        
        # ✅ 1. Emit STATE MANAGEMENT EVENT for context propagation
        context_propagation_data = {
            "type": "context_propagation",
            "source_agent": source_agent,
            "target_agent": target_agent,
            "conversation_id": memory_context.conversation_id,
            "trace_id": memory_context.trace_id,
            "span_id": memory_context.span_id,
            "baggage_keys": list(baggage_data.keys()),
            "context_size": len(json.dumps(context_data)),
            "chat_history": chat_history or [],
            "propagated_data": context_data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in memory data list
        self.state_management_data.append(context_propagation_data)
        
        # ✅ Log context propagation event
        logger.info(
            json.dumps(context_propagation_data),
            extra={
                "event.name": STATE_MANAGEMENT_EVENT,
                "gen_ai.system": SYSTEM,
            }
        )
        
        # ✅ 2. Emit AGENT INTERACTION EVENT
        interaction_data = {
            "source_agent": source_agent,
            "target_agent": target_agent,
            "reasoning_to_trigger": context_data.get("selection_reason", "Agent coordination decision"),
            "interaction_type": "context_propagation",
            "conversation_id": memory_context.conversation_id,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(
            json.dumps(interaction_data),
            extra={
                "event.name": AGENT_INTERACTION_EVENT,
                "gen_ai.system": SYSTEM,
            }
        )
        
        # Create memory event
        memory_event = SharedMemoryEvent(
            event_type="agent_transition",
            source_agent=source_agent,
            target_agent=target_agent,
            context_data=context_data
        )
        self.memory_events.append(memory_event)
        
        logger.info(
            f"Context propagated from {source_agent} to {target_agent}",
            extra={
                "conversation_id": memory_context.conversation_id,
                "trace_id": memory_context.trace_id,
                "context_size": len(json.dumps(context_data))
            }
        )
        
        return token
    
    async def store_rag_document(
        self, 
        document_content: str, 
        document_metadata: Dict[str, Any],
        agent_id: str,
        memory_context: AgentMemoryContext
    ) -> str:
        """Store RAG document and capture as reference memory."""
        
        # Generate embedding if available
        embedding = np.zeros(1536)  # Placeholder
        if self.embedding_generator:
            try:
                embedding_result = await self.embedding_generator.generate_embeddings([document_content])
                embedding = np.array(embedding_result[0])
            except Exception as e:
                logger.warning(f"Failed to generate embedding for RAG document: {e}")
        
        # Create RAG memory record
        rag_record = MemoryRecord(
            is_reference=True,  # Mark as reference memory
            external_source_name=document_metadata.get("source", "rag_system"),
            id=f"rag_{uuid.uuid4()}",
            description=f"RAG document for agent {agent_id}: {document_metadata.get('title', 'Untitled')}",
            text=document_content,
            additional_metadata=json.dumps({
                **document_metadata,
                "agent_id": agent_id,
                "storage_type": "rag_reference",
                "conversation_id": memory_context.conversation_id,
                "ingestion_timestamp": datetime.now().isoformat()
            }),
            embedding=embedding
        )
        
        # Store in both shared memory and agent-specific memory
        rag_key = await self.shared_store.upsert_with_telemetry(
            "rag_documents", 
            rag_record, 
            memory_context
        )
        
        agent_store = self.get_agent_memory(agent_id)
        await agent_store.upsert_with_telemetry(
            f"{agent_id}_rag_knowledge",
            rag_record,
            memory_context
        )
        
        # ✅ 1. Emit STATE MANAGEMENT EVENT for RAG knowledge
        rag_data = {
            "type": "rag_knowledge",
            "agent_id": agent_id,
            "document_key": rag_key,
            "document_title": document_metadata.get("title", ""),
            "document_type": document_metadata.get("type", "unknown"),
            "content_length": len(document_content),
            "storage_locations": ["shared_memory", "agent_memory"],
            "reference_memory": True,
            "conversation_id": memory_context.conversation_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": document_metadata
        }
        
        # Store in memory data list
        self.state_management_data.append(rag_data)
        
        # ✅ Log RAG knowledge event
        logger.info(
            json.dumps(rag_data),
            extra={
                "event.name": STATE_MANAGEMENT_EVENT,
                "gen_ai.system": SYSTEM,
            }
        )
        
        # Create memory event
        memory_event = SharedMemoryEvent(
            event_type="knowledge_enrichment",
            source_agent="rag_system",
            target_agent=agent_id,
            context_data={
                "document_key": rag_key,
                "document_metadata": document_metadata,
                "content_length": len(document_content)
            }
        )
        self.memory_events.append(memory_event)
        
        return rag_key
    
    async def retrieve_agent_context(self, agent_id: str) -> Dict[str, Any]:
        """Retrieve context information for an agent from baggage and memory."""
        
        # Extract from current baggage
        current_baggage = {}
        try:
            baggage_context = baggage.get_all()
            current_baggage = {k: v for k, v in baggage_context.items() if k.startswith("agent_transition_")}
        except Exception as e:
            logger.warning(f"Failed to retrieve baggage context: {e}")
        
        # Retrieve from agent memory
        agent_store = self.get_agent_memory(agent_id)
        try:
            collections = await agent_store.get_collections()
            agent_context = {
                "agent_id": agent_id,
                "available_collections": collections,
                "baggage_context": current_baggage,
                "memory_events": [
                    event for event in self.memory_events[-10:]  # Last 10 events
                    if event.target_agent == agent_id or event.source_agent == agent_id
                ]
            }
            
            # ✅ Emit STATE MANAGEMENT EVENT for local agent memory access
            local_memory_data = {
                "type": "local_agent_memory",
                "agent_id": agent_id,
                "available_collections": collections,
                "baggage_context": current_baggage,
                "memory_events_count": len(agent_context["memory_events"]),
                "operation": "retrieve_context",
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in memory data list
            self.state_management_data.append(local_memory_data)
            
            # ✅ Log local memory access event
            logger.info(
                json.dumps(local_memory_data),
                extra={
                    "event.name": STATE_MANAGEMENT_EVENT,
                    "gen_ai.system": SYSTEM,
                }
            )
            
            return agent_context
        except Exception as e:
            logger.error(f"Failed to retrieve agent context for {agent_id}: {e}")
            return {"agent_id": agent_id, "error": str(e)}
    
    def get_all_state_management_data(self) -> List[Dict[str, Any]]:
        """Get all state management data as list of JSON objects."""
        return self.state_management_data
    
    async def query_shared_memory(
        self, 
        query: str, 
        collection: str = "conversation_context",
        limit: int = 5
    ) -> List[Tuple[MemoryRecord, float]]:
        """Query shared memory with semantic search."""
        if not self.embedding_generator:
            logger.warning("No embedding generator available for semantic search")
            return []
        
        try:
            # Generate query embedding
            query_embeddings = await self.embedding_generator.generate_embeddings([query])
            query_embedding = np.array(query_embeddings[0])
            
            # Search shared memory
            results = await self.shared_store.get_nearest_matches(
                collection_name=collection,
                embedding=query_embedding,
                limit=limit,
                min_relevance_score=0.3
            )
            
            return results
        except Exception as e:
            logger.error(f"Failed to query shared memory: {e}")
            return []


# ✅ RAG Integration Touch Points
class RAGIntegrationManager:
    """Manages RAG document integration and knowledge enrichment."""
    
    def __init__(self, shared_memory: SharedMemoryManager):
        self.shared_memory = shared_memory
        
    async def enrich_agent_knowledge(
        self, 
        agent_id: str,
        documents: List[Dict[str, Any]],
        memory_context: AgentMemoryContext
    ) -> List[str]:
        """Bulk RAG document ingestion for agent knowledge enrichment."""
        
        stored_keys = []
        for i, doc in enumerate(documents):
            try:
                key = await self.shared_memory.store_rag_document(
                    document_content=doc["content"],
                    document_metadata=doc.get("metadata", {}),
                    agent_id=agent_id,
                    memory_context=memory_context
                )
                stored_keys.append(key)
                
            except Exception as e:
                logger.error(f"Failed to store document {i} for agent {agent_id}: {e}")
        
        return stored_keys
    
    async def contextual_rag_retrieval(
        self, 
        agent_id: str,
        query: str,
        memory_context: AgentMemoryContext,
        max_results: int = 3
    ) -> List[Dict[str, Any]]:
        """Context-aware RAG retrieval during agent operations."""
        
        # Search both shared RAG documents and agent-specific knowledge
        shared_results = await self.shared_memory.query_shared_memory(
            query=query,
            collection="rag_documents",
            limit=max_results
        )
        
        agent_store = self.shared_memory.get_agent_memory(agent_id)
        agent_results = []
        
        try:
            if self.shared_memory.embedding_generator:
                query_embeddings = await self.shared_memory.embedding_generator.generate_embeddings([query])
                query_embedding = np.array(query_embeddings[0])
                
                agent_results = await agent_store.get_nearest_matches(
                    collection_name=f"{agent_id}_rag_knowledge",
                    embedding=query_embedding,
                    limit=max_results,
                    min_relevance_score=0.3
                )
        except Exception as e:
            logger.warning(f"Failed to search agent-specific RAG: {e}")
        
        # Combine and format results
        all_results = []
        for record, score in shared_results + agent_results:
            result_dict = {
                "content": record._text,
                "metadata": json.loads(record._additional_metadata or "{}"),
                "relevance_score": float(score),
                "source": "shared" if record._id.startswith("rag_") else "agent_specific",
                "record_id": record._id
            }
            all_results.append(result_dict)
        
        # ✅ Emit STATE MANAGEMENT EVENT for RAG retrieval
        if all_results:
            rag_retrieval_data = {
                "type": "rag_knowledge",
                "agent_id": agent_id,
                "query": query[:100],  # Truncate
                "results_count": len(all_results),
                "operation": "retrieval",
                "conversation_id": memory_context.conversation_id,
                "timestamp": datetime.now().isoformat(),
                "retrieved_documents": [
                    {
                        "record_id": r["record_id"],
                        "relevance_score": r["relevance_score"],
                        "source": r["source"]
                    } for r in all_results
                ]
            }
            
            # Store in memory data list
            self.shared_memory.state_management_data.append(rag_retrieval_data)
            
            # ✅ Log RAG retrieval event
            logger.info(
                json.dumps(rag_retrieval_data),
                extra={
                    "event.name": STATE_MANAGEMENT_EVENT,
                    "gen_ai.system": SYSTEM,
                }
            )
        
        # Sort by relevance and deduplicate
        all_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        unique_results = []
        seen_content = set()
        
        for result in all_results:
            content_hash = hash(result["content"])
            if content_hash not in seen_content:
                unique_results.append(result)
                seen_content.add(content_hash)
            
            if len(unique_results) >= max_results:
                break
        
        return unique_results


# Global memory manager instance
_shared_memory_manager: Optional[SharedMemoryManager] = None

def get_shared_memory_manager() -> SharedMemoryManager:
    """Get the global shared memory manager instance."""
    global _shared_memory_manager
    if _shared_memory_manager is None:
        _shared_memory_manager = SharedMemoryManager()
    return _shared_memory_manager
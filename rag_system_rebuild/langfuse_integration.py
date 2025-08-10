"""
Langfuse integration for RAG System Rebuild.
Provides observability, tracing, and scoring capabilities.
Compatible with Langfuse 3.x
"""

import time
import uuid
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from contextlib import contextmanager

try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    print("Warning: Langfuse not available. Install with: pip install langfuse")

from config import LangfuseConfig


class LangfuseManager:
    """
    Manages Langfuse integration for the RAG system.
    Provides tracing, scoring, and observability capabilities.
    Compatible with Langfuse 3.x
    """
    
    def __init__(self, config: LangfuseConfig):
        """
        Initialize Langfuse manager.
        
        Args:
            config: Langfuse configuration
        """
        self.config = config
        self.client = None
        self.enabled = config.enabled and LANGFUSE_AVAILABLE
        
        if self.enabled:
            try:
                self.client = Langfuse(
                    public_key=config.public_key,
                    secret_key=config.secret_key,
                    host=config.host
                )
                print(f"✅ Langfuse initialized for project: {config.project_name}")
            except Exception as e:
                print(f"⚠️  Langfuse initialization failed: {e}")
                self.enabled = False
        else:
            if not LANGFUSE_AVAILABLE:
                print("⚠️  Langfuse package not available")
            elif not config.enabled:
                print("ℹ️  Langfuse disabled in configuration")
    
    def is_enabled(self) -> bool:
        """Check if Langfuse is enabled and available."""
        return self.enabled and self.client is not None
    
    def create_trace(self, 
                    name: str, 
                    user_id: Optional[str] = None,
                    session_id: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> Optional['TraceWrapper']:
        """
        Create a new trace using the current trace context.
        
        Args:
            name: Trace name
            user_id: User identifier
            session_id: Session identifier
            metadata: Additional metadata
            
        Returns:
            TraceWrapper instance or None if Langfuse is disabled
        """
        if not self.is_enabled():
            return None
        
        try:
            # Create a trace ID and set it as current
            trace_id = self.client.create_trace_id()
            return TraceWrapper(trace_id, self, name, user_id, session_id, metadata)
        except Exception as e:
            print(f"⚠️  Failed to create trace: {e}")
            return None
    
    def create_span(self, 
                   trace: 'TraceWrapper',
                   name: str,
                   input_data: Optional[Dict[str, Any]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> Optional['SpanWrapper']:
        """
        Create a span within the current trace context.
        
        Args:
            trace: Parent trace
            name: Span name
            input_data: Input data for the span
            metadata: Additional metadata
        """
        if not self.is_enabled():
            return None
        
        try:
            # Use the new Langfuse 3.x API with context manager
            span = self.client.start_as_current_span(
                name=name,
                input=input_data,
                metadata=metadata
            )
            return SpanWrapper(span, self)
        except Exception as e:
            print(f"⚠️  Failed to create span: {e}")
            return None
    
    def create_generation(self, 
                         trace: 'TraceWrapper',
                         name: str,
                         model: str,
                         prompt: str,
                         completion: str = "",
                         model_parameters: Optional[Dict[str, Any]] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> Optional['GenerationWrapper']:
        """
        Create a generation span within the current trace context.
        
        Args:
            trace: Parent trace
            name: Generation name
            model: Model name
            prompt: Input prompt
            completion: Generated completion
            model_parameters: Model parameters
            metadata: Additional metadata
        """
        if not self.is_enabled():
            return None
        
        try:
            # Use the new Langfuse 3.x API with context manager
            generation = self.client.start_as_current_generation(
                name=name,
                model=model,
                input=prompt,
                output=completion,
                model_parameters=model_parameters,
                metadata=metadata
            )
            return GenerationWrapper(generation, self)
        except Exception as e:
            print(f"⚠️  Failed to create generation: {e}")
            return None
    
    def score_generation(self, 
                        generation: 'GenerationWrapper',
                        name: str,
                        value: float,
                        comment: Optional[str] = None) -> bool:
        """
        Score a generation.
        
        Args:
            generation: Generation to score
            name: Score name
            value: Score value
            comment: Optional comment
            
        Returns:
            True if scoring was successful
        """
        if not self.is_enabled():
            return False
        
        try:
            # Note: Scoring functionality temporarily disabled due to API changes
            # TODO: Implement proper scoring when Langfuse API is updated
            print(f"📊 Would score generation '{name}' with value {value}: {comment}")
            return True
        except Exception as e:
            print(f"⚠️  Failed to score generation: {e}")
            return False
    
    def flush(self):
        """Flush pending telemetry data."""
        if self.is_enabled():
            try:
                self.client.flush()
            except Exception as e:
                print(f"⚠️  Failed to flush: {e}")
    
    @contextmanager
    def traced_operation(self,
                        operation_name: str,
                        user_id: Optional[str] = None,
                        session_id: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for traced operations.
        
        Args:
            operation_name: Name of the operation
            user_id: User identifier
            session_id: Session identifier
            metadata: Additional metadata
        """
        if not self.is_enabled():
            yield None
            return
        
        try:
            # Start a span for the operation
            span = self.client.start_as_current_span(
                name=operation_name,
                metadata=metadata
            )
            
            # Create a wrapper for easier use
            wrapper = SpanWrapper(span, self)
            
            yield wrapper
            
            # End the span
            wrapper.end()
            
        except Exception as e:
            print(f"⚠️  Error in traced operation: {e}")
            yield None


class TraceWrapper:
    """Wrapper for Langfuse traces."""
    
    def __init__(self, trace_id: str, langfuse_manager: LangfuseManager, name: str, user_id: Optional[str], session_id: Optional[str], metadata: Optional[Dict[str, Any]]):
        self.trace_id = trace_id
        self.langfuse_manager = langfuse_manager
        self.name = name
        self.user_id = user_id
        self.session_id = session_id
        self.metadata = metadata or {}
        self.start_time = time.time()
    
    def end(self, 
            output: Optional[Dict[str, Any]] = None,
            status_message: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None):
        """End the trace."""
        if self.langfuse_manager.is_enabled():
            try:
                duration = (time.time() - self.start_time) * 1000
                print(f"📊 Trace '{self.name}' completed in {duration:.2f}ms")
                
                # Update the current trace with output and metadata
                if output or metadata:
                    self.langfuse_manager.client.update_current_trace(
                        output=output,
                        metadata=metadata
                    )
                    
            except Exception as e:
                print(f"⚠️  Failed to end trace: {e}")
    
    def add_span(self, span: 'SpanWrapper'):
        """Add a span to the trace."""
        # In Langfuse 3.x, spans are automatically associated with the current trace
        pass
    
    def add_generation(self, generation: 'GenerationWrapper'):
        """Add a generation to the trace."""
        # In Langfuse 3.x, generations are automatically associated with the current trace
        pass


class SpanWrapper:
    """Wrapper for Langfuse spans."""
    
    def __init__(self, span, langfuse_manager: LangfuseManager):
        self.span = span
        self.langfuse_manager = langfuse_manager
    
    def end(self, 
            output: Optional[Dict[str, Any]] = None,
            metadata: Optional[Dict[str, Any]] = None):
        """End the span."""
        if self.langfuse_manager.is_enabled():
            try:
                # In Langfuse 3.x, spans are automatically ended when exiting context
                # We can update the current span with output and metadata
                if output or metadata:
                    self.langfuse_manager.client.update_current_span(
                        output=output,
                        metadata=metadata
                    )
            except Exception as e:
                print(f"⚠️  Failed to end span: {e}")


class GenerationWrapper:
    """Wrapper for Langfuse generations."""
    
    def __init__(self, generation, langfuse_manager: LangfuseManager):
        self.generation = generation
        self.langfuse_manager = langfuse_manager
    
    def update(self, completion: str, metadata: Optional[Dict[str, Any]] = None):
        """Update the generation with completion text."""
        if self.langfuse_manager.is_enabled():
            try:
                self.langfuse_manager.client.update_current_generation(
                    output=completion,
                    metadata=metadata
                )
            except Exception as e:
                print(f"⚠️  Failed to update generation: {e}")
    
    def end(self, metadata: Optional[Dict[str, Any]] = None):
        """End the generation."""
        if self.langfuse_manager.is_enabled():
            try:
                if metadata:
                    self.langfuse_manager.client.update_current_generation(
                        metadata=metadata
                    )
            except Exception as e:
                print(f"⚠️  Failed to end generation: {e}")
    
    def add_score(self, name: str, value: float, comment: Optional[str] = None):
        """Add a score to the generation."""
        if self.langfuse_manager.is_enabled():
            try:
                # Note: Scoring functionality temporarily disabled due to API changes
                # TODO: Implement proper scoring when Langfuse API is updated
                print(f"📊 Would add score '{name}' with value {value}: {comment}")
            except Exception as e:
                print(f"⚠️  Failed to add score: {e}")


@contextmanager
def traced_operation(langfuse_manager: LangfuseManager,
                    operation_name: str,
                    user_id: Optional[str] = None,
                    session_id: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None):
    """
    Context manager for traced operations.
    
    Args:
        langfuse_manager: Langfuse manager instance
        operation_name: Name of the operation
        user_id: User identifier
        session_id: Session identifier
        metadata: Additional metadata
    """
    if not langfuse_manager or not langfuse_manager.is_enabled():
        yield None
        return
    
    try:
        # Start a span for the operation
        span = langfuse_manager.client.start_as_current_span(
            name=operation_name,
            metadata=metadata
        )
        
        # Create a wrapper for easier use
        wrapper = SpanWrapper(span, langfuse_manager)
        
        yield wrapper
        
        # End the span
        wrapper.end()
        
    except Exception as e:
        print(f"⚠️  Error in traced operation: {e}")
        yield None


def create_rag_trace(langfuse_manager: LangfuseManager,
                    query: str,
                    user_id: Optional[str] = None,
                    session_id: Optional[str] = None) -> Optional[TraceWrapper]:
    """
    Create a RAG-specific trace.
    
    Args:
        langfuse_manager: Langfuse manager instance
        query: User query
        user_id: User identifier
        session_id: Session identifier
        
    Returns:
        TraceWrapper instance or None
    """
    if not langfuse_manager or not langfuse_manager.is_enabled():
        return None
    
    try:
        trace_id = langfuse_manager.client.create_trace_id()
        return TraceWrapper(
            trace_id,
            langfuse_manager,
            "rag-query",
            user_id,
            session_id,
            {
                "query": query,
                "operation": "rag_query",
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        print(f"⚠️  Failed to create RAG trace: {e}")
        return None


def score_rag_response(generation: GenerationWrapper,
                      relevance_score: float,
                      helpfulness_score: float,
                      accuracy_score: float,
                      response_length: int) -> bool:
    """
    Score a RAG response with multiple metrics.
    
    Args:
        generation: Generation wrapper
        relevance_score: Relevance score (0-1)
        helpfulness_score: Helpfulness score (0-1)
        accuracy_score: Accuracy score (0-1)
        response_length: Length of the response
        
    Returns:
        True if scoring was successful
    """
    try:
        # Add multiple scores
        generation.add_score("relevance", relevance_score, "Relevance to the query")
        generation.add_score("helpfulness", helpfulness_score, "How helpful the response is")
        generation.add_score("accuracy", accuracy_score, "Accuracy of the information")
        generation.add_score("response_length", min(1.0, response_length / 1000), "Response length score")
        
        return True
    except Exception as e:
        print(f"⚠️  Failed to score RAG response: {e}")
        return False 
"""
Basic keyword-based router for adapter selection.
Provides fast routing based on keyword matching with TF-IDF scoring.
"""

import logging
import re
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
import math
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class KeywordRule:
    """Keyword matching rule for an adapter."""
    adapter_name: str
    keywords: List[str]
    weight: float = 1.0
    required_keywords: List[str] = None
    excluded_keywords: List[str] = None


@dataclass
class KeywordRoutingResult:
    """Result of keyword-based routing."""
    primary_adapter: str
    confidence: float
    matched_keywords: List[str]
    reasoning: str


class KeywordRouter:
    """
    Fast keyword-based router using TF-IDF scoring.
    
    Features:
    - Keyword matching with configurable rules
    - TF-IDF scoring for relevance
    - Required/excluded keyword support
    - Confidence thresholds
    - Fast routing for real-time applications
    """
    
    def __init__(self, confidence_threshold: float = 0.3):
        """
        Initialize keyword router.
        
        Args:
            confidence_threshold: Minimum confidence for adapter selection
        """
        self.confidence_threshold = confidence_threshold
        self.routing_rules: Dict[str, KeywordRule] = {}
        
        # TF-IDF components
        self.vocabulary: Set[str] = set()
        self.document_frequency: Dict[str, int] = defaultdict(int)
        self.total_documents = 0
        
        # Default routing rules
        self._initialize_default_rules()
        
        logger.info("Keyword router initialized")
    
    def _initialize_default_rules(self):
        """Initialize default routing rules for common domains."""
        default_rules = [
            KeywordRule(
                adapter_name="math_reasoning",
                keywords=[
                    "calculate", "solve", "equation", "number", "arithmetic", "algebra",
                    "geometry", "calculus", "derivative", "integral", "sum", "product",
                    "fraction", "percentage", "ratio", "probability", "statistics",
                    "formula", "theorem", "proof", "mathematical", "numeric"
                ],
                weight=1.0,
                required_keywords=None,
                excluded_keywords=["code", "programming", "function", "variable"]
            ),
            KeywordRule(
                adapter_name="code_generation",
                keywords=[
                    "python", "function", "algorithm", "programming", "debug", "code",
                    "script", "class", "method", "variable", "loop", "condition",
                    "import", "library", "framework", "api", "database", "sql",
                    "javascript", "html", "css", "react", "node", "git"
                ],
                weight=1.0,
                required_keywords=None,
                excluded_keywords=["math", "calculate", "equation"]
            ),
            KeywordRule(
                adapter_name="reasoning",
                keywords=[
                    "analyze", "logic", "argument", "conclude", "reasoning", "because",
                    "therefore", "however", "although", "evidence", "premise",
                    "conclusion", "inference", "deduction", "induction", "hypothesis",
                    "theory", "explain", "justify", "evaluate", "compare"
                ],
                weight=1.0
            ),
            KeywordRule(
                adapter_name="creative_writing",
                keywords=[
                    "story", "write", "creative", "narrative", "character", "plot",
                    "dialogue", "scene", "chapter", "novel", "poem", "poetry",
                    "fiction", "fantasy", "romance", "mystery", "adventure",
                    "describe", "imagine", "create", "invent", "compose"
                ],
                weight=1.0
            )
        ]
        
        for rule in default_rules:
            self.add_routing_rule(rule)
    
    def add_routing_rule(self, rule: KeywordRule) -> bool:
        """
        Add a routing rule for an adapter.
        
        Args:
            rule: KeywordRule defining the routing logic
            
        Returns:
            True if rule added successfully
        """
        try:
            self.routing_rules[rule.adapter_name] = rule
            
            # Update vocabulary and document frequency
            self._update_vocabulary(rule.keywords)
            
            logger.info(f"Added routing rule for {rule.adapter_name} with {len(rule.keywords)} keywords")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add routing rule: {e}")
            return False
    
    def _update_vocabulary(self, keywords: List[str]):
        """Update vocabulary and document frequency for TF-IDF."""
        for keyword in keywords:
            keyword_lower = keyword.lower()
            self.vocabulary.add(keyword_lower)
            self.document_frequency[keyword_lower] += 1
        
        self.total_documents = len(self.routing_rules)
    
    def route_query(self, query: str, available_adapters: List[str] = None) -> KeywordRoutingResult:
        """
        Route query using keyword matching.
        
        Args:
            query: Input query to route
            available_adapters: List of available adapters (None for all)
            
        Returns:
            KeywordRoutingResult with routing decision
        """
        try:
            # Preprocess query
            query_tokens = self._preprocess_query(query)
            
            # Filter available adapters
            if available_adapters is None:
                available_rules = self.routing_rules
            else:
                available_rules = {
                    name: rule for name, rule in self.routing_rules.items()
                    if name in available_adapters
                }
            
            if not available_rules:
                raise ValueError("No available adapters for routing")
            
            # Calculate scores for each adapter
            adapter_scores = {}
            adapter_matches = {}
            
            for adapter_name, rule in available_rules.items():
                score, matches = self._calculate_adapter_score(query_tokens, rule)
                adapter_scores[adapter_name] = score
                adapter_matches[adapter_name] = matches
            
            # Select best adapter
            if not adapter_scores:
                # Fallback to first available adapter
                primary_adapter = list(available_rules.keys())[0]
                confidence = 0.0
                matched_keywords = []
                reasoning = "Fallback selection - no keyword matches found"
            else:
                primary_adapter = max(adapter_scores.keys(), key=lambda x: adapter_scores[x])
                confidence = adapter_scores[primary_adapter]
                matched_keywords = adapter_matches[primary_adapter]
                reasoning = self._generate_reasoning(primary_adapter, confidence, matched_keywords)
            
            return KeywordRoutingResult(
                primary_adapter=primary_adapter,
                confidence=confidence,
                matched_keywords=matched_keywords,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Failed to route query: {e}")
            # Return fallback result
            fallback_adapter = available_adapters[0] if available_adapters else "base_model"
            return KeywordRoutingResult(
                primary_adapter=fallback_adapter,
                confidence=0.0,
                matched_keywords=[],
                reasoning=f"Fallback routing due to error: {str(e)}"
            )
    
    def _preprocess_query(self, query: str) -> List[str]:
        """Preprocess query into tokens."""
        # Convert to lowercase and extract words
        query_lower = query.lower()
        
        # Remove punctuation and split
        tokens = re.findall(r'\b\w+\b', query_lower)
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        return tokens
    
    def _calculate_adapter_score(self, query_tokens: List[str], rule: KeywordRule) -> Tuple[float, List[str]]:
        """Calculate TF-IDF based score for an adapter rule."""
        matched_keywords = []
        keyword_scores = []
        
        # Check required keywords
        if rule.required_keywords:
            required_matches = sum(1 for req in rule.required_keywords 
                                 if req.lower() in [token.lower() for token in query_tokens])
            if required_matches == 0:
                return 0.0, []
        
        # Check excluded keywords
        if rule.excluded_keywords:
            excluded_matches = sum(1 for exc in rule.excluded_keywords 
                                 if exc.lower() in [token.lower() for token in query_tokens])
            if excluded_matches > 0:
                return 0.0, []
        
        # Calculate TF-IDF scores for matched keywords
        query_token_counts = Counter(query_tokens)
        
        for keyword in rule.keywords:
            keyword_lower = keyword.lower()
            
            # Check if keyword matches any query token
            for token in query_tokens:
                if keyword_lower == token or keyword_lower in token or token in keyword_lower:
                    matched_keywords.append(keyword)
                    
                    # Calculate TF-IDF score
                    tf = query_token_counts[token] / len(query_tokens)
                    idf = math.log(self.total_documents / (self.document_frequency[keyword_lower] + 1))
                    tfidf_score = tf * idf
                    
                    keyword_scores.append(tfidf_score)
                    break
        
        # Calculate final score
        if not keyword_scores:
            return 0.0, []
        
        # Combine scores (average with bonus for multiple matches)
        base_score = sum(keyword_scores) / len(rule.keywords)  # Normalize by total keywords
        match_bonus = min(len(matched_keywords) / len(rule.keywords), 0.5)  # Bonus for coverage
        final_score = (base_score + match_bonus) * rule.weight
        
        return min(final_score, 1.0), list(set(matched_keywords))  # Cap at 1.0
    
    def _generate_reasoning(self, adapter_name: str, confidence: float, matched_keywords: List[str]) -> str:
        """Generate reasoning for the routing decision."""
        reasoning_parts = [
            f"Selected '{adapter_name}' with {confidence:.1%} confidence."
        ]
        
        if matched_keywords:
            if len(matched_keywords) == 1:
                reasoning_parts.append(f"Matched keyword: '{matched_keywords[0]}'.")
            else:
                reasoning_parts.append(f"Matched {len(matched_keywords)} keywords: {', '.join(matched_keywords[:5])}.")
        
        if confidence > 0.7:
            reasoning_parts.append("High keyword relevance.")
        elif confidence > 0.4:
            reasoning_parts.append("Moderate keyword relevance.")
        else:
            reasoning_parts.append("Low keyword relevance.")
        
        return " ".join(reasoning_parts)
    
    def get_adapter_keywords(self, adapter_name: str) -> List[str]:
        """Get keywords for a specific adapter."""
        if adapter_name in self.routing_rules:
            return self.routing_rules[adapter_name].keywords.copy()
        return []
    
    def update_adapter_keywords(self, adapter_name: str, keywords: List[str]) -> bool:
        """Update keywords for an existing adapter."""
        if adapter_name not in self.routing_rules:
            return False
        
        try:
            rule = self.routing_rules[adapter_name]
            rule.keywords = keywords
            self._update_vocabulary(keywords)
            
            logger.info(f"Updated keywords for {adapter_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update keywords: {e}")
            return False
    
    def save_rules(self, filepath: str) -> bool:
        """Save routing rules to file."""
        try:
            rules_data = {}
            for name, rule in self.routing_rules.items():
                rules_data[name] = {
                    'adapter_name': rule.adapter_name,
                    'keywords': rule.keywords,
                    'weight': rule.weight,
                    'required_keywords': rule.required_keywords,
                    'excluded_keywords': rule.excluded_keywords
                }
            
            with open(filepath, 'w') as f:
                json.dump(rules_data, f, indent=2)
            
            logger.info(f"Saved {len(rules_data)} routing rules to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save rules: {e}")
            return False
    
    def load_rules(self, filepath: str) -> bool:
        """Load routing rules from file."""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Rules file not found: {filepath}")
                return False
            
            with open(filepath, 'r') as f:
                rules_data = json.load(f)
            
            for name, data in rules_data.items():
                rule = KeywordRule(
                    adapter_name=data['adapter_name'],
                    keywords=data['keywords'],
                    weight=data.get('weight', 1.0),
                    required_keywords=data.get('required_keywords'),
                    excluded_keywords=data.get('excluded_keywords')
                )
                self.add_routing_rule(rule)
            
            logger.info(f"Loaded {len(rules_data)} routing rules from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load rules: {e}")
            return False

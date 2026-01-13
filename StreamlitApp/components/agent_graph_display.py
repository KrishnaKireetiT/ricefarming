"""
Component for displaying the agent graph structure in Langfuse-style format.
"""

import streamlit as st
import streamlit.components.v1 as components
from typing import Dict, Any, List, Optional
import json


# Default agent graph structure (matches the notebook pipeline)
DEFAULT_AGENT_GRAPH = {
    "name": "Rice_Farming_Advisor",
    "type": "agent",
    "children": [
        {
            "name": "Extract_Question_Entities",
            "type": "generation",
            "description": "LLM extracts key entities from the question"
        },
        {
            "name": "Align_Entities_To_KG",
            "type": "chain",
            "description": "Align extracted entities to KG using embedding similarity"
        },
        {
            "name": "Multi_Source_Evidence_Retrieval",
            "type": "chain",
            "description": "Parallel retrieval from multiple sources",
            "children": [
                {
                    "name": "KG_Graph_Traversal",
                    "type": "tool",
                    "description": "Traverse knowledge graph for related facts"
                },
                {
                    "name": "Vector_Semantic_Search",
                    "type": "tool",
                    "description": "Semantic search over chunks and visuals"
                },
                {
                    "name": "Keyword_Fulltext_Search",
                    "type": "tool",
                    "description": "Fulltext keyword search over chunks"
                }
            ]
        },
        {
            "name": "Aggregate_Evidence_Hybrid_Search",
            "type": "chain",
            "description": "Merge and deduplicate evidence from all sources"
        },
        {
            "name": "Generate_Farmer_Answer",
            "type": "generation",
            "description": "LLM generates farmer-friendly answer from context"
        }
    ]
}


def _get_type_color(node_type: str) -> dict:
    """Get color scheme for node type."""
    colors = {
        "agent": {"bg": "#e3f2fd", "border": "#1565c0", "text": "#1565c0"},
        "chain": {"bg": "#f3e5f5", "border": "#7b1fa2", "text": "#7b1fa2"},
        "tool": {"bg": "#fff3e0", "border": "#ef6c00", "text": "#ef6c00"},
        "generation": {"bg": "#e8f5e9", "border": "#2e7d32", "text": "#2e7d32"},
    }
    return colors.get(node_type.lower(), {"bg": "#eceff1", "border": "#607d8b", "text": "#607d8b"})


def _generate_graph_html(graph_data: Dict[str, Any], execution_data: Dict[str, Any] = None) -> str:
    """
    Generate HTML for Langfuse-style agent graph visualization.
    
    Args:
        graph_data: Static graph structure
        execution_data: Optional execution data with timing info
        
    Returns:
        HTML string for the graph visualization
    """
    css = """
    <style>
        .graph-container {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            padding: 20px;
        }
        .graph-node {
            display: flex;
            align-items: flex-start;
            margin: 8px 0;
        }
        .node-connector {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-right: 8px;
        }
        .connector-line {
            width: 2px;
            height: 20px;
            background: #e0e0e0;
        }
        .connector-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #e0e0e0;
        }
        .node-content {
            flex: 1;
            border-radius: 8px;
            padding: 12px 16px;
            border: 2px solid;
            transition: all 0.2s ease;
        }
        .node-content:hover {
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transform: translateX(4px);
        }
        .node-header {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .node-type-badge {
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .node-name {
            font-weight: 600;
            font-size: 14px;
            color: #333;
        }
        .node-time {
            margin-left: auto;
            font-size: 12px;
            color: #666;
            font-weight: 500;
        }
        .node-description {
            margin-top: 6px;
            font-size: 12px;
            color: #666;
        }
        .children-container {
            margin-left: 30px;
            border-left: 2px dashed #e0e0e0;
            padding-left: 15px;
        }
        .parallel-indicator {
            font-size: 10px;
            color: #666;
            padding: 2px 6px;
            background: #f5f5f5;
            border-radius: 4px;
            margin-left: 8px;
        }
    </style>
    """
    
    def render_node(node: Dict, level: int = 0, is_last: bool = True) -> str:
        name = node.get("name", "Unknown")
        node_type = node.get("type", "span")
        description = node.get("description", "")
        children = node.get("children", [])
        
        colors = _get_type_color(node_type)
        
        # Get execution time if available
        time_str = ""
        if execution_data and name in execution_data:
            duration = execution_data[name].get("duration_ms", 0)
            if duration >= 1000:
                time_str = f"⏱ {duration/1000:.2f}s"
            else:
                time_str = f"⏱ {duration}ms"
        
        # Check if children are parallel (tools under retrieval)
        parallel = len(children) > 1 and all(c.get("type") == "tool" for c in children)
        
        html = f"""
        <div class="graph-node">
            <div class="node-content" style="background: {colors['bg']}; border-color: {colors['border']};">
                <div class="node-header">
                    <span class="node-type-badge" style="background: {colors['border']}; color: white;">
                        {node_type}
                    </span>
                    <span class="node-name">{name}</span>
                    {"<span class='parallel-indicator'>⚡ parallel</span>" if parallel else ""}
                    <span class="node-time">{time_str}</span>
                </div>
                {"<div class='node-description'>" + description + "</div>" if description else ""}
            </div>
        </div>
        """
        
        if children:
            html += '<div class="children-container">'
            for i, child in enumerate(children):
                html += render_node(child, level + 1, i == len(children) - 1)
            html += '</div>'
        
        return html
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>{css}</head>
    <body>
        <div class="graph-container">
            {render_node(graph_data)}
        </div>
    </body>
    </html>
    """


def display_agent_graph(
    execution_data: Dict[str, Any] = None,
    graph_structure: Dict[str, Any] = None,
    height: int = 450
):
    """
    Display the agent graph structure with execution timing.
    
    Args:
        execution_data: Dict mapping node names to execution info
        graph_structure: Custom graph structure (uses default if None)
        height: Height of the component
    """
    st.markdown("### 🕸️ Agent Graph")
    
    graph = graph_structure or DEFAULT_AGENT_GRAPH
    html = _generate_graph_html(graph, execution_data)
    components.html(html, height=height, scrolling=True)


def display_agent_graph_text():
    """Display a simple text-based agent graph."""
    st.markdown("### 🕸️ Agent Graph Structure")
    
    graph_text = """
```
Rice_Farming_Advisor (agent)
├─ Extract_Question_Entities (generation)
├─ Align_Entities_To_KG (chain)
├─ Multi_Source_Evidence_Retrieval (chain)
│  ├─ KG_Graph_Traversal (tool)      ⚡ parallel
│  ├─ Vector_Semantic_Search (tool)  ⚡ parallel
│  └─ Keyword_Fulltext_Search (tool) ⚡ parallel
├─ Aggregate_Evidence_Hybrid_Search (chain)
└─ Generate_Farmer_Answer (generation)
```
    """
    st.markdown(graph_text)


def build_execution_data_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build execution data dict from pipeline result for graph display.
    
    Args:
        result: Pipeline result dictionary
        
    Returns:
        Dict mapping node names to timing info
    """
    # This would be populated from actual trace data
    # For now, return empty dict (times will not be shown)
    return {}

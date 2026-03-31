"""
Component for displaying Langfuse trace visualization.
Fetches trace data from Langfuse API and renders it in Langfuse-style format.
"""

import streamlit as st
import streamlit.components.v1 as components
from typing import Dict, Any, Optional
import json

import config


def display_trace_link(trace_id: str, trace_url: str):
    """
    Display a link to the Langfuse trace.
    
    Args:
        trace_id: The trace ID
        trace_url: Full URL to the trace
    """
    st.markdown("### 📊 Langfuse Trace")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.code(trace_id, language=None)
    with col2:
        st.link_button("Open in Langfuse", trace_url, use_container_width=True)


def _generate_trace_html(trace_data: Dict[str, Any]) -> str:
    """
    Generate HTML for Langfuse-style trace visualization.
    
    Args:
        trace_data: Trace data with observations
        
    Returns:
        HTML string for the trace visualization
    """
    # CSS styles matching Langfuse UI
    css = """
    <style>
        .trace-container {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-color, #ffffff);
            border-radius: 8px;
            padding: 16px;
            color: var(--text-color, #1a1a1a);
        }
        .trace-node {
            margin-left: 20px;
            padding: 8px 12px;
            border-left: 2px solid #e0e0e0;
            margin-bottom: 4px;
        }
        .trace-node:hover {
            background: #f5f5f5;
            border-radius: 4px;
        }
        .node-header {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .node-type {
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }
        .type-agent { background: #e3f2fd; color: #1565c0; }
        .type-chain { background: #f3e5f5; color: #7b1fa2; }
        .type-tool { background: #fff3e0; color: #ef6c00; }
        .type-generation { background: #e8f5e9; color: #2e7d32; }
        .node-name {
            font-weight: 500;
            font-size: 14px;
        }
        .node-time {
            color: #666;
            font-size: 12px;
            margin-left: auto;
        }
        .node-details {
            margin-top: 8px;
            padding: 8px;
            background: #fafafa;
            border-radius: 4px;
            font-size: 12px;
        }
        .node-io {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-top: 8px;
        }
        .io-section {
            padding: 8px;
            background: #f5f5f5;
            border-radius: 4px;
        }
        .io-label {
            font-weight: 600;
            font-size: 11px;
            color: #666;
            margin-bottom: 4px;
        }
        .io-content {
            font-size: 12px;
            white-space: pre-wrap;
            max-height: 100px;
            overflow-y: auto;
        }
        .tree-line {
            border-left: 2px solid #e0e0e0;
            margin-left: 10px;
        }
        .expand-btn {
            cursor: pointer;
            color: #1976d2;
            font-size: 11px;
        }
    </style>
    """
    
    def render_node(node: Dict, level: int = 0) -> str:
        """Recursively render a trace node."""
        node_type = node.get("type", "span").lower()
        name = node.get("name", "Unknown")
        duration = node.get("duration_ms", 0)
        input_data = node.get("input", {})
        output_data = node.get("output", {})
        children = node.get("children", [])
        
        indent = level * 20
        type_class = f"type-{node_type}" if node_type in ["agent", "chain", "tool", "generation"] else ""
        
        # Format duration
        if duration >= 1000:
            time_str = f"{duration/1000:.2f}s"
        else:
            time_str = f"{duration}ms"
        
        html = f"""
        <div class="trace-node" style="margin-left: {indent}px;">
            <div class="node-header">
                <span class="node-type {type_class}">{node_type}</span>
                <span class="node-name">{name}</span>
                <span class="node-time">⏱ {time_str}</span>
            </div>
        """
        
        # Add input/output if present
        if input_data or output_data:
            html += '<div class="node-io">'
            if input_data:
                input_str = json.dumps(input_data, indent=2, ensure_ascii=False)[:200]
                html += f'''
                <div class="io-section">
                    <div class="io-label">INPUT</div>
                    <div class="io-content">{input_str}</div>
                </div>
                '''
            if output_data:
                output_str = json.dumps(output_data, indent=2, ensure_ascii=False)[:200]
                html += f'''
                <div class="io-section">
                    <div class="io-label">OUTPUT</div>
                    <div class="io-content">{output_str}</div>
                </div>
                '''
            html += '</div>'
        
        html += '</div>'
        
        # Render children
        for child in children:
            html += render_node(child, level + 1)
        
        return html
    
    # Build the full HTML
    nodes_html = render_node(trace_data)
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        {css}
    </head>
    <body>
        <div class="trace-container">
            {nodes_html}
        </div>
    </body>
    </html>
    """


def display_trace_embedded(trace_id: str, trace_url: str, height: int = 500, cached_trace_data: Dict[str, Any] = None):
    """
    Display embedded Langfuse trace visualization.
    Uses cached data if available, otherwise falls back to API fetch.
    
    Args:
        trace_id: The trace ID
        trace_url: Full URL to the trace
        height: Height of the embedded component
        cached_trace_data: Optional pre-fetched trace data from database
    """
    st.markdown("### 📊 Langfuse Trace")
    
    # First show the trace link
    col1, col2 = st.columns([3, 1])
    with col1:
        st.code(trace_id, language=None)
    with col2:
        st.link_button("Open in Langfuse ↗", trace_url, use_container_width=True)
    
    # Try to use cached data first
    if cached_trace_data and cached_trace_data.get("observations"):
        try:
            # Build tree structure from cached observations
            trace_data = _build_trace_tree_from_dict(cached_trace_data)
            html = _generate_trace_html(trace_data)
            components.html(html, height=height, scrolling=True)
            return
        except Exception as e:
            st.warning(f"Failed to render cached trace: {str(e)[:100]}")
    
    # Fallback: Try to fetch from API
    if not trace_id or not trace_id.strip():
        st.info("Trace ID not available. Langfuse tracing may not have captured this run.")
        return

    try:
        from langfuse import Langfuse
        
        langfuse = Langfuse(timeout=config.LANGFUSE_TIMEOUT)
        trace = langfuse.api.trace.get(trace_id)
        
        if trace and hasattr(trace, 'observations'):
            # Build tree structure from observations
            trace_data = _build_trace_tree(trace)
            html = _generate_trace_html(trace_data)
            components.html(html, height=height, scrolling=True)
        else:
            st.info("Trace data not yet available. Click the link above to view in Langfuse.")
            
    except Exception as e:
        # Final fallback: show message
        st.warning(f"Could not fetch trace details: {str(e)[:100]}")
        st.info("The trace is being processed. View it directly in Langfuse:")
        

def _build_trace_tree_from_dict(trace_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a hierarchical tree structure from cached trace dictionary.
    
    Args:
        trace_dict: Cached trace data from database
        
    Returns:
        Hierarchical dict representation of the trace
    """
    import logging
    logger = logging.getLogger(__name__)
    
    observations = trace_dict.get("observations", [])
    
    logger.info(f"Building trace tree from {len(observations)} observations")
    
    if not observations:
        return {
            "name": trace_dict.get("name", "Rice_Farming_Advisor"),
            "type": "agent",
            "duration_ms": 0,
            "input": {},
            "output": {},
            "children": []
        }
    
    # Create lookup by ID
    obs_map = {}
    for idx, obs in enumerate(observations):
        from datetime import datetime
        
        obs_id = obs.get("id")
        if not obs_id:
            logger.warning(f"Observation {idx} has no ID, skipping")
            continue
        
        # Calculate duration
        duration_ms = 0
        if obs.get("start_time") and obs.get("end_time"):
            try:
                start = datetime.fromisoformat(obs["start_time"].replace('Z', '+00:00'))
                end = datetime.fromisoformat(obs["end_time"].replace('Z', '+00:00'))
                duration_ms = int((end - start).total_seconds() * 1000)
            except Exception as e:
                logger.warning(f"Failed to parse timestamps: {e}")
        
        obs_dict = {
            "id": obs_id,
            "name": obs.get("name", "Unknown"),
            "type": obs.get("type", "span"),
            "input": obs.get("input", {}),
            "output": obs.get("output", {}),
            "duration_ms": duration_ms,
            "parent_id": obs.get("parent_observation_id"),
            "start_time": obs.get("start_time", ""),
            "idx": idx,
            "children": []
        }
        obs_map[obs_id] = obs_dict
        logger.debug(f"Obs {idx}: {obs_dict['name']} (id={obs_id}, parent={obs_dict['parent_id']})")
    
    # Build tree by linking children to parents
    root_nodes = []
    for obs_id, obs in obs_map.items():
        parent_id = obs.get("parent_id")
        if parent_id and parent_id in obs_map:
            # This observation has a parent - add as child
            obs_map[parent_id]["children"].append(obs)
            logger.debug(f"Linked {obs['name']} as child of {obs_map[parent_id]['name']}")
        else:
            # This is a root node
            root_nodes.append(obs)
            if parent_id:
                logger.warning(f"Observation {obs['name']} has parent_id {parent_id} but parent not found")
    
    logger.info(f"Found {len(root_nodes)} root nodes")
    
    # Sort children by start_time so steps appear in chronological order
    for obs in obs_map.values():
        if obs["children"]:
            obs["children"].sort(key=lambda c: (c.get("start_time", ""), c.get("idx", 0)))
            logger.debug(f"{obs['name']} has {len(obs['children'])} children")

    # Return the root (usually the agent)
    if root_nodes:
        root_nodes.sort(key=lambda c: (c.get("start_time", ""), c.get("idx", 0)))
        root = root_nodes[0]
        logger.info(f"Returning root: {root['name']} with {len(root['children'])} children")
        return root
    
    # Fallback
    logger.warning("No root nodes found, returning empty tree")
    return {
        "name": "Rice_Farming_Advisor",
        "type": "agent",
        "duration_ms": 0,
        "input": {},
        "output": {},
        "children": []
    }


def _build_trace_tree(trace) -> Dict[str, Any]:
    """
    Build a hierarchical tree structure from flat Langfuse observations.
    
    Args:
        trace: Langfuse trace object
        
    Returns:
        Hierarchical dict representation of the trace
    """
    observations = trace.observations if hasattr(trace, 'observations') else []
    
    # Create lookup by ID
    obs_map = {}
    for idx, obs in enumerate(observations):
        obs_dict = {
            "id": obs.id,
            "name": obs.name,
            "type": obs.type if hasattr(obs, 'type') else "span",
            "input": obs.input if hasattr(obs, 'input') else {},
            "output": obs.output if hasattr(obs, 'output') else {},
            "duration_ms": int((obs.end_time - obs.start_time).total_seconds() * 1000) 
                          if hasattr(obs, 'start_time') and hasattr(obs, 'end_time') 
                          and obs.start_time and obs.end_time else 0,
            "parent_id": obs.parent_observation_id if hasattr(obs, 'parent_observation_id') else None,
            "start_time": obs.start_time if hasattr(obs, 'start_time') else None,
            "idx": idx,
            "children": []
        }
        obs_map[obs.id] = obs_dict
    
    # Build tree
    root_nodes = []
    for obs_id, obs in obs_map.items():
        parent_id = obs.get("parent_id")
        if parent_id and parent_id in obs_map:
            obs_map[parent_id]["children"].append(obs)
        else:
            root_nodes.append(obs)
    
    # Sort children by start_time (with index as tiebreaker) so steps appear in chronological order
    for obs in obs_map.values():
        obs["children"].sort(key=lambda c: (c.get("start_time") or "", c.get("idx", 0)))

    # Return the root (usually the agent)
    if root_nodes:
        root_nodes.sort(key=lambda c: (c.get("start_time") or "", c.get("idx", 0)))
        return root_nodes[0]
    
    # Fallback: create a placeholder
    return {
        "name": "Rice_Farming_Advisor",
        "type": "agent",
        "duration_ms": 0,
        "input": {},
        "output": {},
        "children": []
    }

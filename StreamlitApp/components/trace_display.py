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


def display_trace_embedded(trace_id: str, trace_url: str, height: int = 500):
    """
    Display embedded Langfuse trace visualization.
    Falls back to iframe if API fetch fails.
    
    Args:
        trace_id: The trace ID
        trace_url: Full URL to the trace
        height: Height of the embedded component
    """
    st.markdown("### 📊 Langfuse Trace")
    
    # First show the trace link
    col1, col2 = st.columns([3, 1])
    with col1:
        st.code(trace_id, language=None)
    with col2:
        st.link_button("Open in Langfuse ↗", trace_url, use_container_width=True)
    
    # Try to fetch and display trace data
    try:
        from langfuse import Langfuse
        
        langfuse = Langfuse(timeout=config.LANGFUSE_TIMEOUT)
        
        # Fetch trace with observations (works on both Cloud and self-hosted)
        try:
            # Try new API first
            trace = langfuse.api.trace.get(trace_id)
            
            # Extract observations from the response
            observations = None
            if trace:
                # Check different possible response formats
                if hasattr(trace, 'observations'):
                    observations = trace.observations
                elif hasattr(trace, 'data') and hasattr(trace.data, 'observations'):
                    observations = trace.data.observations
                elif isinstance(trace, dict) and 'observations' in trace:
                    observations = trace['observations']
            
            if observations:
                # Build tree structure from observations
                trace_data = _build_trace_tree_from_observations(observations, trace_id)
                html = _generate_trace_html(trace_data)
                components.html(html, height=height, scrolling=True)
            else:
                st.info("Trace data not yet available. Click the link above to view in Langfuse.")
                
        except Exception as get_error:
            # Log the specific error for debugging
            st.info(f"Trace is being processed. View it in Langfuse for now.")
            
    except Exception as e:
        # Fallback: show link only
        st.info("The trace is being processed. View it directly in Langfuse:")
        

def _build_trace_tree_from_observations(observations: list, trace_id: str) -> Dict[str, Any]:
    """
    Build hierarchical tree structure from observations v2 API response.
    
    Args:
        observations: List of observations from observations_v_2.get_many()
        trace_id: The trace ID
        
    Returns:
        Hierarchical dict representation of the trace
    """
    if not observations:
        return {
            "name": "Rice_Farming_Advisor",
            "type": "agent",
            "duration_ms": 0,
            "input": {},
            "output": {},
            "children": []
        }
    
    # Create lookup by ID
    obs_map = {}
    for obs in observations:
        # Handle both dict and object responses
        if isinstance(obs, dict):
            obs_id = obs.get('id')
            obs_name = obs.get('name', 'Unknown')
            obs_type = obs.get('type', 'span')
            obs_input = obs.get('input', {})
            obs_output = obs.get('output', {})
            parent_id = obs.get('parent_observation_id')
            start_time = obs.get('start_time')
            end_time = obs.get('end_time')
        else:
            obs_id = getattr(obs, 'id', None)
            obs_name = getattr(obs, 'name', 'Unknown')
            obs_type = getattr(obs, 'type', 'span')
            obs_input = getattr(obs, 'input', {})
            obs_output = getattr(obs, 'output', {})
            parent_id = getattr(obs, 'parent_observation_id', None)
            start_time = getattr(obs, 'start_time', None)
            end_time = getattr(obs, 'end_time', None)
        
        # Calculate duration
        duration_ms = 0
        if start_time and end_time:
            try:
                if hasattr(start_time, 'total_seconds'):
                    duration_ms = int((end_time - start_time).total_seconds() * 1000)
                else:
                    # Handle datetime strings
                    from datetime import datetime
                    if isinstance(start_time, str):
                        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                        end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                        duration_ms = int((end_dt - start_dt).total_seconds() * 1000)
            except:
                pass
        
        obs_dict = {
            "id": obs_id,
            "name": obs_name,
            "type": obs_type.lower() if obs_type else "span",
            "input": obs_input if obs_input else {},
            "output": obs_output if obs_output else {},
            "duration_ms": duration_ms,
            "parent_id": parent_id,
            "children": []
        }
        obs_map[obs_id] = obs_dict
    
    # Build tree
    root_nodes = []
    for obs_id, obs in obs_map.items():
        parent_id = obs.get("parent_id")
        if parent_id and parent_id in obs_map:
            obs_map[parent_id]["children"].append(obs)
        else:
            root_nodes.append(obs)
    
    # Return the root (usually the agent)
    if root_nodes:
        return root_nodes[0]
    
    # Fallback
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
    for obs in observations:
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
    
    # Return the root (usually the agent)
    if root_nodes:
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

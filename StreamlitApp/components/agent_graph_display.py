"""
Component for displaying the agent graph structure as an interactive flow diagram.
"""

import streamlit as st
import streamlit.components.v1 as components
from typing import Dict, Any, List
import json


def display_agent_graph(
    graph_data: Dict[str, Any],
    height: int = 600,
    theme: str = "light"
):
    """
    Display the agent graph as an interactive flow diagram.
    
    Args:
        graph_data: Dict with 'nodes' and 'edges' from pipeline.get_agent_graph()
        height: Height of the component in pixels
        theme: "light" or "dark" for background color
    """
    st.markdown("### 🕸️ Agent Graph")
    
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    
    # Color scheme based on node type
    color_map = {
        "start": "#4caf50",
        "end": "#f44336",
        "agent": "#1565c0",
        "chain": "#7b1fa2", 
        "tool": "#ef6c00",
        "generation": "#2e7d32",
    }
    
    # Convert to vis.js format
    vis_nodes = []
    for node in nodes:
        node_type = node.get("type", "chain")
        color = color_map.get(node_type, "#607d8b")
        shape = "ellipse" if node_type in ["start", "end"] else "box"
        
        vis_nodes.append({
            "id": node["id"],
            "label": node["label"],
            "color": color,
            "font": {"color": "white", "size": 12, "face": "arial"},
            "shape": shape,
            "margin": 8
        })
    
    vis_edges = []
    for edge in edges:
        vis_edges.append({
            "from": edge[0],
            "to": edge[1],
            "arrows": "to",
            "smooth": {"type": "cubicBezier", "roundness": 0.3}
        })
    
    nodes_json = json.dumps(vis_nodes)
    edges_json = json.dumps(vis_edges)
    
    # Theme-aware colors
    if theme == "dark":
        bg_color = "#1a1d24"
        border_color = "#3d4654"
    else:
        bg_color = "#fafafa"
        border_color = "#e0e0e0"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
                overflow: hidden;
            }}
            #mynetwork {{
                width: 100%;
                height: 100%;
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 8px;
            }}
        </style>
    </head>
    <body>
        <div id="mynetwork"></div>
        <script type="text/javascript">
            var nodes = new vis.DataSet({nodes_json});
            var edges = new vis.DataSet({edges_json});
            
            var container = document.getElementById('mynetwork');
            var data = {{ nodes: nodes, edges: edges }};
            
            var options = {{
                layout: {{
                    hierarchical: {{
                        enabled: true,
                        direction: "UD",
                        sortMethod: "directed",
                        nodeSpacing: 180,
                        levelSeparation: 70,
                        treeSpacing: 200,
                        blockShifting: true,
                        edgeMinimization: true,
                        parentCentralization: true
                    }}
                }},
                physics: {{
                    enabled: false
                }},
                edges: {{
                    color: {{ color: '#848484', highlight: '#ff6b6b' }},
                    width: 2
                }},
                nodes: {{
                    borderWidth: 0,
                    shadow: {{
                        enabled: true,
                        color: 'rgba(0,0,0,0.15)',
                        size: 8,
                        x: 2,
                        y: 2
                    }}
                }},
                interaction: {{
                    hover: true,
                    zoomView: true,
                    dragView: true,
                    zoomSpeed: 0.5
                }}
            }};
            
            var network = new vis.Network(container, data, options);
            
            // Fit and zoom appropriately
            network.once('stabilizationIterationsDone', function() {{
                network.fit({{
                    animation: false
                }});
                // Zoom out a bit for better overview
                var scale = network.getScale();
                network.moveTo({{
                    scale: Math.min(scale, 0.9),
                    animation: false
                }});
            }});
        </script>
    </body>
    </html>
    """
    
    components.html(html, height=height, scrolling=False)

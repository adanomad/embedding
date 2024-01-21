# parses a DOT file and converts it into the JSON format compatible with Cytoscape.js
import re
import json


def parse_dot(dot_file):
    with open(dot_file, "r") as file:
        dot_content = file.read()

    # Regex patterns for nodes and edges
    node_pattern = r'(\w+)\s*\[label="([^"]+)"\];'
    edge_pattern = r"(\w+)\s*->\s*(\w+);"

    # Find all nodes and edges
    nodes = re.findall(node_pattern, dot_content)
    edges = re.findall(edge_pattern, dot_content)

    # Convert to JSON format
    elements = []
    for node_id, label in nodes:
        elements.append({"data": {"id": node_id, "label": label}})

    for source, target in edges:
        elements.append({"data": {"source": source, "target": target}})

    return elements


dot_file = "bjj.dot"
elements = parse_dot(dot_file)

# Print or save the JSON output
print(json.dumps({"elements": elements}, indent=4))

with open(dot_file + ".json", "w") as file:
    file.write(json.dumps({"elements": elements}, indent=4))

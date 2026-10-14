import re
import networkx as nx


graph = nx.Graph()


def extract_entities(text):
    pattern = r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b"

    entities = re.findall(pattern, text)

    ignored = {
        "The", "A", "An", "This", "That",
        "These", "Those", "However"
    }

    return sorted({
        entity.strip()
        for entity in entities
        if entity.strip() not in ignored
    })


def add_text_to_graph(text, source):
    entities = extract_entities(text)

    for entity in entities:
        graph.add_node(entity, entity_type="extracted_phrase")

    for index, first in enumerate(entities):
        for second in entities[index + 1:]:
            if first != second:
                graph.add_edge(
                    first,
                    second,
                    source=source,
                    relation="co_occurs"
                )

    return entities


def graph_summary():
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "entities": list(graph.nodes())
    }
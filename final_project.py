# final_project.py

import heapq

# Algorithm 1: Lowest Cost Delivery Between Two Locations

def dijkstra(graph, start, end):
    """
    Find the lowest cost delivery route between two locations.

    Args:
        graph (dict): A weighted graph where keys are nodes and values are lists of (neighbor, weight) tuples.
        start (str): Starting location.
        end (str): Ending location.

    Returns:
        tuple: (path as list of nodes, total cost as int)
    """
    # Ensure all nodes are keys in the graph
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        for neighbor, _ in neighbors:
            all_nodes.add(neighbor)
    # Build graph keys for missing nodes
    for node in all_nodes:
        graph.setdefault(node, [])

    # Priority queue to store (cost, node)
    queue = []
    heapq.heappush(queue, (0, start))

    # Dictionary to store the shortest known cost to each node
    costs = {node: float('inf') for node in graph}
    costs[start] = 0

    # Dictionary to store the previous node to reconstruct the path
    previous = {}

    while queue:
        current_cost, current_node = heapq.heappop(queue)

        # If we've reached the end node, stop
        if current_node == end:
            break

        for neighbor, weight in graph.get(current_node, []):
            new_cost = current_cost + weight
            if new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                previous[neighbor] = current_node
                heapq.heappush(queue, (new_cost, neighbor))

    # Reconstruct path
    path = []
    current = end
    while current in previous:
        path.append(current)
        current = previous[current]
    if current == start:
        path.append(start)
        path.reverse()
        return path, costs[end]
    else:
        # No path exists
        return None, None

# Algorithm 2: Best Path from the Hub

def prim(graph, start):
    """
    Find the minimum spanning tree (MST) starting from a hub location.

    Args:
        graph (dict): A weighted graph where keys are nodes and values are lists of (neighbor, weight) tuples.
        start (str): Starting (hub) location.

    Returns:
        tuple: (MST as list of (from, to, cost), total MST cost as int)
    """
    MST = []
    visited = set()
    total_cost = 0

    # Priority queue to store (cost, current_node, from_node)
    queue = [(0, start, None)]

    while queue and len(visited) < len(graph):
        cost, node, from_node = heapq.heappop(queue)

        if node not in visited:
            visited.add(node)
            if from_node is not None:
                MST.append((from_node, node, cost))
                total_cost += cost

            for neighbor, weight in graph.get(node, []):
                if neighbor not in visited:
                    heapq.heappush(queue, (weight, neighbor, node))

    # If the graph was disconnected, the MST will not include all nodes
    if len(visited) != len(graph):
        print("Warning: The graph is disconnected. MST may be incomplete.")

    return MST, total_cost

# Algorithm 3: Dynamic Network Changes

def dynamic_mst(graph, start, removed_edges, added_edges):
    """
    Adapt MST dynamically after changes in the graph.

    Args:
        graph (dict): A weighted graph where keys are nodes and values are lists of (neighbor, weight) tuples.
        start (str): Hub location.
        removed_edges (list): List of edges to remove, each in "Node1-Node2" format.
        added_edges (list): List of edges to add, each as (Node1, Node2, weight) tuple.

    Returns:
        tuple: (Updated MST, total cost)
    """
    # Remove edges
    for edge in removed_edges:
        node1, node2 = edge.split('-')
        # Remove connection from node1 to node2
        graph[node1] = [(n, w) for n, w in graph.get(node1, []) if n != node2]
        graph[node2] = [(n, w) for n, w in graph.get(node2, []) if n != node1]

    # Add edges
    for u, v, w in added_edges:
        graph.setdefault(u, []).append((v, w))
        graph.setdefault(v, []).append((u, w))

    # Recompute MST using updated graph
    return prim(graph, start)

# ------------------------------
# Test Cases (Examples)
# ------------------------------

if __name__ == "__main__":
    example_graph = {
        "A": [("B", 4), ("C", 2)],
        "B": [("A", 4), ("C", 1), ("D", 5)],
        "C": [("A", 2), ("B", 1), ("D", 8), ("E", 10)],
        "D": [("B", 5), ("C", 8), ("E", 2)],
        "E": [("C", 10), ("D", 2)]
    }

    # --- Algorithm 1 Tests ---
    print("Algorithm 1 Tests:")
    path, cost = dijkstra(example_graph, "A", "E")
    print(f"Path: {path}, Cost: {cost}")  # Should be ['A', 'C', 'B', 'D', 'E'], Cost 11

    path, cost = dijkstra(example_graph, "A", "B")
    print(f"Path: {path}, Cost: {cost}")  # Should be ['A', 'C', 'B'], Cost 3

    # Disconnected graph
    path, cost = dijkstra({"A": [("B", 3)], "B": [], "C": []}, "A", "C")
    print(f"Path: {path}, Cost: {cost}")  # Should be None, None (no path)

    # One-node graph
    single_node_graph = {"A": []}
    path, cost = dijkstra(single_node_graph, "A", "A")
    print(f"One-node graph Path: {path}, Cost: {cost}")  # Should be ['A'], Cost 0

    # Redundant edges (two paths between same nodes)
    redundant_graph = {
        "A": [("B", 1), ("B", 2)],
        "B": [("A", 1), ("A", 2)]
    }
    path, cost = dijkstra(redundant_graph, "A", "B")
    print(f"Redundant edges Path: {path}, Cost: {cost}")  # Should take the smaller weight, Cost 1

    # --- Algorithm 2 Tests ---
    print("\nAlgorithm 2 Tests:")
    mst, total_mst_cost = prim(example_graph, "A")
    print(f"MST: {mst}, Total Cost: {total_mst_cost}")  # Cost should be 10

    mst, total_mst_cost = prim({
        "A": [("B", 1)],
        "B": [("A", 1), ("C", 2)],
        "C": [("B", 2)]
    }, "A")
    print(f"Linear graph MST: {mst}, Total Cost: {total_mst_cost}")  # Cost 3

    # Disconnected graph
    mst, total_mst_cost = prim({
        "A": [("B", 1)],
        "B": [("A", 1)],
        "C": []
    }, "A")
    print(f"Disconnected graph MST: {mst}, Total Cost: {total_mst_cost}")  # Warning about disconnected

    # One-node graph
    mst, total_mst_cost = prim({"A": []}, "A")
    print(f"One-node MST: {mst}, Total Cost: {total_mst_cost}")  # Empty MST, Cost 0

    # Redundant edges graph
    mst, total_mst_cost = prim({
        "A": [("B", 1), ("B", 2)],
        "B": [("A", 1), ("A", 2)]
    }, "A")
    print(f"Redundant edges MST: {mst}, Total Cost: {total_mst_cost}")  # Should use weight 1

    # --- Algorithm 3 Tests ---
    print("\nAlgorithm 3 Tests:")
    modified_graph = {
        "A": [("B", 4), ("C", 2)],
        "B": [("A", 4), ("C", 1), ("D", 5)],
        "C": [("A", 2), ("B", 1), ("D", 8), ("E", 10)],
        "D": [("B", 5), ("C", 8), ("E", 2)],
        "E": [("C", 10), ("D", 2)]
    }
    mst, total_mst_cost = dynamic_mst(modified_graph, "A", ["C-E"], [("B", "E", 3)])
    print(f"Updated MST after dynamic changes: {mst}, Total Cost: {total_mst_cost}")  # Should change

    # One-node graph dynamic MST
    one_node_graph = {"A": []}
    mst, total_mst_cost = dynamic_mst(one_node_graph, "A", [], [])
    print(f"One-node dynamic MST: {mst}, Total Cost: {total_mst_cost}")  # Empty MST, Cost 0
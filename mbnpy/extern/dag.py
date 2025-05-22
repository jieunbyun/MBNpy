from collections import deque

class DAG:
    """Directed acyclic graph structure to manage pipeline task dependecies."""

    def __init__(self):
        self.graph = {}


    def add(self, node, points_to=None):
        """Add new task not to the graph, specify optional 'points_to' parameter."""
        if node not in self.graph:
            self.graph[node] = []

        if points_to:
            if points_to not in self.graph:
                self.graph[points_to] = []
            self.graph[node].append(points_to) # todo: need to make sure not to add duplicates

        # if sorted tasks and original graph lengths there must be a cycle 
        if len(self.sort()) != len(self.graph):
            raise Exception('A cycle is detected in the graph')


    def sort(self):
        """Sort all the task nodes based on the dependencies."""
        self.in_degrees()

        nodes_to_visit = deque()
        for node, pointers in self.degrees.items():
            # find all root nodes
            if pointers == 0:
                nodes_to_visit.append(node)

        sorted_nodes = []

        while nodes_to_visit:
            node = nodes_to_visit.popleft()
            for pointer in self.graph[node]:
                self.degrees[pointer] -= 1
                if self.degrees[pointer] == 0:
                    nodes_to_visit.append(pointer)
            sorted_nodes.append(node)

        return sorted_nodes


    def in_degrees(self):
        """Determing number of in-coming edges for each task node."""
        self.degrees = {}

        for node in self.graph:
            if node not in self.degrees:
                self.degrees[node] = 0
            for pointed in self.graph[node]:
                if pointed not in self.degrees:
                    self.degrees[pointed] = 0
                self.degrees[pointed] += 1



from collections import deque


class TemporalSemanticMemory:

    def __init__(self, max_entries=5):
        self.memory = deque(maxlen=max_entries)

    def add(self, report):
        self.memory.append(report)

    def get_recent(self):
        return list(self.memory)
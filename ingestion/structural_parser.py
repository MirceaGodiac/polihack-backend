import json
from .parser_rules import RULES
from .legal_ids import make_unit_id

class StructuralParser:
    def __init__(self, corpus_id: str):
        self.corpus_id = corpus_id
        self.units = []
        self.edges = []
        # tracks current state: {type: value}
        self.state = {
            'titlu': None,
            'capitol': None,
            'sectiune': None,
            'articol': None,
            'alineat': None,
            'litera': None
        }
        self.level_order = ['titlu', 'capitol', 'sectiune', 'articol', 'alineat', 'litera']

    def parse(self, lines: list):
        for line in lines:
            line = line.strip()
            if not line:
                continue

            matched = False
            for type_name, regex in RULES:
                match = regex.match(line)
                if match:
                    val = match.group(1)
                    self._update_state(type_name, val)
                    
                    unit = self._create_unit(type_name, val, line)
                    self.units.append(unit)
                    
                    self._create_edge(unit['id'])
                    matched = True
                    break
            
            if not matched:
                if self.units:
                    # If no structure match, append text to the last unit
                    self.units[-1]['raw_text'] += "\n" + line
                else:
                    # Safety fallback: ignore leading unstructured text
                    pass

        return self.units, self.edges

    def _update_state(self, type_name, val):
        # When a higher level changes, reset all lower levels
        idx = self.level_order.index(type_name)
        self.state[type_name] = val
        for i in range(idx + 1, len(self.level_order)):
            self.state[self.level_order[i]] = None

    def _get_current_path(self):
        path = []
        for level in self.level_order:
            if self.state[level]:
                path.append((level, self.state[level]))
        return path

    def _create_unit(self, type_name, val, text):
        path = self._get_current_path()
        unit_id = make_unit_id(self.corpus_id, path)
        
        return {
            "id": unit_id,
            "type": type_name,
            "raw_text": text,
            "hierarchy_path": [p[1] for p in path],
            "corpus_id": self.corpus_id
        }

    def _create_edge(self, unit_id):
        # Find the parent ID
        path = self._get_current_path()
        if len(path) > 1:
            parent_path = path[:-1]
            parent_id = make_unit_id(self.corpus_id, parent_path)
            self.edges.append({
                "source_id": parent_id,
                "target_id": unit_id,
                "type": "contains"
            })

    def save(self, units_path="legal_units.json", edges_path="legal_edges.json"):
        with open(units_path, "w", encoding="utf-8") as f:
            json.dump(self.units, f, indent=2, ensure_ascii=False)
        with open(edges_path, "w", encoding="utf-8") as f:
            json.dump(self.edges, f, indent=2, ensure_ascii=False)

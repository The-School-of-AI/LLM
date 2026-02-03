"""
Code Probe Generator - Programming language structure shells.

Generates synthetic code structures for tokenizer evaluation WITHOUT
using real benchmark code. Tests tokenizer handling of:
- Function signatures and class definitions
- JSON schemas
- Code comments and docstrings
- Multi-language patterns
"""

import random
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class CodeProbe:
    """A single code structure probe."""
    content: str
    language: str
    category: str  # function, class, json, comment
    difficulty: str


class CodeProbeGenerator:
    """
    Generates code structure probes (shells).
    
    Creates syntactically valid code structures with placeholder
    content to test tokenizer behavior on programming languages.
    """
    
    # Python templates
    PYTHON_FUNCTIONS = [
        '''def {func_name}({params}):
    """{docstring}"""
    {body}
    return {return_val}''',
    
        '''async def {func_name}({params}) -> {return_type}:
    """{docstring}"""
    {body}''',
    
        '''@{decorator}
def {func_name}({params}):
    {body}''',
    
        '''def {func_name}(
    {param1}: {type1},
    {param2}: {type2},
    *args,
    **kwargs
) -> {return_type}:
    {body}''',
    ]
    
    PYTHON_CLASSES = [
        '''class {class_name}:
    """{docstring}"""
    
    def __init__(self, {params}):
        self.{attr1} = {attr1}
        self.{attr2} = {attr2}
    
    def {method_name}(self):
        {body}''',
    
        '''@dataclass
class {class_name}:
    {attr1}: {type1}
    {attr2}: {type2}
    {attr3}: Optional[{type3}] = None''',
    
        '''class {class_name}({base_class}):
    
    def __init__(self, {params}):
        super().__init__()
        {body}''',
    ]
    
    # JavaScript templates
    JS_FUNCTIONS = [
        '''function {func_name}({params}) {{
    {body}
    return {return_val};
}}''',
    
        '''const {func_name} = ({params}) => {{
    {body}
}};''',
    
        '''async function {func_name}({params}) {{
    try {{
        {body}
    }} catch (error) {{
        console.error(error);
    }}
}}''',
    
        '''export const {func_name} = ({params}: {type}) => {{
    {body}
}};''',
    ]
    
    JS_CLASSES = [
        '''class {class_name} {{
    constructor({params}) {{
        this.{attr1} = {attr1};
        this.{attr2} = {attr2};
    }}
    
    {method_name}() {{
        {body}
    }}
}}''',
    
        '''class {class_name} extends {base_class} {{
    constructor({params}) {{
        super();
        {body}
    }}
}}''',
    ]
    
    # JSON schema templates
    JSON_SCHEMAS = [
        '''{{
    "type": "object",
    "properties": {{
        "{prop1}": {{"type": "{type1}"}},
        "{prop2}": {{"type": "{type2}"}},
        "{prop3}": {{"type": "array", "items": {{"type": "{type3}"}}}}
    }},
    "required": ["{prop1}", "{prop2}"]
}}''',
    
        '''{{
    "{key1}": {val1},
    "{key2}": "{str_val}",
    "{key3}": [{arr_items}],
    "{key4}": {{
        "nested_{key1}": {nested_val}
    }}
}}''',
    
        '''[
    {{"id": {id1}, "name": "{name1}", "active": {bool1}}},
    {{"id": {id2}, "name": "{name2}", "active": {bool2}}},
    {{"id": {id3}, "name": "{name3}", "active": {bool3}}}
]''',
    ]
    
    # Comment templates
    COMMENT_TEMPLATES = {
        "python": [
            '# {comment}',
            '# TODO: {todo}',
            '# FIXME: {fixme}',
            '"""\n{multiline_comment}\n"""',
        ],
        "javascript": [
            '// {comment}',
            '/* {block_comment} */',
            '/**\n * {jsdoc}\n * @param {param_doc}\n * @returns {return_doc}\n */',
        ],
        "java": [
            '// {comment}',
            '/* {block_comment} */',
            '/**\n * {javadoc}\n * @param {param_name} {param_doc}\n * @return {return_doc}\n */',
        ],
    }
    
    # Variable/function name components
    NAME_PREFIXES = ["get", "set", "is", "has", "create", "update", "delete", "process", "validate", "parse"]
    NAME_SUFFIXES = ["Data", "Item", "List", "Map", "Config", "Result", "Handler", "Manager", "Service", "Helper"]
    TYPES = ["str", "int", "float", "bool", "List", "Dict", "Any", "None", "Tuple", "Optional"]
    JS_TYPES = ["string", "number", "boolean", "object", "array", "null", "undefined"]
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
    
    def _generate_name(self, style: str = "camel") -> str:
        """Generate a synthetic function/variable name."""
        prefix = self.rng.choice(self.NAME_PREFIXES)
        suffix = self.rng.choice(self.NAME_SUFFIXES)
        
        if style == "camel":
            return prefix + suffix
        elif style == "snake":
            return f"{prefix.lower()}_{suffix.lower()}"
        else:
            return prefix.lower() + suffix
    
    def _generate_params(self, count: int = 2, style: str = "python") -> str:
        """Generate parameter list."""
        params = []
        for _ in range(count):
            name = self._generate_name("snake")
            if style == "python":
                if self.rng.random() < 0.5:
                    type_ = self.rng.choice(self.TYPES)
                    params.append(f"{name}: {type_}")
                else:
                    params.append(name)
            else:
                params.append(name)
        return ", ".join(params)
    
    def _generate_body(self, language: str = "python") -> str:
        """Generate placeholder function body."""
        if language == "python":
            return f"    result = {self._generate_name('snake')}"
        else:
            return f"    const result = {self._generate_name('camel')}();"
    
    def generate_python_probe(self, category: str = "function", difficulty: str = "medium") -> CodeProbe:
        """Generate a Python code probe."""
        if category == "function":
            template = self.rng.choice(self.PYTHON_FUNCTIONS)
        else:
            template = self.rng.choice(self.PYTHON_CLASSES)
        
        content = template.format(
            func_name=self._generate_name("snake"),
            method_name=self._generate_name("snake"),
            class_name=self._generate_name("camel"),
            base_class=self._generate_name("camel"),
            params=self._generate_params(self.rng.randint(1, 4)),
            param1=self._generate_name("snake"),
            param2=self._generate_name("snake"),
            type1=self.rng.choice(self.TYPES),
            type2=self.rng.choice(self.TYPES),
            type3=self.rng.choice(self.TYPES),
            return_type=self.rng.choice(self.TYPES),
            docstring=f"Placeholder docstring for {self._generate_name()}",
            body=self._generate_body("python"),
            return_val=self._generate_name("snake"),
            decorator=self.rng.choice(["staticmethod", "classmethod", "property", "lru_cache"]),
            attr1=self._generate_name("snake"),
            attr2=self._generate_name("snake"),
            attr3=self._generate_name("snake"),
        )
        
        return CodeProbe(content=content, language="python", category=category, difficulty=difficulty)
    
    def generate_javascript_probe(self, category: str = "function", difficulty: str = "medium") -> CodeProbe:
        """Generate a JavaScript code probe."""
        if category == "function":
            template = self.rng.choice(self.JS_FUNCTIONS)
        else:
            template = self.rng.choice(self.JS_CLASSES)
        
        content = template.format(
            func_name=self._generate_name("camel"),
            method_name=self._generate_name("camel"),
            class_name=self._generate_name("camel"),
            base_class=self._generate_name("camel"),
            params=self._generate_params(self.rng.randint(1, 3), "js"),
            type=self.rng.choice(self.JS_TYPES),
            body=self._generate_body("javascript"),
            return_val=self._generate_name("camel"),
            attr1=self._generate_name("camel"),
            attr2=self._generate_name("camel"),
        )
        
        return CodeProbe(content=content, language="javascript", category=category, difficulty=difficulty)
    
    def generate_json_probe(self, difficulty: str = "medium") -> CodeProbe:
        """Generate a JSON structure probe."""
        template = self.rng.choice(self.JSON_SCHEMAS)
        
        content = template.format(
            prop1=self._generate_name("snake"),
            prop2=self._generate_name("snake"),
            prop3=self._generate_name("snake"),
            type1=self.rng.choice(["string", "number", "boolean"]),
            type2=self.rng.choice(["string", "number", "boolean"]),
            type3=self.rng.choice(["string", "number", "object"]),
            key1=self._generate_name("snake"),
            key2=self._generate_name("snake"),
            key3=self._generate_name("snake"),
            key4=self._generate_name("snake"),
            val1=self.rng.randint(1, 1000),
            str_val=f"placeholder_{self.rng.randint(1, 100)}",
            arr_items=", ".join([str(self.rng.randint(1, 100)) for _ in range(3)]),
            nested_val=self.rng.randint(1, 100),
            id1=self.rng.randint(1, 1000),
            id2=self.rng.randint(1, 1000),
            id3=self.rng.randint(1, 1000),
            name1=f"item_{self.rng.randint(1, 100)}",
            name2=f"item_{self.rng.randint(1, 100)}",
            name3=f"item_{self.rng.randint(1, 100)}",
            bool1=str(self.rng.choice([True, False])).lower(),
            bool2=str(self.rng.choice([True, False])).lower(),
            bool3=str(self.rng.choice([True, False])).lower(),
        )
        
        return CodeProbe(content=content, language="json", category="json", difficulty=difficulty)
    
    def generate_batch(
        self,
        count: int = 100,
        languages: List[str] = None,
        difficulties: List[str] = None
    ) -> List[CodeProbe]:
        """Generate a batch of code probes."""
        languages = languages or ["python", "javascript", "json"]
        difficulties = difficulties or ["easy", "medium", "hard"]
        
        probes = []
        for _ in range(count):
            language = self.rng.choice(languages)
            difficulty = self.rng.choice(difficulties)
            category = self.rng.choice(["function", "class"]) if language != "json" else "json"
            
            if language == "python":
                probe = self.generate_python_probe(category, difficulty)
            elif language == "javascript":
                probe = self.generate_javascript_probe(category, difficulty)
            else:
                probe = self.generate_json_probe(difficulty)
            
            probes.append(probe)
        
        return probes
    
    def get_corpus(self, count: int = 500) -> str:
        """Get probes as a single corpus string."""
        probes = self.generate_batch(count)
        return "\n\n---\n\n".join(p.content for p in probes)

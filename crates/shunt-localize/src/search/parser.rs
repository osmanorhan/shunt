use super::{ChunkId, ChunkKind, CodeChunk, SearchError};
use std::path::Path;
use tree_sitter::{Node, Parser};

pub fn parse_file(
    next_id: &mut ChunkId,
    path: &Path,
    language: &str,
    content: &str,
) -> Result<Vec<CodeChunk>, SearchError> {
    match language {
        "rust" => RustExtractor.parse(next_id, path, content),
        "javascript" => JavaScriptExtractor.parse(next_id, path, content),
        "typescript" => TypeScriptExtractor.parse(next_id, path, content),
        "tsx" => TsxExtractor.parse(next_id, path, content),
        "python" => PythonExtractor.parse(next_id, path, content),
        other => Err(SearchError::ParserUnavailable(other.to_string())),
    }
}

trait LanguageExtractor {
    fn language(&self) -> &'static str;
    fn configure_parser(&self, parser: &mut Parser) -> Result<(), SearchError>;

    fn parse(
        &self,
        next_id: &mut ChunkId,
        path: &Path,
        content: &str,
    ) -> Result<Vec<CodeChunk>, SearchError> {
        let mut parser = Parser::new();
        self.configure_parser(&mut parser)?;

        let Some(tree) = parser.parse(content, None) else {
            return Ok(fallback_file_chunk(next_id, path, self.language(), content));
        };

        let root = tree.root_node();
        let mut chunks = Vec::new();
        self.collect_chunks(next_id, path, content, root, None, &mut chunks);

        if chunks.is_empty() {
            return Ok(fallback_file_chunk(next_id, path, self.language(), content));
        }

        Ok(chunks)
    }

    fn collect_chunks(
        &self,
        next_id: &mut ChunkId,
        path: &Path,
        content: &str,
        node: Node<'_>,
        parent_id: Option<ChunkId>,
        chunks: &mut Vec<CodeChunk>,
    );
}

struct RustExtractor;
struct JavaScriptExtractor;
struct TypeScriptExtractor;
struct TsxExtractor;
struct PythonExtractor;

impl LanguageExtractor for RustExtractor {
    fn language(&self) -> &'static str {
        "rust"
    }

    fn configure_parser(&self, parser: &mut Parser) -> Result<(), SearchError> {
        parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .map_err(|_| SearchError::ParserUnavailable(self.language().to_string()))
    }

    fn collect_chunks(
        &self,
        next_id: &mut ChunkId,
        path: &Path,
        content: &str,
        node: Node<'_>,
        parent_id: Option<ChunkId>,
        chunks: &mut Vec<CodeChunk>,
    ) {
        let file_imports = rust_imports(content, node);
        collect_rust_chunks(
            next_id,
            path,
            content,
            node,
            parent_id,
            &file_imports,
            chunks,
        );
    }
}

impl LanguageExtractor for JavaScriptExtractor {
    fn language(&self) -> &'static str {
        "javascript"
    }

    fn configure_parser(&self, parser: &mut Parser) -> Result<(), SearchError> {
        parser
            .set_language(&tree_sitter_javascript::LANGUAGE.into())
            .map_err(|_| SearchError::ParserUnavailable(self.language().to_string()))
    }

    fn collect_chunks(
        &self,
        next_id: &mut ChunkId,
        path: &Path,
        content: &str,
        node: Node<'_>,
        parent_id: Option<ChunkId>,
        chunks: &mut Vec<CodeChunk>,
    ) {
        let file_imports = script_imports(content, node);
        let context = ScriptChunkContext {
            path,
            content,
            language: self.language(),
            inherited_imports: &file_imports,
        };
        collect_script_chunks(next_id, &context, node, parent_id, chunks);
    }
}

impl LanguageExtractor for TypeScriptExtractor {
    fn language(&self) -> &'static str {
        "typescript"
    }

    fn configure_parser(&self, parser: &mut Parser) -> Result<(), SearchError> {
        parser
            .set_language(&tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
            .map_err(|_| SearchError::ParserUnavailable(self.language().to_string()))
    }

    fn collect_chunks(
        &self,
        next_id: &mut ChunkId,
        path: &Path,
        content: &str,
        node: Node<'_>,
        parent_id: Option<ChunkId>,
        chunks: &mut Vec<CodeChunk>,
    ) {
        let file_imports = script_imports(content, node);
        let context = ScriptChunkContext {
            path,
            content,
            language: self.language(),
            inherited_imports: &file_imports,
        };
        collect_script_chunks(next_id, &context, node, parent_id, chunks);
    }
}

impl LanguageExtractor for TsxExtractor {
    fn language(&self) -> &'static str {
        "tsx"
    }

    fn configure_parser(&self, parser: &mut Parser) -> Result<(), SearchError> {
        parser
            .set_language(&tree_sitter_typescript::LANGUAGE_TSX.into())
            .map_err(|_| SearchError::ParserUnavailable(self.language().to_string()))
    }

    fn collect_chunks(
        &self,
        next_id: &mut ChunkId,
        path: &Path,
        content: &str,
        node: Node<'_>,
        parent_id: Option<ChunkId>,
        chunks: &mut Vec<CodeChunk>,
    ) {
        let file_imports = script_imports(content, node);
        let context = ScriptChunkContext {
            path,
            content,
            language: self.language(),
            inherited_imports: &file_imports,
        };
        collect_script_chunks(next_id, &context, node, parent_id, chunks);
    }
}

impl LanguageExtractor for PythonExtractor {
    fn language(&self) -> &'static str {
        "python"
    }

    fn configure_parser(&self, parser: &mut Parser) -> Result<(), SearchError> {
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .map_err(|_| SearchError::ParserUnavailable(self.language().to_string()))
    }

    fn collect_chunks(
        &self,
        next_id: &mut ChunkId,
        path: &Path,
        content: &str,
        node: Node<'_>,
        parent_id: Option<ChunkId>,
        chunks: &mut Vec<CodeChunk>,
    ) {
        let file_imports = python_imports(content, node);
        let context = PythonChunkContext {
            path,
            content,
            inherited_imports: &file_imports,
        };
        collect_python_chunks(next_id, &context, node, parent_id, chunks);
    }
}

fn collect_rust_chunks(
    next_id: &mut ChunkId,
    path: &Path,
    content: &str,
    node: Node<'_>,
    parent_id: Option<ChunkId>,
    inherited_imports: &[String],
    chunks: &mut Vec<CodeChunk>,
) {
    let kind = rust_chunk_kind(node.kind());
    let current_parent = if let Some(chunk_type) = kind {
        let id = *next_id;
        *next_id += 1;
        let chunk_text = chunk_content(content, node, &chunk_type);
        let mut chunk = CodeChunk::new(id, path.to_string_lossy(), "rust", chunk_type, chunk_text);
        chunk.start_line = node.start_position().row as u32 + 1;
        chunk.end_line = node.end_position().row as u32 + 1;
        chunk.parent_id = parent_id;
        chunk.definitions = rust_definitions(content, node, &chunk.chunk_type);
        chunk.references = rust_references(content, node);
        chunk.calls = rust_calls(content, node);
        chunk.imports = rust_imports(content, node);
        chunk.imports.extend_from_slice(inherited_imports);
        chunk.imports.sort();
        chunk.imports.dedup();
        chunks.push(chunk);
        Some(id)
    } else {
        parent_id
    };

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_rust_chunks(
            next_id,
            path,
            content,
            child,
            current_parent,
            inherited_imports,
            chunks,
        );
    }
}

fn collect_script_chunks(
    next_id: &mut ChunkId,
    context: &ScriptChunkContext<'_>,
    node: Node<'_>,
    parent_id: Option<ChunkId>,
    chunks: &mut Vec<CodeChunk>,
) {
    let kind = script_chunk_kind(node);
    let current_parent = if let Some(chunk_type) = kind {
        let id = *next_id;
        *next_id += 1;
        let chunk_text = chunk_content(context.content, node, &chunk_type);
        let mut chunk = CodeChunk::new(
            id,
            context.path.to_string_lossy(),
            context.language,
            chunk_type,
            chunk_text,
        );
        chunk.start_line = node.start_position().row as u32 + 1;
        chunk.end_line = node.end_position().row as u32 + 1;
        chunk.parent_id = parent_id;
        chunk.definitions = script_definitions(context.content, node, &chunk.chunk_type);
        chunk.references = script_references(context.content, node);
        chunk.calls = script_calls(context.content, node);
        chunk.imports = script_imports(context.content, node);
        chunk.imports.extend_from_slice(context.inherited_imports);
        chunk.imports.sort();
        chunk.imports.dedup();
        chunks.push(chunk);
        Some(id)
    } else {
        parent_id
    };

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_script_chunks(next_id, context, child, current_parent, chunks);
    }
}

struct ScriptChunkContext<'a> {
    path: &'a Path,
    content: &'a str,
    language: &'a str,
    inherited_imports: &'a [String],
}

struct PythonChunkContext<'a> {
    path: &'a Path,
    content: &'a str,
    inherited_imports: &'a [String],
}

fn rust_chunk_kind(kind: &str) -> Option<ChunkKind> {
    match kind {
        "function_item" => Some(ChunkKind::Function),
        "impl_item" => Some(ChunkKind::Impl),
        "struct_item" => Some(ChunkKind::Struct),
        "trait_item" => Some(ChunkKind::Trait),
        "mod_item" => Some(ChunkKind::Module),
        _ => None,
    }
}

fn script_chunk_kind(node: Node<'_>) -> Option<ChunkKind> {
    match node.kind() {
        "function_declaration"
        | "method_definition"
        | "method_signature"
        | "abstract_method_signature" => Some(ChunkKind::Function),
        "class_declaration" | "abstract_class_declaration" | "enum_declaration" => {
            Some(ChunkKind::Struct)
        }
        "interface_declaration" => Some(ChunkKind::Trait),
        "internal_module" => Some(ChunkKind::Module),
        "variable_declarator" if is_function_value(node) => Some(ChunkKind::Function),
        _ => None,
    }
}

fn python_chunk_kind(node: Node<'_>) -> Option<ChunkKind> {
    match node.kind() {
        "function_definition" => Some(ChunkKind::Function),
        "class_definition" => Some(ChunkKind::Struct),
        "decorated_definition" => decorated_python_chunk_kind(node),
        _ => None,
    }
}

fn decorated_python_chunk_kind(node: Node<'_>) -> Option<ChunkKind> {
    let mut cursor = node.walk();
    node.children(&mut cursor)
        .find_map(|child| python_chunk_kind(child))
}

fn is_function_value(node: Node<'_>) -> bool {
    node.child_by_field_name("value")
        .is_some_and(|value| matches!(value.kind(), "arrow_function" | "function_expression"))
}

fn rust_definitions(content: &str, node: Node<'_>, chunk_type: &ChunkKind) -> Vec<String> {
    let mut definitions = Vec::new();

    match chunk_type {
        ChunkKind::Function | ChunkKind::Struct | ChunkKind::Trait | ChunkKind::Module => {
            if let Some(name) = named_child_text(content, node, "name") {
                definitions.push(name);
            }
        }
        ChunkKind::Impl => {
            if let Some(type_node) = first_child_kind(node, "type_identifier") {
                definitions.push(node_text(content, type_node));
            }
            if let Some(generic_node) = first_child_kind(node, "generic_type") {
                definitions.push(node_text(content, generic_node));
            }
        }
        _ => {}
    }

    normalize_symbols(definitions)
}

fn script_definitions(content: &str, node: Node<'_>, chunk_type: &ChunkKind) -> Vec<String> {
    let mut definitions = Vec::new();

    match chunk_type {
        ChunkKind::Function | ChunkKind::Struct | ChunkKind::Trait | ChunkKind::Module => {
            if let Some(name) = named_child_text(content, node, "name") {
                definitions.push(name);
            } else if (node.kind() == "method_definition" || node.kind() == "method_signature")
                && let Some(name) = node.child_by_field_name("name")
            {
                definitions.push(property_name(content, name));
            }
        }
        ChunkKind::Impl | ChunkKind::Comment | ChunkKind::Other(_) => {}
    }

    normalize_symbols(definitions)
}

fn python_definitions(content: &str, node: Node<'_>, chunk_type: &ChunkKind) -> Vec<String> {
    let mut definitions = Vec::new();

    match chunk_type {
        ChunkKind::Function | ChunkKind::Struct => {
            if let Some(name) = named_child_text(content, node, "name") {
                definitions.push(name);
            } else if node.kind() == "decorated_definition"
                && let Some(definition) = first_named_child(node)
                && let Some(name) = named_child_text(content, definition, "name")
            {
                definitions.push(name);
            }
        }
        ChunkKind::Impl
        | ChunkKind::Trait
        | ChunkKind::Module
        | ChunkKind::Comment
        | ChunkKind::Other(_) => {}
    }

    normalize_symbols(definitions)
}

fn rust_references(content: &str, node: Node<'_>) -> Vec<String> {
    let mut references = Vec::new();
    collect_rust_reference_nodes(content, node, &mut references);
    normalize_symbols(references)
}

fn script_references(content: &str, node: Node<'_>) -> Vec<String> {
    let mut references = Vec::new();
    collect_script_reference_nodes(content, node, &mut references);
    normalize_symbols(references)
}

fn python_references(content: &str, node: Node<'_>) -> Vec<String> {
    let mut references = Vec::new();
    collect_python_reference_nodes(content, node, &mut references);
    normalize_symbols(references)
}

fn rust_calls(content: &str, node: Node<'_>) -> Vec<String> {
    let mut calls = Vec::new();
    collect_rust_call_nodes(content, node, &mut calls);
    normalize_symbols(calls)
}

fn script_calls(content: &str, node: Node<'_>) -> Vec<String> {
    let mut calls = Vec::new();
    collect_script_call_nodes(content, node, &mut calls);
    normalize_symbols(calls)
}

fn python_calls(content: &str, node: Node<'_>) -> Vec<String> {
    let mut calls = Vec::new();
    collect_python_call_nodes(content, node, &mut calls);
    normalize_symbols(calls)
}

fn collect_rust_call_nodes(content: &str, node: Node<'_>, calls: &mut Vec<String>) {
    if node.kind() == "call_expression"
        && let Some(function) = node.child_by_field_name("function")
    {
        calls.push(rust_call_name(content, function));
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_rust_call_nodes(content, child, calls);
    }
}

fn collect_script_call_nodes(content: &str, node: Node<'_>, calls: &mut Vec<String>) {
    match node.kind() {
        "call_expression" | "new_expression" => {
            if let Some(function) = node.child_by_field_name("function") {
                calls.push(script_call_name(content, function));
            } else if let Some(constructor) = node.child_by_field_name("constructor") {
                calls.push(script_call_name(content, constructor));
            }
        }
        _ => {}
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_script_call_nodes(content, child, calls);
    }
}

fn collect_python_call_nodes(content: &str, node: Node<'_>, calls: &mut Vec<String>) {
    if node.kind() == "call"
        && let Some(function) = node.child_by_field_name("function")
    {
        calls.push(python_call_name(content, function));
        return;
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_python_call_nodes(content, child, calls);
    }
}

fn rust_call_name(content: &str, node: Node<'_>) -> String {
    match node.kind() {
        "identifier" | "field_identifier" | "scoped_identifier" => node_text(content, node),
        "field_expression" => node
            .child_by_field_name("field")
            .map(|field| node_text(content, field))
            .unwrap_or_else(|| node_text(content, node)),
        _ => node_text(content, node),
    }
}

fn script_call_name(content: &str, node: Node<'_>) -> String {
    match node.kind() {
        "identifier" | "property_identifier" => node_text(content, node),
        "member_expression" => node
            .child_by_field_name("property")
            .map(|property| property_name(content, property))
            .unwrap_or_else(|| node_text(content, node)),
        "subscript_expression" => node
            .child_by_field_name("index")
            .map(|index| node_text(content, index))
            .unwrap_or_else(|| node_text(content, node)),
        _ => node_text(content, node),
    }
}

fn python_call_name(content: &str, node: Node<'_>) -> String {
    match node.kind() {
        "identifier" => node_text(content, node),
        "attribute" => node
            .child_by_field_name("attribute")
            .map(|attribute| node_text(content, attribute))
            .unwrap_or_else(|| node_text(content, node)),
        _ => node_text(content, node),
    }
}

fn collect_rust_reference_nodes(content: &str, node: Node<'_>, references: &mut Vec<String>) {
    match node.kind() {
        "type_identifier" | "scoped_identifier" | "scoped_type_identifier" => {
            references.push(node_text(content, node));
        }
        "field_identifier" if is_reference_field(node) => {
            references.push(node_text(content, node));
        }
        "call_expression" => {
            return;
        }
        "field_expression" => {
            if let Some(field) = node.child_by_field_name("field") {
                references.push(node_text(content, field));
            }
            return;
        }
        _ => {}
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_rust_reference_nodes(content, child, references);
    }
}

fn collect_script_reference_nodes(content: &str, node: Node<'_>, references: &mut Vec<String>) {
    match node.kind() {
        "type_identifier" | "predefined_type" | "nested_type_identifier" => {
            references.push(node_text(content, node));
        }
        "member_expression" => {
            if let Some(property) = node.child_by_field_name("property") {
                references.push(property_name(content, property));
            }
        }
        "call_expression" | "new_expression" => {
            return;
        }
        _ => {}
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_script_reference_nodes(content, child, references);
    }
}

fn collect_python_reference_nodes(content: &str, node: Node<'_>, references: &mut Vec<String>) {
    match node.kind() {
        "attribute" => {
            if let Some(attribute) = node.child_by_field_name("attribute") {
                references.push(node_text(content, attribute));
            }
        }
        "type" | "identifier" if is_python_type_reference(node) => {
            references.push(node_text(content, node));
        }
        "call" => {
            return;
        }
        _ => {}
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_python_reference_nodes(content, child, references);
    }
}

fn is_python_type_reference(node: Node<'_>) -> bool {
    node.parent().is_some_and(|parent| {
        matches!(
            parent.kind(),
            "type" | "typed_parameter" | "parameters" | "return_type"
        )
    })
}

fn is_reference_field(node: Node<'_>) -> bool {
    node.parent()
        .is_some_and(|parent| matches!(parent.kind(), "field_expression" | "field_declaration"))
}

fn rust_imports(content: &str, node: Node<'_>) -> Vec<String> {
    let mut imports = Vec::new();
    collect_use_items(content, node, &mut imports);
    normalize_imports(imports)
}

fn script_imports(content: &str, node: Node<'_>) -> Vec<String> {
    let mut imports = Vec::new();
    collect_import_statements(content, node, &mut imports);
    normalize_imports(imports)
}

fn python_imports(content: &str, node: Node<'_>) -> Vec<String> {
    let mut imports = Vec::new();
    collect_python_imports(content, node, &mut imports);
    normalize_imports(imports)
}

fn collect_use_items(content: &str, node: Node<'_>, imports: &mut Vec<String>) {
    if node.kind() == "use_declaration" {
        imports.push(
            node_text(content, node)
                .trim()
                .trim_start_matches("use")
                .trim()
                .trim_end_matches(';')
                .trim()
                .to_string(),
        );
        return;
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_use_items(content, child, imports);
    }
}

fn collect_import_statements(content: &str, node: Node<'_>, imports: &mut Vec<String>) {
    if node.kind() == "import_statement" {
        imports.push(
            node_text(content, node)
                .trim()
                .trim_end_matches(';')
                .to_string(),
        );
        return;
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_import_statements(content, child, imports);
    }
}

fn collect_python_imports(content: &str, node: Node<'_>, imports: &mut Vec<String>) {
    if matches!(node.kind(), "import_statement" | "import_from_statement") {
        imports.push(node_text(content, node).trim().to_string());
        return;
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_python_imports(content, child, imports);
    }
}

fn property_name(content: &str, node: Node<'_>) -> String {
    match node.kind() {
        "property_identifier" | "private_property_identifier" | "identifier" => {
            node_text(content, node)
        }
        _ => node_text(content, node),
    }
}

fn named_child_text(content: &str, node: Node<'_>, field_name: &str) -> Option<String> {
    node.child_by_field_name(field_name)
        .map(|child| node_text(content, child))
}

fn first_child_kind<'tree>(node: Node<'tree>, kind: &str) -> Option<Node<'tree>> {
    let mut cursor = node.walk();
    node.children(&mut cursor)
        .find(|child| child.kind() == kind)
}

fn first_named_child<'tree>(node: Node<'tree>) -> Option<Node<'tree>> {
    let mut cursor = node.walk();
    node.children(&mut cursor).find(|child| child.is_named())
}

fn collect_python_chunks(
    next_id: &mut ChunkId,
    context: &PythonChunkContext<'_>,
    node: Node<'_>,
    parent_id: Option<ChunkId>,
    chunks: &mut Vec<CodeChunk>,
) {
    let kind = python_chunk_kind(node);
    let current_parent = if let Some(chunk_type) = kind {
        let id = *next_id;
        *next_id += 1;
        let chunk_text = chunk_content(context.content, node, &chunk_type);
        let mut chunk = CodeChunk::new(
            id,
            context.path.to_string_lossy(),
            "python",
            chunk_type,
            chunk_text,
        );
        chunk.start_line = node.start_position().row as u32 + 1;
        chunk.end_line = node.end_position().row as u32 + 1;
        chunk.parent_id = parent_id;
        chunk.definitions = python_definitions(context.content, node, &chunk.chunk_type);
        chunk.references = python_references(context.content, node);
        chunk.calls = python_calls(context.content, node);
        chunk.imports = python_imports(context.content, node);
        chunk.imports.extend_from_slice(context.inherited_imports);
        chunk.imports.sort();
        chunk.imports.dedup();
        chunks.push(chunk);
        Some(id)
    } else {
        parent_id
    };

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_python_chunks(next_id, context, child, current_parent, chunks);
    }
}

fn normalize_symbols(symbols: Vec<String>) -> Vec<String> {
    let mut normalized = symbols
        .into_iter()
        .flat_map(|symbol| {
            symbol
                .split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
                .filter(|piece| !piece.is_empty())
                .map(|piece| piece.to_ascii_lowercase())
                .collect::<Vec<_>>()
        })
        .filter(|symbol| symbol.len() > 1)
        .collect::<Vec<_>>();
    normalized.sort();
    normalized.dedup();
    normalized
}

fn normalize_imports(imports: Vec<String>) -> Vec<String> {
    let mut normalized = Vec::new();
    for import in imports {
        normalized.extend(normalize_symbols(vec![import.clone()]));
        let compact = import
            .split_whitespace()
            .collect::<String>()
            .to_ascii_lowercase();
        if !compact.is_empty() {
            normalized.push(compact.clone());
            normalized.extend(normalize_symbols(vec![compact]));
        }
    }
    normalized.sort();
    normalized.dedup();
    normalized
}

fn node_text(content: &str, node: Node<'_>) -> String {
    content
        .get(node.start_byte()..node.end_byte())
        .unwrap_or_default()
        .to_string()
}

fn chunk_content(content: &str, node: Node<'_>, chunk_type: &ChunkKind) -> String {
    let text = node_text(content, node);
    match chunk_type {
        ChunkKind::Impl | ChunkKind::Trait | ChunkKind::Module | ChunkKind::Struct => {
            header_only(&text)
        }
        _ => text,
    }
}

fn header_only(text: &str) -> String {
    let Some(index) = text.find('{') else {
        return text.to_string();
    };

    format!("{} {{ ... }}", text[..index].trim())
}

fn fallback_file_chunk(
    next_id: &mut ChunkId,
    path: &Path,
    language: &str,
    content: &str,
) -> Vec<CodeChunk> {
    let id = *next_id;
    *next_id += 1;
    let mut chunk = CodeChunk::new(
        id,
        path.to_string_lossy(),
        language,
        ChunkKind::Module,
        content.to_string(),
    );
    chunk.start_line = 1;
    chunk.end_line = content.lines().count() as u32;
    chunk.references = normalize_symbols(vec![content.to_string()]);
    chunk.calls = Vec::new();
    vec![chunk]
}

#[cfg(test)]
mod tests {
    use super::ChunkKind;
    use super::parse_file;
    use std::path::Path;

    #[test]
    fn extracts_rust_items_with_parent_links() {
        let mut next_id = 1;
        let chunks = parse_file(
            &mut next_id,
            Path::new("src/lib.rs"),
            "rust",
            r#"
            pub struct Store;

            impl Store {
                pub fn load_user(&self) {}
            }
            "#,
        )
        .unwrap();

        assert!(
            chunks
                .iter()
                .any(|chunk| chunk.chunk_type == ChunkKind::Struct)
        );
        let impl_chunk = chunks
            .iter()
            .find(|chunk| chunk.chunk_type == ChunkKind::Impl)
            .unwrap();
        let function_chunk = chunks
            .iter()
            .find(|chunk| chunk.chunk_type == ChunkKind::Function)
            .unwrap();
        assert_eq!(function_chunk.parent_id, Some(impl_chunk.id));
        assert!(!impl_chunk.content.contains("load_user"));
        assert!(
            function_chunk
                .definitions
                .contains(&"load_user".to_string())
        );
        assert!(impl_chunk.definitions.contains(&"store".to_string()));
    }

    #[test]
    fn extracts_rust_references_and_imports() {
        let mut next_id = 1;
        let chunks = parse_file(
            &mut next_id,
            Path::new("src/lib.rs"),
            "rust",
            r#"
            use super::db::Pool;

            pub fn save_user(pool: Pool, user: User) {
                pool.execute(user.id);
            }
            "#,
        )
        .unwrap();
        let function_chunk = chunks
            .iter()
            .find(|chunk| chunk.chunk_type == ChunkKind::Function)
            .unwrap();

        assert!(
            function_chunk
                .definitions
                .contains(&"save_user".to_string())
        );
        assert!(function_chunk.references.contains(&"pool".to_string()));
        assert!(function_chunk.references.contains(&"user".to_string()));
        assert!(function_chunk.calls.contains(&"execute".to_string()));
    }

    #[test]
    fn does_not_index_plain_local_identifiers_as_references() {
        let mut next_id = 1;
        let chunks = parse_file(
            &mut next_id,
            Path::new("src/lib.rs"),
            "rust",
            r#"
            pub fn main() {
                let args = std::env::args();
                run(args);
            }
            "#,
        )
        .unwrap();
        let function_chunk = chunks
            .iter()
            .find(|chunk| chunk.chunk_type == ChunkKind::Function)
            .unwrap();

        assert!(!function_chunk.references.contains(&"args".to_string()));
        assert!(!function_chunk.references.contains(&"main".to_string()));
        assert!(function_chunk.calls.contains(&"run".to_string()));
    }

    #[test]
    fn extracts_typescript_chunks_and_symbols() {
        let mut next_id = 1;
        let chunks = parse_file(
            &mut next_id,
            Path::new("src/store.ts"),
            "typescript",
            r#"
            import { ApiClient } from "./client";

            export interface Store {
                loadUser(id: UserId): Promise<User>;
            }

            export class SessionStore {
                async loadUser(id: UserId) {
                    return api.fetchUser(id);
                }
            }
            "#,
        )
        .unwrap();

        let class_chunk = chunks
            .iter()
            .find(|chunk| chunk.chunk_type == ChunkKind::Struct)
            .unwrap();
        let method_chunk = chunks
            .iter()
            .find(|chunk| {
                chunk.chunk_type == ChunkKind::Function
                    && chunk.definitions.contains(&"loaduser".to_string())
                    && chunk.calls.contains(&"fetchuser".to_string())
            })
            .unwrap();

        assert!(
            class_chunk
                .definitions
                .contains(&"sessionstore".to_string())
        );
        assert_eq!(method_chunk.parent_id, Some(class_chunk.id));
        assert!(method_chunk.calls.contains(&"fetchuser".to_string()));
        assert!(method_chunk.references.contains(&"userid".to_string()));
        assert!(method_chunk.imports.contains(&"apiclient".to_string()));
    }

    #[test]
    fn extracts_javascript_function_variables() {
        let mut next_id = 1;
        let chunks = parse_file(
            &mut next_id,
            Path::new("src/server.js"),
            "javascript",
            r#"
            import http from "node:http";

            const handleRequest = (req, res) => {
                res.end(renderPage());
            };
            "#,
        )
        .unwrap();

        let function_chunk = chunks
            .iter()
            .find(|chunk| {
                chunk.chunk_type == ChunkKind::Function
                    && chunk.definitions.contains(&"handlerequest".to_string())
            })
            .unwrap();

        assert!(function_chunk.calls.contains(&"end".to_string()));
        assert!(function_chunk.calls.contains(&"renderpage".to_string()));
        assert!(function_chunk.imports.contains(&"http".to_string()));
    }

    #[test]
    fn extracts_python_chunks_and_symbols() {
        let mut next_id = 1;
        let chunks = parse_file(
            &mut next_id,
            Path::new("src/service.py"),
            "python",
            r#"
from client import ApiClient

class SessionStore:
    def load_user(self, user_id: UserId) -> User:
        return api.fetch_user(user_id)
"#,
        )
        .unwrap();

        let class_chunk = chunks
            .iter()
            .find(|chunk| chunk.chunk_type == ChunkKind::Struct)
            .unwrap();
        let method_chunk = chunks
            .iter()
            .find(|chunk| {
                chunk.chunk_type == ChunkKind::Function
                    && chunk.definitions.contains(&"load_user".to_string())
            })
            .unwrap();

        assert!(
            class_chunk
                .definitions
                .contains(&"sessionstore".to_string())
        );
        assert_eq!(method_chunk.parent_id, Some(class_chunk.id));
        assert!(method_chunk.calls.contains(&"fetch_user".to_string()));
        assert!(method_chunk.references.contains(&"userid".to_string()));
        assert!(method_chunk.imports.contains(&"apiclient".to_string()));
    }
}

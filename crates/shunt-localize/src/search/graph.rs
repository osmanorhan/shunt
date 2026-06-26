use super::chunk::{ChunkId, CodeChunk};
use super::symbol::{SymbolIndex, normalize_exact_symbol};
use super::types::{ChunkNeighbor, GraphEdge, GraphStepDirection, NeighborKind};
use std::collections::HashMap;

/// Immutable typed relationships between active code chunks.
pub struct RelationshipIndex {
    parents: HashMap<ChunkId, ChunkId>,
    children: HashMap<ChunkId, Vec<ChunkId>>,
    file_chunks: HashMap<String, Vec<ChunkId>>,
    chunk_file_positions: HashMap<ChunkId, (String, usize)>,
    related_neighbors: HashMap<ChunkId, Vec<ChunkNeighbor>>,
    outgoing_edges: HashMap<ChunkId, Vec<GraphEdge>>,
    incoming_edges: HashMap<ChunkId, Vec<GraphEdge>>,
}

impl RelationshipIndex {
    /// Build containment, file-membership, and exact-symbol relationships.
    pub fn build(chunks: &[CodeChunk], symbols: &SymbolIndex) -> Self {
        let mut parents = HashMap::new();
        let mut children: HashMap<ChunkId, Vec<ChunkId>> = HashMap::new();
        let mut file_chunks: HashMap<String, Vec<ChunkId>> = HashMap::new();
        let mut chunk_file_positions = HashMap::new();
        let mut start_lines = HashMap::new();

        for chunk in chunks.iter().filter(|chunk| chunk.active) {
            start_lines.insert(chunk.id, chunk.start_line);
            file_chunks
                .entry(chunk.file_path.clone())
                .or_default()
                .push(chunk.id);
            if let Some(parent_id) = chunk.parent_id {
                parents.insert(chunk.id, parent_id);
                children.entry(parent_id).or_default().push(chunk.id);
            }
        }

        for chunk_ids in children.values_mut() {
            chunk_ids.sort_by_key(|chunk_id| {
                (
                    start_lines.get(chunk_id).copied().unwrap_or(u32::MAX),
                    *chunk_id,
                )
            });
        }

        for chunk_ids in file_chunks.values_mut() {
            chunk_ids.sort_by_key(|chunk_id| {
                (
                    start_lines.get(chunk_id).copied().unwrap_or(u32::MAX),
                    *chunk_id,
                )
            });
        }

        for (file_path, chunk_ids) in &file_chunks {
            for (index, chunk_id) in chunk_ids.iter().enumerate() {
                chunk_file_positions.insert(*chunk_id, (file_path.clone(), index));
            }
        }

        let mut outgoing_edges: HashMap<ChunkId, Vec<GraphEdge>> = HashMap::new();
        let mut incoming_edges: HashMap<ChunkId, Vec<GraphEdge>> = HashMap::new();

        for chunk in chunks.iter().filter(|chunk| chunk.active) {
            if let Some(parent_id) = chunk.parent_id {
                push_edge(
                    &mut outgoing_edges,
                    &mut incoming_edges,
                    GraphEdge {
                        from_chunk_id: chunk.id,
                        to_chunk_id: parent_id,
                        kind: NeighborKind::Parent,
                        symbol: None,
                    },
                );
                push_edge(
                    &mut outgoing_edges,
                    &mut incoming_edges,
                    GraphEdge {
                        from_chunk_id: parent_id,
                        to_chunk_id: chunk.id,
                        kind: NeighborKind::Child,
                        symbol: None,
                    },
                );
            }

            for edge in build_symbol_edges(chunk, symbols) {
                push_edge(&mut outgoing_edges, &mut incoming_edges, edge);
            }
        }

        for edges in outgoing_edges.values_mut() {
            sort_and_dedup_edges(edges);
        }
        for edges in incoming_edges.values_mut() {
            sort_and_dedup_edges(edges);
        }

        let related_neighbors = outgoing_edges
            .iter()
            .map(|(chunk_id, edges)| {
                let neighbors = edges
                    .iter()
                    .filter(|edge| {
                        matches!(
                            edge.kind,
                            NeighborKind::Definition
                                | NeighborKind::Reference
                                | NeighborKind::Call
                                | NeighborKind::Import
                        )
                    })
                    .map(|edge| ChunkNeighbor {
                        chunk_id: edge.to_chunk_id,
                        kind: edge.kind,
                        symbol: edge.symbol.clone(),
                    })
                    .collect::<Vec<_>>();
                (*chunk_id, neighbors)
            })
            .collect();

        Self {
            parents,
            children,
            file_chunks,
            chunk_file_positions,
            related_neighbors,
            outgoing_edges,
            incoming_edges,
        }
    }

    /// Return the direct parent chunk, if present.
    pub fn parent_of(&self, chunk_id: ChunkId) -> Option<ChunkId> {
        self.parents.get(&chunk_id).copied()
    }

    /// Return direct children in source order.
    pub fn children_of(&self, chunk_id: ChunkId) -> &[ChunkId] {
        self.children
            .get(&chunk_id)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Return chunks belonging to an exact or suffix-matching file path.
    pub fn file_chunk_ids(&self, file_path: &str) -> Vec<ChunkId> {
        if let Some(chunk_ids) = self.file_chunks.get(file_path) {
            return chunk_ids.clone();
        }

        self.file_chunks
            .iter()
            .filter(|(path, _)| path.ends_with(file_path))
            .flat_map(|(_, chunk_ids)| chunk_ids.iter().copied())
            .collect()
    }

    /// Return typed exact-symbol relationships for a chunk.
    pub fn related_neighbors(&self, chunk_id: ChunkId) -> &[ChunkNeighbor] {
        self.related_neighbors
            .get(&chunk_id)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Return all graph edges that originate from the provided chunk.
    pub fn outgoing_edges(&self, chunk_id: ChunkId) -> &[GraphEdge] {
        self.outgoing_edges
            .get(&chunk_id)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Return all graph edges that terminate at the provided chunk.
    pub fn incoming_edges(&self, chunk_id: ChunkId) -> &[GraphEdge] {
        self.incoming_edges
            .get(&chunk_id)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Return nearby chunks from the same file, alternating before and after.
    pub fn nearby_file_chunk_ids(&self, chunk_id: ChunkId, limit: usize) -> Vec<ChunkId> {
        if limit == 0 {
            return Vec::new();
        }

        let Some((file_path, position)) = self.chunk_file_positions.get(&chunk_id) else {
            return Vec::new();
        };
        let Some(chunk_ids) = self.file_chunks.get(file_path) else {
            return Vec::new();
        };

        let mut nearby = Vec::new();
        let mut offset = 1usize;
        while nearby.len() < limit && (*position >= offset || position + offset < chunk_ids.len()) {
            if *position >= offset {
                let neighbor_id = chunk_ids[position - offset];
                if neighbor_id != chunk_id {
                    nearby.push(neighbor_id);
                    if nearby.len() == limit {
                        break;
                    }
                }
            }

            if position + offset < chunk_ids.len() {
                let neighbor_id = chunk_ids[position + offset];
                if neighbor_id != chunk_id {
                    nearby.push(neighbor_id);
                    if nearby.len() == limit {
                        break;
                    }
                }
            }

            offset += 1;
        }

        nearby
    }
}

pub(crate) fn graph_hop_to_neighbor(
    edge: &GraphEdge,
    direction: GraphStepDirection,
) -> ChunkNeighbor {
    let chunk_id = match direction {
        GraphStepDirection::Outgoing => edge.to_chunk_id,
        GraphStepDirection::Incoming => edge.from_chunk_id,
    };
    ChunkNeighbor {
        chunk_id,
        kind: edge.kind,
        symbol: edge.symbol.clone(),
    }
}

fn build_symbol_edges(chunk: &CodeChunk, symbols: &SymbolIndex) -> Vec<GraphEdge> {
    let mut edges = Vec::new();

    for symbol in exact_symbols(&chunk.definitions) {
        if let Some(chunk_ids) = symbols.reference_chunk_ids_for_exact_symbol(&symbol) {
            for chunk_id in chunk_ids {
                edges.push(GraphEdge {
                    from_chunk_id: chunk.id,
                    to_chunk_id: *chunk_id,
                    kind: NeighborKind::Reference,
                    symbol: Some(symbol.clone()),
                });
            }
        }
    }

    for symbol in exact_symbols(&chunk.calls) {
        if !is_useful_neighbor_symbol(&symbol) {
            continue;
        }

        if let Some(chunk_ids) = symbols.definition_chunk_ids_for_exact_symbol(&symbol) {
            for chunk_id in chunk_ids {
                edges.push(GraphEdge {
                    from_chunk_id: chunk.id,
                    to_chunk_id: *chunk_id,
                    kind: NeighborKind::Definition,
                    symbol: Some(symbol.clone()),
                });
            }
        }

        if let Some(chunk_ids) = symbols.call_chunk_ids_for_exact_symbol(&symbol) {
            for chunk_id in chunk_ids {
                edges.push(GraphEdge {
                    from_chunk_id: chunk.id,
                    to_chunk_id: *chunk_id,
                    kind: NeighborKind::Call,
                    symbol: Some(symbol.clone()),
                });
            }
        }
    }

    for symbol in exact_symbols(&chunk.references) {
        if !is_useful_neighbor_symbol(&symbol) {
            continue;
        }

        if let Some(chunk_ids) = symbols.definition_chunk_ids_for_exact_symbol(&symbol) {
            for chunk_id in chunk_ids {
                edges.push(GraphEdge {
                    from_chunk_id: chunk.id,
                    to_chunk_id: *chunk_id,
                    kind: NeighborKind::Definition,
                    symbol: Some(symbol.clone()),
                });
            }
        }
    }

    for symbol in exact_symbols(&chunk.imports) {
        if let Some(chunk_ids) = symbols.import_chunk_ids_for_exact_symbol(&symbol) {
            for chunk_id in chunk_ids {
                edges.push(GraphEdge {
                    from_chunk_id: chunk.id,
                    to_chunk_id: *chunk_id,
                    kind: NeighborKind::Import,
                    symbol: Some(symbol.clone()),
                });
            }
        }
    }

    edges
}

fn exact_symbols<'a>(symbols: impl IntoIterator<Item = &'a String>) -> Vec<String> {
    let mut exact = symbols
        .into_iter()
        .map(|symbol| normalize_exact_symbol(symbol))
        .filter(|symbol| !symbol.is_empty())
        .collect::<Vec<_>>();
    exact.sort();
    exact.dedup();
    exact
}

fn is_useful_neighbor_symbol(symbol: &str) -> bool {
    let symbol = symbol.trim();
    symbol.len() > 3
        && !matches!(
            symbol,
            "args"
                | "arg"
                | "chunk"
                | "chunks"
                | "config"
                | "default"
                | "error"
                | "from"
                | "index"
                | "input"
                | "item"
                | "items"
                | "new"
                | "path"
                | "query"
                | "result"
                | "self"
                | "value"
        )
}

fn push_edge(
    outgoing_edges: &mut HashMap<ChunkId, Vec<GraphEdge>>,
    incoming_edges: &mut HashMap<ChunkId, Vec<GraphEdge>>,
    edge: GraphEdge,
) {
    outgoing_edges
        .entry(edge.from_chunk_id)
        .or_default()
        .push(edge.clone());
    incoming_edges
        .entry(edge.to_chunk_id)
        .or_default()
        .push(edge);
}

fn sort_and_dedup_edges(edges: &mut Vec<GraphEdge>) {
    edges.sort_by(|left, right| {
        left.kind
            .as_str()
            .cmp(right.kind.as_str())
            .then_with(|| left.from_chunk_id.cmp(&right.from_chunk_id))
            .then_with(|| left.to_chunk_id.cmp(&right.to_chunk_id))
            .then_with(|| left.symbol.cmp(&right.symbol))
    });
    edges.dedup();
}

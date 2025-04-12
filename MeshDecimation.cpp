#include "MeshDecimation.h"

#include <vector>
#include <intrin.h>
#include <assert.h>
#include <math.h>

//
// References:
// - Michael Garland, Paul S. Heckbert. 1997. Surface Simplification Using Quadric Error Metrics.
// - Hugues Hoppe. 1999. New Quadric Metric for Simplifying Meshes with Appearance Attributes.
// - HSUEH-TI DEREK LIU, XIAOTING ZHANG, CEM YUKSEL. 2024. Simplifying Triangle Meshes in the Wild.
// - Hugues Hoppe, Steve Marschner. 2000. Efficient Minimization of New Quadric Metric for Simplifying Meshes with Appearance Attributes.
//

// fenv pragmas are an attempt at making compiler not optimize away (x + 0.f).
#pragma fenv_access(on)
static u64 ComputePositionHash(const Vector3& v) {
	compile_const u64 knuth_golden_ratio = 0x9e3779b97f4a7c55;
	
	static union {
		u64 scalars[2];
		__m128i vector;
	} seed = { knuth_golden_ratio, knuth_golden_ratio };
	
	// Add zero to turn -0.0 to +0.0.
	auto hash = _mm_castps_si128(_mm_add_ps(_mm_setr_ps(v.z, v.z, v.y, v.x), _mm_setzero_ps()));
	hash = _mm_aesdec_si128(hash, seed.vector);
	hash = _mm_aesdec_si128(hash, seed.vector);
	
	return _mm_cvtsi128_si64(hash);
}
#pragma fenv_access(off)

static u64 PackEdgeKey(VertexID vertex_id_0, VertexID vertex_id_1) {
	// Always pack VertexIDs in ascending order to ensure that PackEdgeKey(A, B) == PackEdgeKey(B, A) and they hash to the same value.
	return vertex_id_1.index > vertex_id_0.index ? ((u64)vertex_id_1.index << u32_bit_count) | (u64)vertex_id_0.index : ((u64)vertex_id_0.index << u32_bit_count) | (u64)vertex_id_1.index;
}

template<ElementType element_type_t> auto GetElementID(const Corner& corner) {
	if constexpr (element_type_t == ElementType::Vertex) {
		return corner.vertex_id;
	} else if constexpr (element_type_t == ElementType::Edge) {
		return corner.edge_id;
	} else if constexpr (element_type_t == ElementType::Face) {
		return corner.face_id;
	} else {
		static_assert(false, "Unknown element_type_t.");
	}
}

template<typename ElementID>
static void CornerListInsert(MeshView mesh, ElementID element_id, CornerID new_corner_id) {
	auto& corner  = mesh[new_corner_id];
	auto& element = mesh[element_id];
	
	compile_const u32 element_type = (u32)ElementID::element_type;
	if (element.corner_list_base.index == u32_max) {
		element.corner_list_base = new_corner_id;
		corner.corner_list_around[element_type].prev = new_corner_id;
		corner.corner_list_around[element_type].next = new_corner_id;
	} else {
		auto& existing_corner = mesh[element.corner_list_base];
		mesh[existing_corner.corner_list_around[element_type].prev].corner_list_around[element_type].next = new_corner_id;
		
		corner.corner_list_around[element_type].prev = existing_corner.corner_list_around[element_type].prev;
		corner.corner_list_around[element_type].next = element.corner_list_base;
		existing_corner.corner_list_around[element_type].prev = new_corner_id;
	}
}

template<ElementType element_type_t>
static bool CornerListRemove(MeshView mesh, CornerID corner_id) {
	compile_const u32 element_type = (u32)element_type_t;
	
	auto& corner = mesh[corner_id];
	auto prev_corner_id = corner.corner_list_around[element_type].prev;
	auto next_corner_id = corner.corner_list_around[element_type].next;
	
	mesh[next_corner_id].corner_list_around[element_type].prev = prev_corner_id;
	mesh[prev_corner_id].corner_list_around[element_type].next = next_corner_id;
	
	// Remove the element if it lost it's last corner.
	bool is_last_reference = (prev_corner_id.index == corner_id.index);
	auto new_corner_list_base = is_last_reference ? CornerID{ u32_max } : prev_corner_id;
	
	mesh[GetElementID<element_type_t>(corner)].corner_list_base = new_corner_list_base;
	
	corner.corner_list_around[element_type].prev.index = u32_max;
	corner.corner_list_around[element_type].next.index = u32_max;
	
	return is_last_reference;
}

enum struct IterationControl : u32 {
	Continue = 0,
	Break    = 1,
};

// Iterate linked list around a given element type starting with the base corner id. 
// Removal while iterating is allowed.
template<ElementType element_type_t, typename Lambda>
static void IterateCornerList(MeshView mesh, CornerID corner_list_base, Lambda&& lambda) {
	auto& element = mesh[GetElementID<element_type_t>(mesh[corner_list_base])];
	
	auto current_corner_id = corner_list_base;
	IterationControl control = IterationControl::Continue;
	do {
		auto next_corner_id = mesh[current_corner_id].corner_list_around[(u32)element_type_t].next;
		
		if constexpr (std::is_invocable_r<IterationControl, Lambda&, CornerID>::value) {
			control = lambda(current_corner_id);
		} else if constexpr (std::is_invocable_r<void, Lambda&, CornerID>::value) {
			lambda(current_corner_id);
		} else {
			static_assert(false, "Lambda passed to IterateCornerList should have signature (CornerID) -> void or (CornerID) -> IterationControl.");
		}
		
		current_corner_id = next_corner_id;
	} while (current_corner_id.index != corner_list_base.index && element.corner_list_base.index != u32_max && control == IterationControl::Continue);
}

// Merge linked lists around element_0 and element_1 and remove element_1.
// Patch up references to element_1 with a reference to element_0.
template<typename ElementID>
static void CornerListMerge(MeshView mesh, ElementID element_0, ElementID element_1) {
	auto base_id_0 = mesh[element_0].corner_list_base;
	auto base_id_1 = mesh[element_1].corner_list_base;
	
	compile_const ElementType element_type_t = ElementID::element_type;
	compile_const u32 element_type = (u32)element_type_t;
	
	if (base_id_0.index != u32_max && base_id_1.index != u32_max) {
		IterateCornerList<element_type_t>(mesh, base_id_1, [&](CornerID corner_id) {
			if constexpr (element_type_t == ElementType::Vertex) {
				mesh[corner_id].vertex_id = element_0;
			
				// TODO: This can be iteration over just incoming and outgoing edges of a corner.
				IterateCornerList<ElementType::Face>(mesh, corner_id, [&](CornerID corner_id) {
					auto& edge = mesh[mesh[corner_id].edge_id];
					if (edge.vertex_0.index == element_1.index) edge.vertex_0 = element_0;
					if (edge.vertex_1.index == element_1.index) edge.vertex_1 = element_0;
					assert(edge.vertex_0.index != edge.vertex_1.index);
				});
			} else if constexpr (element_type_t == ElementType::Edge) {
				mesh[corner_id].edge_id = element_0;
			} else {
				static_assert(false, "Unknown element_type_t.");
			}
		});
		
		auto base_id_0_prev = mesh[base_id_0].corner_list_around[element_type].prev;
		auto base_id_1_prev = mesh[base_id_1].corner_list_around[element_type].prev;
		
		mesh[base_id_0].corner_list_around[element_type].prev = base_id_1_prev;
		mesh[base_id_1_prev].corner_list_around[element_type].next = base_id_0;
		
		mesh[base_id_1].corner_list_around[element_type].prev = base_id_0_prev;
		mesh[base_id_0_prev].corner_list_around[element_type].next = base_id_1;
		
		mesh[element_1].corner_list_base.index = u32_max;
	}
}

struct Allocator {
	compile_const u32 max_memory_block_count = 32;
	
	void* memory_blocks[max_memory_block_count] = {};
	u32 memory_block_count = 0;
};

static void* AllocateMemoryBlock(Allocator& allocator, void* old_memory_block, u64 size_bytes) {
	assert(old_memory_block != nullptr || allocator.memory_block_count < Allocator::max_memory_block_count);
	assert(old_memory_block == nullptr || allocator.memory_block_count > 0);
	assert(old_memory_block == nullptr || allocator.memory_blocks[allocator.memory_block_count - 1] == old_memory_block);
	
	void* memory_block = realloc(old_memory_block, size_bytes);
	u32 memory_block_index = old_memory_block ? allocator.memory_block_count - 1 : allocator.memory_block_count++;
	
	allocator.memory_blocks[memory_block_index] = memory_block;
	
	return memory_block;
}

static void AllocatorFreeMemoryBlocks(Allocator& allocator, u32 last_memory_block_index = 0) {
	for (u32 i = allocator.memory_block_count; i > last_memory_block_index; i -= 1) {
		free(allocator.memory_blocks[i - 1]);
	}
	allocator.memory_block_count = last_memory_block_index;
}


#define ARRAY_OPERATORS \
T& operator[] (u32 index) { assert(index < count); return data[index]; } \
const T& operator[] (u32 index) const { assert(index < count); return data[index]; } \
\
T* begin() { return data; } \
T* end() { return data + count; } \
const T* begin() const { return data; } \
const T* end() const { return data + count; }


// Should be used only with simple types.
template<typename T>
struct Array {
	using ValueType = T;
	
	T* data = nullptr;
	u32 count    = 0;
	u32 capacity = 0;
	
	ARRAY_OPERATORS
};
static_assert(sizeof(Array<u32>) == 16);

template<typename T>
struct ArrayView {
	using ValueType = T;
	
	T* data = nullptr;
	u32 count = 0;
	
	ARRAY_OPERATORS
};
static_assert(sizeof(ArrayView<u32>) == 16);

template<typename T, u32 compile_time_capacity>
struct FixedSizeArray {
	using ValueType = T;
	compile_const u32 capacity = compile_time_capacity;
	
	T data[capacity] = {};
	u32 count = 0;

	ARRAY_OPERATORS
};
#undef ARRAY_OPERATORS

static u32 ArrayComputeNewCapacity(u32 old_capacity, u32 required_capacity = 0) {
	u32 new_capacity = old_capacity ? (old_capacity + old_capacity / 2) : 16;
	return new_capacity > required_capacity ? new_capacity : required_capacity;
}

template<typename T>
static void ArrayReserve(Array<T>& array, Allocator& allocator, u32 capacity) {
	if (array.capacity >= capacity) return;
	
	array.data     = (T*)AllocateMemoryBlock(allocator, array.data, capacity * sizeof(T));
	array.capacity = capacity;
}

template<typename T>
static void ArrayResize(Array<T>& array, Allocator& allocator, u32 new_count) { // Doesn't initialize new elements.
	ArrayReserve(array, allocator, new_count);
	array.count = new_count;
}

template<typename T>
static void ArrayResizeMemset(Array<T>& array, Allocator& allocator, u32 new_count, u8 pattern) { // Fills new elements with a byte pattern.
	ArrayResize(array, allocator, new_count);
	memset(array.data, pattern, new_count * sizeof(T));
}

template<typename ArrayT>
static void ArrayAppend(ArrayT& array, typename ArrayT::ValueType value) {
	assert(array.count < array.capacity);
	array.data[array.count++] = value;
}

template<typename T>
static void ArrayAppendMaybeGrow(Array<T>& array, Allocator& allocator, T value) {
	if (array.count >= array.capacity) {
		ArrayReserve(array, allocator, ArrayComputeNewCapacity(array.capacity, array.count + 1));
	}
	array.data[array.count++] = value;
}

template<typename ArrayT>
static void ArrayEraseSwap(ArrayT& array, u32 index) {
	assert(index < array.count);
	
	array.data[index] = array.data[array.count - 1];
	array.count -= 1;
}

template<typename ArrayT>
static ArrayView<typename ArrayT::ValueType> CreateArrayView(ArrayT array, u32 begin_index, u32 end_index) {
	return { array.data + begin_index, end_index - begin_index };
}

template<typename ArrayT>
static ArrayView<typename ArrayT::ValueType> CreateArrayView(ArrayT& array) {
	return { array.data, array.count };
}


struct VertexHashTable {
	std::vector<VertexID> vertex_ids;
};

static VertexID HashTableAddOrFind(VertexHashTable& table, std::vector<Vertex>& vertices, const Vector3& position) {
	u64 table_size = table.vertex_ids.size();
	u64 mod_mask   = table_size - 1u;
	
	u64 hash  = ComputePositionHash(position);
	u64 index = (hash & mod_mask);
	
	for (u64 i = 0; i <= mod_mask; i += 1) {
		auto vertex_id = table.vertex_ids[index];
		
		if (vertex_id.index == u32_max) {
			auto new_vertex_id = VertexID{ (u32)vertices.size() };
			table.vertex_ids[index] = new_vertex_id;
			
			auto& vertex = vertices.emplace_back();
			vertex.position = position;
			vertex.corner_list_base.index = u32_max;
			
			return new_vertex_id;
		}
		
		if (vertices[vertex_id.index].position == position) {
			return vertex_id;
		}
		
		index = (index + i + 1) & mod_mask;
	}
	
	return VertexID{ u32_max };
}

struct EdgeHashTable {
	std::vector<EdgeID> edge_ids;
};

static EdgeID HashTableAddOrFind(EdgeHashTable& table, std::vector<Edge>& edges, u64 edge_key) {
	u64 table_size = table.edge_ids.size();
	u64 mod_mask   = table_size - 1u;
	
	u64 hash  = std::hash<u64>{}(edge_key);
	u64 index = (hash & mod_mask);
	
	for (u64 i = 0; i <= mod_mask; i += 1) {
		auto edge_id = table.edge_ids[index];
		
		if (edge_id.index == u32_max) {
			auto new_edge_id = EdgeID{ (u32)edges.size() };
			table.edge_ids[index] = new_edge_id;
			
			auto& edge = edges.emplace_back();
			edge.edge_key = edge_key;
			edge.corner_list_base.index = u32_max;
			
			return new_edge_id;
		}
		
		if (edges[edge_id.index].edge_key == edge_key) {
			return edge_id;
		}
		
		index = (index + i + 1) & mod_mask;
	}
	
	return EdgeID{ u32_max };
}

struct EdgeDuplicateMap {
	struct KeyValue {
		u64 edge_key;
		EdgeID edge_id;
	};
	
	std::vector<KeyValue> keys_values;
	u32 element_count = 0;
};


static void HashTableClear(EdgeDuplicateMap& table) {
	memset(table.keys_values.data(), 0xFF, table.keys_values.size() * sizeof(EdgeDuplicateMap::KeyValue));
	table.element_count = 0;
}

static EdgeID HashTableAddOrFind(EdgeDuplicateMap& table, u64 edge_key, EdgeID edge_id);

static void HashTableGrow(EdgeDuplicateMap& table) {
	auto old_keys_values = std::move(table.keys_values);
	table.keys_values.resize(old_keys_values.size() * 2);
	
	HashTableClear(table);
	
	for (auto [key, value] : old_keys_values) {
		if (key != u64_max) HashTableAddOrFind(table, key, value);
	}
}

static EdgeID HashTableAddOrFind(EdgeDuplicateMap& table, u64 edge_key, EdgeID edge_id) {
	u64 table_size = table.keys_values.size();

	compile_const u32 load_factor_percent = 85;
	if ((table.element_count + 1) * 100 >= table_size * load_factor_percent) {
		HashTableGrow(table);
		table_size = table.keys_values.size();
	}
	
	u64 mod_mask = table_size - 1u;
	
	u64 hash  = std::hash<u64>{}(edge_key);
	u64 index = (hash & mod_mask);
	
	for (u64 i = 0; i <= mod_mask; i += 1) {
		auto key_value = table.keys_values[index];
		
		if (key_value.edge_key == u64_max) {
			table.element_count += 1;
			table.keys_values[index] = { edge_key, edge_id };
			return edge_id;
		}
		
		if (key_value.edge_key == edge_key) {
			return key_value.edge_id;
		}
		
		index = (index + i + 1) & mod_mask;
	}
	
	return EdgeID{ u32_max };
}

static u64 ComputeHashTableSize(u64 max_element_count) {
	u64 hash_table_size = 1;
	
	while (hash_table_size < max_element_count + max_element_count / 4) {
		hash_table_size = hash_table_size * 2;
	}
	
	return hash_table_size;
}


Mesh ObjMeshToEditableMesh(ObjTriangleMesh triangle_mesh) {
	std::vector<VertexID> src_vertex_index_to_vertex_id;
	src_vertex_index_to_vertex_id.resize(triangle_mesh.vertices.size());
	
	Mesh result_mesh;
	
	u32 triangle_count = (u32)(triangle_mesh.indices.size() / 3);
	result_mesh.vertices.reserve(triangle_mesh.vertices.size());
	result_mesh.corners.resize(triangle_mesh.indices.size());
	result_mesh.faces.resize(triangle_count);
	result_mesh.edges.reserve(triangle_mesh.indices.size());
	result_mesh.attributes.resize(triangle_mesh.vertices.size() * attribute_stride_dwords);
	
	auto mesh = MeshToMeshView(result_mesh);
	
	
	VertexHashTable table;
	table.vertex_ids.resize(ComputeHashTableSize(triangle_mesh.vertices.size()), VertexID{ u32_max });
	
	for (u32 vertex_index = 0; vertex_index < triangle_mesh.vertices.size(); vertex_index += 1) {
		auto& position = triangle_mesh.vertices[vertex_index].position;
		memcpy(mesh[AttributesID{ vertex_index }], (float*)&triangle_mesh.vertices[vertex_index] + 3, attribute_stride_dwords * sizeof(u32));
		
		auto vertex_id = HashTableAddOrFind(table, result_mesh.vertices, position);
		src_vertex_index_to_vertex_id[vertex_index] = vertex_id;
	}
	mesh.vertex_count = (u32)result_mesh.vertices.size();
	
	
	EdgeHashTable edge_table;
	edge_table.edge_ids.resize(ComputeHashTableSize(triangle_mesh.indices.size()), EdgeID{ u32_max });
	
	u32 active_face_count = 0;
	for (u32 triangle_index = 0; triangle_index < triangle_count; triangle_index += 1) {
		u32 indices[3] = {
			triangle_mesh.indices[triangle_index * 3 + 0],
			triangle_mesh.indices[triangle_index * 3 + 1],
			triangle_mesh.indices[triangle_index * 3 + 2],
		};
		
		VertexID vertex_ids[3] = {
			src_vertex_index_to_vertex_id[indices[0]],
			src_vertex_index_to_vertex_id[indices[1]],
			src_vertex_index_to_vertex_id[indices[2]],
		};
		
		u64 edge_keys[3] = {
			PackEdgeKey(vertex_ids[0], vertex_ids[1]),
			PackEdgeKey(vertex_ids[1], vertex_ids[2]),
			PackEdgeKey(vertex_ids[2], vertex_ids[0]),
		};
		
		bool has_duplicate_vertices = 
			vertex_ids[0].index == vertex_ids[1].index ||
			vertex_ids[0].index == vertex_ids[2].index ||
			vertex_ids[1].index == vertex_ids[2].index;
		
		// Skip primitives that reduce to a single line or a point. PerformEdgeCollapse(...) can't reliably handle them.
		if (has_duplicate_vertices) continue;
		
		
		auto face_id = FaceID{ active_face_count };
		active_face_count += 1;
		
		mesh[face_id].corner_list_base.index = u32_max;
		
		for (u32 corner_index = 0; corner_index < 3; corner_index += 1) {
			auto corner_id = CornerID{ face_id.index * 3 + corner_index };
			
			auto edge_id = HashTableAddOrFind(edge_table, result_mesh.edges, edge_keys[corner_index]);
			
			auto& corner = mesh[corner_id];
			corner.face_id       = face_id;
			corner.edge_id       = edge_id;
			corner.vertex_id     = vertex_ids[corner_index];
			corner.attributes_id = { indices[corner_index] };
			
			CornerListInsert<VertexID>(mesh, corner.vertex_id, corner_id);
			CornerListInsert<EdgeID>(mesh, corner.edge_id, corner_id);
			CornerListInsert<FaceID>(mesh, corner.face_id, corner_id);
		}
	}
	result_mesh.faces.resize(active_face_count);
	
	return result_mesh;
}

ObjTriangleMesh EditableMeshToObjMesh(MeshView mesh) {
	std::vector<VertexID> attribute_vertex_ids;
	attribute_vertex_ids.resize(mesh.attribute_count, VertexID{ u32_max });
	
	std::vector<u32> attribute_id_remap;
	attribute_id_remap.resize(mesh.attribute_count);
	
	
	for (VertexID vertex_id = { 0 }; vertex_id.index < mesh.vertex_count; vertex_id.index += 1) {
		auto& vertex = mesh[vertex_id];
		
		if (vertex.corner_list_base.index != u32_max) {
			IterateCornerList<ElementType::Vertex>(mesh, vertex.corner_list_base, [&](CornerID corner_id) {
				u32 attribute_index = mesh[corner_id].attributes_id.index;
				attribute_vertex_ids[attribute_index] = vertex_id;
			});
		}
	}
	
	u32 attribute_count = 0;
	for (AttributesID attribute_id = { 0 }; attribute_id.index < mesh.attribute_count; attribute_id.index += 1) {
		bool is_active = (attribute_vertex_ids[attribute_id.index].index != u32_max);
		attribute_id_remap[attribute_id.index] = attribute_count;
		attribute_count += is_active ? 1 : 0;
	}
	
	u32 face_count = 0;
	for (FaceID face_id = { 0 }; face_id.index < mesh.face_count; face_id.index += 1) {
		face_count +=  mesh[face_id].corner_list_base.index != u32_max ? 1 : 0;
	}
	
	ObjTriangleMesh triangle_mesh;
	triangle_mesh.indices.reserve(face_count * 3);
	triangle_mesh.vertices.reserve(attribute_count);
	
	for (FaceID face_id = { 0 }; face_id.index < mesh.face_count; face_id.index += 1) {
		auto& face = mesh[face_id];
		if (face.corner_list_base.index == u32_max) continue;
		
		IterateCornerList<ElementType::Face>(mesh, face.corner_list_base, [&](CornerID corner_id) {
			auto& corner = mesh[corner_id];
			triangle_mesh.indices.emplace_back(attribute_id_remap[corner.attributes_id.index]);
		});
	}
	
	for (AttributesID attribute_id = { 0 }; attribute_id.index < mesh.attribute_count; attribute_id.index += 1) {
		auto vertex_id = attribute_vertex_ids[attribute_id.index];
		if (vertex_id.index != u32_max) {
			auto& vertex = triangle_mesh.vertices.emplace_back();
			vertex.position = mesh[vertex_id].position;
			memcpy((float*)&vertex + 3, mesh[attribute_id], attribute_stride_dwords * sizeof(u32));
		}
	}
	
	return triangle_mesh;
}

MeshView MeshToMeshView(Mesh& mesh) {
	MeshView view;
	view.faces           = mesh.faces.data();
	view.edges           = mesh.edges.data();
	view.vertices        = mesh.vertices.data();
	view.corners         = mesh.corners.data();
	view.attributes      = mesh.attributes.data();
	view.face_count      = (u32)mesh.faces.size();
	view.edge_count      = (u32)mesh.edges.size();
	view.vertex_count    = (u32)mesh.vertices.size();
	view.corner_count    = (u32)mesh.corners.size();
	view.attribute_count = (u32)mesh.attributes.size() / attribute_stride_dwords;
	return view;
}

using RemovedEdgeArray = std::vector<EdgeID>;

struct EdgeCollapseResult {
	CornerID remaning_base_id;
	u32 removed_face_count = 0;
};

static EdgeCollapseResult PerformEdgeCollapse(MeshView mesh, EdgeID edge_id, EdgeDuplicateMap& edge_duplicate_map, RemovedEdgeArray& removed_edge_array) {
	auto& edge = mesh[edge_id];
	
	assert(edge.vertex_0.index != edge.vertex_1.index);
	auto& vertex_0 = mesh[edge.vertex_0];
	auto& vertex_1 = mesh[edge.vertex_1];

	assert(edge.corner_list_base.index != u32_max);
	assert(vertex_0.corner_list_base.index != u32_max);
	assert(vertex_1.corner_list_base.index != u32_max);
	assert(vertex_0.corner_list_base.index != vertex_1.corner_list_base.index);
	
	removed_edge_array.clear();
	u32 removed_face_count = 0;
	IterateCornerList<ElementType::Edge>(mesh, edge.corner_list_base, [&](CornerID corner_id) {
		auto& corner = mesh[corner_id];
		auto& face   = mesh[corner.face_id];
		
		IterateCornerList<ElementType::Face>(mesh, face.corner_list_base, [&](CornerID corner_id) {
			CornerListRemove<ElementType::Vertex>(mesh, corner_id);
			bool edge_removed = CornerListRemove<ElementType::Edge>(mesh, corner_id);
			bool face_removed = CornerListRemove<ElementType::Face>(mesh, corner_id);
			
			if (edge_removed) removed_edge_array.push_back(mesh[corner_id].edge_id);
			removed_face_count += (u32)face_removed;
		});
	});
	
	CornerListMerge<VertexID>(mesh, edge.vertex_0, edge.vertex_1);
	
	auto remaning_base_id = vertex_0.corner_list_base.index != u32_max ? vertex_0.corner_list_base : vertex_1.corner_list_base;
	if (remaning_base_id.index != u32_max) {
		IterateCornerList<ElementType::Vertex>(mesh, remaning_base_id, [&](CornerID corner_id) {
			// TODO: This can be iteration over just incoming and outgoing edges of a corner.
			IterateCornerList<ElementType::Face>(mesh, corner_id, [&](CornerID corner_id) {
				auto edge_id_1 = mesh[corner_id].edge_id;
				auto& edge_1 = mesh[edge_id_1];
				
				auto edge_id_0 = HashTableAddOrFind(edge_duplicate_map, PackEdgeKey(edge_1.vertex_0, edge_1.vertex_1), edge_id_1);
				if (edge_id_0.index != edge_id_1.index) {
					CornerListMerge<EdgeID>(mesh, edge_id_0, edge_id_1);
					removed_edge_array.push_back(edge_id_1);
				}
			});
		});
		
		IterateCornerList<ElementType::Vertex>(mesh, remaning_base_id, [&](CornerID corner_id_0) {
			IterateCornerList<ElementType::Face>(mesh, corner_id_0, [&](CornerID corner_id_1) {
				if (remaning_base_id.index == corner_id_1.index) return;
				
				IterateCornerList<ElementType::Vertex>(mesh, corner_id_1, [&](CornerID corner_id_2) {
					if (corner_id_1.index == corner_id_2.index) return;
					
					IterateCornerList<ElementType::Face>(mesh, corner_id_2, [&](CornerID corner_id) {
						auto edge_id = mesh[corner_id].edge_id;
						auto& edge = mesh[edge_id];
						
						HashTableAddOrFind(edge_duplicate_map, PackEdgeKey(edge.vertex_0, edge.vertex_1), edge_id);
					});
				});
			});
		});
	}
	
	
	EdgeCollapseResult result;
	result.remaning_base_id   = remaning_base_id;
	result.removed_face_count = removed_face_count;
	
	return result;
}


//
// Basic geometric quadric. For reference see [Garland and Heckbert 1997], [Hugues Hoppe 1999].
// Notation is based on [Hugues Hoppe 1999].
//
struct Quadric {
	//
	// Symmetric matrix A:
	// (a00, a01, a02)
	// (a01, a11, a12)
	// (a02, a12, a22)
	//
	float a00 = 0.f;
	float a11 = 0.f;
	float a22 = 0.f;
	
	float a01 = 0.f;
	float a02 = 0.f;
	float a12 = 0.f;
	
	Vector3 b = { 0.f, 0.f, 0.f };
	
	float c = 0.f;
	
	float weight = 0.f;
};

#define ENABLE_ATTRIBUTE_SUPPORT 1

struct QuadricWithAttributes : Quadric {
#if ENABLE_ATTRIBUTE_SUPPORT
	struct {
		Vector3 g = { 0.f, 0.f, 0.f };
		float d = 0.f;
	} attributes[attribute_stride_dwords];
#endif // ENABLE_ATTRIBUTE_SUPPORT
};


//
// Matt Pharr's blog. 2019. Accurate Differences of Products with Kahan's Algorithm.
//
static float DifferenceOfProducts(float a, float b, float c, float d) {
	float cd = c * d;
	float err = fmaf(c, -d,  cd);
	float dop = fmaf(a,  b, -cd);
	return dop + err;
}

static Vector3 operator+ (const Vector3& lh, const Vector3& rh) { return Vector3{ lh.x + rh.x, lh.y + rh.y, lh.z + rh.z }; }
static Vector3 operator- (const Vector3& lh, const Vector3& rh) { return Vector3{ lh.x - rh.x, lh.y - rh.y, lh.z - rh.z }; }
static Vector3 operator* (const Vector3& lh, float rh) { return Vector3{ lh.x * rh, lh.y * rh, lh.z * rh }; }
static float DotProduct(const Vector3& lh, const Vector3& rh) { return lh.x * rh.x + lh.y * rh.y + lh.z * rh.z; }
static float Length(const Vector3& v) { return sqrtf(DotProduct(v, v)); }
static Vector3 Normalize(const Vector3& v) { float length = Length(v); return length < FLT_EPSILON ? v : v * (1.f / length); }
static Vector3 CrossProduct(const Vector3& lh, const Vector3& rh) { return Vector3{ lh.y * rh.z - lh.z * rh.y, lh.z * rh.x - lh.x * rh.z, lh.x * rh.y - lh.y * rh.x }; }


static void AccumulateQuadric(Quadric& accumulator, const Quadric& quadric) {
	accumulator.a00 += quadric.a00;
	accumulator.a11 += quadric.a11;
	accumulator.a22 += quadric.a22;
	accumulator.a01 += quadric.a01;
	accumulator.a02 += quadric.a02;
	accumulator.a12 += quadric.a12;
	
	accumulator.b.x += quadric.b.x;
	accumulator.b.y += quadric.b.y;
	accumulator.b.z += quadric.b.z;
	
	accumulator.c += quadric.c;
	accumulator.weight += quadric.weight;
}

static void AccumulateQuadricWithAttributes(QuadricWithAttributes& accumulator, const QuadricWithAttributes& quadric) {
	AccumulateQuadric(accumulator, quadric);
	
#if ENABLE_ATTRIBUTE_SUPPORT
	for (u32 i = 0; i < attribute_stride_dwords; i += 1) {
		auto& attribute_accumulator = accumulator.attributes[i];
		auto& attribute_quadric     = quadric.attributes[i];

		attribute_accumulator.g.x += attribute_quadric.g.x;
		attribute_accumulator.g.y += attribute_quadric.g.y;
		attribute_accumulator.g.z += attribute_quadric.g.z;
		attribute_accumulator.d   += attribute_quadric.d;
	}
#endif // ENABLE_ATTRIBUTE_SUPPORT
}

static void ComputePlanarQuadric(Quadric& quadric, const Vector3& n, float d, float weight) {
	//
	// For reference see [Hugues Hoppe 1999] Section 3 Previous Quadric Error Metrics.
	//
	// A = n * n^T
	// b = d * n
	// c = d^2
	//
	// (n.x)
	// (n.y) * (n.x, n.y, n.z)
	// (n.z)
	//
	quadric.a00 = (n.x * n.x) * weight;
	quadric.a11 = (n.y * n.y) * weight;
	quadric.a22 = (n.z * n.z) * weight;
	quadric.a01 = (n.x * n.y) * weight;
	quadric.a02 = (n.x * n.z) * weight;
	quadric.a12 = (n.y * n.z) * weight;
	
	quadric.b.x = (n.x * d) * weight;
	quadric.b.y = (n.y * d) * weight;
	quadric.b.z = (n.z * d) * weight;
	
	quadric.c = (d * d) * weight;
	quadric.weight = weight;
}

// Assumes that the quadric edge is (p0, p1)
static void ComputeEdgeQuadric(Quadric& quadric, const Vector3& p0, const Vector3& p1, const Vector3& p2, float weight) {
	auto p10 = p1 - p0;
	auto p20 = p2 - p0;
	
	auto face_normal_direction = CrossProduct(p10, p20);
	auto normal_direction      = CrossProduct(p10, face_normal_direction);
	auto normal_length         = Length(normal_direction);
	auto normal                = normal_length < FLT_EPSILON ? normal_direction : normal_direction * (1.f / normal_length); // TODO: Potential division by zero. Handle better.
	auto distance_to_triangle  = -DotProduct(normal, p1);
	
	ComputePlanarQuadric(quadric, normal, distance_to_triangle, DotProduct(p10, p10) * weight);
}

// TODO: Convert these constants into user editable settings.
compile_const float attribute_weight = 1.f / (1024.f * 1024.f);
compile_const float inversion_error  = 1.f;

static void ComputeFaceQuadricWithAttributes(QuadricWithAttributes& quadric, const Vector3& p0, const Vector3& p1, const Vector3& p2, float* a0, float* a1, float* a2) {
	auto p10 = p1 - p0;
	auto p20 = p2 - p0;
	
	auto scaled_normal       = CrossProduct(p10, p20);
	auto twice_triangle_area = Length(scaled_normal);
	auto n                   = twice_triangle_area < FLT_EPSILON ? scaled_normal : scaled_normal * (1.f / twice_triangle_area); // TODO: Potential division by zero. Handle better.
	
	float weight = twice_triangle_area * 0.5f;
	ComputePlanarQuadric(quadric, n, -DotProduct(n, p0), weight);
	
	
#if ENABLE_ATTRIBUTE_SUPPORT
	//
	// For reference see [Hugues Hoppe 1999] Section 4 New Quadric Error Metric.
	//
	// (p0^T, 1)   (g.x)   (s0);
	// (p1^T, 1) * (g.z) = (s1);
	// (p2^T, 1)   (g.y)   (s2);
	// (n^T,  0)   ( d )   (0);
	//
	// ((p1 - p0)^T)   (g.x)   (s1 - s0)
	// ((p2 - p0)^T) * (g.y) = (s2 - s0)
	// (    n^T    )   (g.z)   (   0   )
	//
	// (p10^T)   (g.x)   (s10)
	// (p20^T) * (g.y) = (s20)
	// ( n^T )   (g.z)   ( 0 )
	//
	// A * x = b
	//
	
	// Compute determinant of a 3x3 matrix A with rows p10, p20, n.
	float det0 = p10.x * DifferenceOfProducts(p20.y, n.z, p20.z, n.y);
	float det1 = p10.y * DifferenceOfProducts(p20.z, n.x, p20.x, n.z);
	float det2 = p10.z * DifferenceOfProducts(p20.x, n.y, p20.y, n.x);
	float determinant = det0 + det1 + det2;
	float determinant_rcp = fabsf(determinant) < FLT_EPSILON ? 0.f : 1.f / determinant;
	
	// Compute first two colums of A^-1.
	float a_inv_00 = DifferenceOfProducts(p20.y, n.z, p20.z, n.y) * determinant_rcp;
	float a_inv_01 = DifferenceOfProducts(p10.z, n.y, p10.y, n.z) * determinant_rcp;
	float a_inv_10 = DifferenceOfProducts(p20.z, n.x, p20.x, n.z) * determinant_rcp;
	float a_inv_11 = DifferenceOfProducts(p10.x, n.z, p10.z, n.x) * determinant_rcp;
	float a_inv_20 = DifferenceOfProducts(p20.x, n.y, p20.y, n.x) * determinant_rcp;
	float a_inv_21 = DifferenceOfProducts(p10.y, n.x, p10.x, n.y) * determinant_rcp;
	
	for (u32 i = 0; i < attribute_stride_dwords; i += 1) {
		float s0 = a0[i] * attribute_weight;
		float s1 = a1[i] * attribute_weight;
		float s2 = a2[i] * attribute_weight;
		
		float s10 = s1 - s0;
		float s20 = s2 - s0;
		
		Vector3 g;
		g.x = a_inv_00 * s10 + a_inv_01 * s20;
		g.y = a_inv_10 * s10 + a_inv_11 * s20;
		g.z = a_inv_20 * s10 + a_inv_21 * s20;
		
		float d = s0 - DotProduct(p0, g);
		
		//
		// A += g * g^T
		// b += d * g
		// c += d^2
		//
		quadric.a00 += (g.x * g.x) * weight;
		quadric.a11 += (g.y * g.y) * weight;
		quadric.a22 += (g.z * g.z) * weight;
		quadric.a01 += (g.x * g.y) * weight;
		quadric.a02 += (g.x * g.z) * weight;
		quadric.a12 += (g.y * g.z) * weight;
		
		quadric.b.x += (d * g.x) * weight;
		quadric.b.y += (d * g.y) * weight;
		quadric.b.z += (d * g.z) * weight;
		
		quadric.c += (d * d) * weight;
		
		quadric.attributes[i].g = g * weight;
		quadric.attributes[i].d = d * weight;
	}
#endif // ENABLE_ATTRIBUTE_SUPPORT
}

static float ComputeQuadricError(const Quadric& q, const Vector3& p) {
	//
	// error = p^T * A * p + 2 * b * v + c
	//
	//                   (a00, a01, a02)   (p.x)
	// (p.x, p.y, p.z) * (a01, a11, a12) * (p.y) + 2 * b * p + c
	//                   (a02, a12, a22)   (p.z)
	// 
	float weighted_error = 
		p.x * (p.x * q.a00 + p.y * q.a01 + p.z * q.a02) +
		p.y * (p.x * q.a01 + p.y * q.a11 + p.z * q.a12) +
		p.z * (p.x * q.a02 + p.y * q.a12 + p.z * q.a22) +
		2.f * DotProduct(q.b, p) +
		q.c;
	
	return fabsf(weighted_error);
}

static float ComputeQuadricErrorWithAttributes(const QuadricWithAttributes& q, const Vector3& p) {
	//
	// error = p^T * A * p + 2 * b * v + c
	//
	//                           ( a00,   a01,   a02,  -g0.x, -gi.x)   (p.x)
	// (p.x, p.y, p.z, s0, si) * ( a01,   a11,   a12,  -g0.y, -gi.y) * (p.y) + 2 * (b, -d0, -di) * (p, s0, si) + (c + d0^2 + di^2)
	//                           ( a02,   a12,   a22,  -g0.z, -gi.z)   (p.z)
	//                           (-g0.x, -g0.y, -g0.z,  1.0,   0.0 )   (s0 )
	//                           (-gi.x, -gi.y, -gi.z,  0.0,   1.0 )   (si )
	// 
	float weighted_error = ComputeQuadricError(q, p);
	
#if ENABLE_ATTRIBUTE_SUPPORT
	if (q.weight < FLT_EPSILON) return weighted_error;
	
	float rcp_weight = 1.f / q.weight;
	for (u32 i = 0; i < attribute_stride_dwords; i += 1) {
		auto g = q.attributes[i].g;
		auto d = q.attributes[i].d;
		
		float s = (DotProduct(g, p) + d) * rcp_weight;
		
		//
		// Simplified by replacing first three lines with a dot product, and substituting -DotProduct(g, p) for (d - s * q.weight).
		//
		// p.x * (-g.x * s) +
		// p.y * (-g.y * s) +
		// p.z * (-g.z * s) +
		// s * (-DotProduct(g, p) + s) +
		// -2.f * d * s;
		//
		float weighted_attribute_error = s * s * (1.f - 2.f * q.weight);
		
		weighted_error += weighted_attribute_error;
	}
#endif // ENABLE_ATTRIBUTE_SUPPORT
	
	return fabsf(weighted_error);
}

#if ENABLE_ATTRIBUTE_SUPPORT
// Attribute computation for zero weight quadrics should be handled by the caller.
static bool ComputeWedgeAttributes(const QuadricWithAttributes& q, const Vector3& p, float* attributes) {
	if constexpr (attribute_weight <= 0.f) return false;
	if (q.weight < FLT_EPSILON) return false;
	
	float rcp_weight = 1.f / q.weight;
	for (u32 i = 0; i < attribute_stride_dwords; i += 1) {
		auto g = q.attributes[i].g;
		auto d = q.attributes[i].d;
		
		float s = (DotProduct(g, p) + d) * rcp_weight;
		
		attributes[i] = s * (1.f / attribute_weight);
	}
	
	return true;
}
#endif // ENABLE_ATTRIBUTE_SUPPORT

static bool ComputeOptimalVertexPosition(const QuadricWithAttributes& quadric, Vector3& optimal_position) {
	//
	// For reference see [Hugues Hoppe 2000] Section 3 Quadric Metric Minimization.
	//
	// Note that definition of b = (d, d0, di) is negated in [Hugues Hoppe 2000] relative to [Hugues Hoppe 1999].
	// We're using definitions from [Hugues Hoppe 2000] where g*p + d = s.
	//
	
	if (quadric.weight < FLT_EPSILON) return false;
	
	// K = B * B^T
	float k00 = 0.f;
	float k11 = 0.f;
	float k22 = 0.f;
	float k01 = 0.f;
	float k02 = 0.f;
	float k12 = 0.f;
	
	// h = B * b2
	float h0 = 0.f;
	float h1 = 0.f;
	float h2 = 0.f;
	
#if ENABLE_ATTRIBUTE_SUPPORT
	for (u32 i = 0; i < attribute_stride_dwords; i += 1) {
		auto g = quadric.attributes[i].g;
		auto d = quadric.attributes[i].d;
		
		// B * B^T
		k00 += (g.x * g.x);
		k11 += (g.y * g.y);
		k22 += (g.z * g.z);
		k01 += (g.x * g.y);
		k02 += (g.x * g.z);
		k12 += (g.y * g.z);
		
		// B * b2
		h0 += ((float)g.x * (float)d);
		h1 += ((float)g.y * (float)d);
		h2 += ((float)g.z * (float)d);
	}
#endif // ENABLE_ATTRIBUTE_SUPPORT
	
	// M = C - B * B^T * (1.0 / alpha)
	float rcp_weight = 1.f / quadric.weight;
	float m00 = quadric.a00 - k00 * rcp_weight;
	float m11 = quadric.a11 - k11 * rcp_weight;
	float m22 = quadric.a22 - k22 * rcp_weight;
	float m01 = quadric.a01 - k01 * rcp_weight;
	float m02 = quadric.a02 - k02 * rcp_weight;
	float m12 = quadric.a12 - k12 * rcp_weight;
	
	// Note that this expression is negated relative to [Hugues Hoppe 2000] because our b is using notation
	// from [Hugues Hoppe 1999] where it's negated. So both h and quadric.b are negated.
	float j0 = (h0 * rcp_weight - quadric.b.x);
	float j1 = (h1 * rcp_weight - quadric.b.y);
	float j2 = (h2 * rcp_weight - quadric.b.z);
	
	// Determinant of M.
	float det0 = m00 * DifferenceOfProducts(m11, m22, m12, m12);
	float det1 = m01 * DifferenceOfProducts(m01, m22, m12, m02);
	float det2 = m02 * DifferenceOfProducts(m01, m12, m11, m02);
	float determinant = det0 - det1 + det2;
	
	//
	// As an alternative to inverting M [HSUEH-TI DEREK LIU 2024] suggests Cholesky decomposition.
	// LU decomposition should work too. SVD could be used to find least squares solution.
	//
	if (fabsf(determinant) < FLT_EPSILON) return false;
	float determinant_rcp = 1.f / determinant;
	
	//
	// M inverse. Has to be computed with extra precision, otherwise vertex placement is all over the place.
	//
	float m_inv_00 = DifferenceOfProducts(m11, m22, m12, m12);
	float m_inv_01 = DifferenceOfProducts(m02, m12, m01, m22);
	float m_inv_02 = DifferenceOfProducts(m01, m12, m02, m11);
	float m_inv_11 = DifferenceOfProducts(m00, m22, m02, m02);
	float m_inv_12 = DifferenceOfProducts(m02, m01, m00, m12);
	float m_inv_22 = DifferenceOfProducts(m00, m11, m01, m01);
	
	optimal_position.x = (m_inv_00 * j0 + m_inv_01 * j1 + m_inv_02 * j2) * determinant_rcp;
	optimal_position.y = (m_inv_01 * j0 + m_inv_11 * j1 + m_inv_12 * j2) * determinant_rcp;
	optimal_position.z = (m_inv_02 * j0 + m_inv_12 * j1 + m_inv_22 * j2) * determinant_rcp;
	
	return true;
}

// Check if any triangle around the collapsed edge is flipped or becomes zero area, excluding collapsed triangles.
static u32 ValidateEdgeCollapsePositions(MeshView mesh, Edge edge, Vector3* candidate_positions, u32 candidate_position_count) {
	u32 valid_position_mask = (1u << candidate_position_count) - 1u;
	
	auto check_triangle_flip_for_vertex = [&](CornerID corner_id)-> IterationControl {
		auto& c1 = mesh[corner_id];
		
		auto v0 = mesh[c1.corner_list_around[(u32)ElementType::Face].prev].vertex_id;
		auto v1 = c1.vertex_id;
		auto v2 = mesh[c1.corner_list_around[(u32)ElementType::Face].next].vertex_id;
		
		Vector3 p[3] = {
			mesh[v0].position,
			mesh[v1].position,
			mesh[v2].position,
		};
		
		auto n0 = CrossProduct(p[1] - p[0], p[2] - p[0]);
		
		u32 replaced_vertex_count = 0;
		u32 replaced_vertex_index = 0;
		if (v0.index == edge.vertex_0.index || v0.index == edge.vertex_1.index) { replaced_vertex_count += 1; replaced_vertex_index = 0; }
		if (v1.index == edge.vertex_0.index || v1.index == edge.vertex_1.index) { replaced_vertex_count += 1; replaced_vertex_index = 1; }
		if (v2.index == edge.vertex_0.index || v2.index == edge.vertex_1.index) { replaced_vertex_count += 1; replaced_vertex_index = 2; }
		
		// Replaced vertex count == 0 is impossible.
		// Replaced vertex count == 2 is true for triangles that would get collapsed, we don't need to check if they flip.
		if (replaced_vertex_count == 1) {
			for (u32 i = 0; i < candidate_position_count; i += 1) {
				p[replaced_vertex_index] = candidate_positions[i];
				
				auto n1 = CrossProduct(p[1] - p[0], p[2] - p[0]);
				
				// Prevent flipped or zero area triangles.
				bool reject_edge_collapse = (DotProduct(n0, n1) < 0.f) || (DotProduct(n0, n0) > FLT_EPSILON && DotProduct(n1, n1) < FLT_EPSILON);
				
				if (reject_edge_collapse) {
					valid_position_mask &= ~(1u << i);
				}
			}
		}
		
		return valid_position_mask ? IterationControl::Continue : IterationControl::Break;
	};
	
	if (valid_position_mask) IterateCornerList<ElementType::Vertex>(mesh, mesh[edge.vertex_0].corner_list_base, check_triangle_flip_for_vertex);
	if (valid_position_mask) IterateCornerList<ElementType::Vertex>(mesh, mesh[edge.vertex_1].corner_list_base, check_triangle_flip_for_vertex);
	
	return valid_position_mask;
}


struct AttributeWedgeMap {
	compile_const u32 capacity = 256;
	
	AttributesID keys[capacity];
	u8 values[capacity];
	u32 count = 0;
};

static void AttributeWedgeMapAdd(AttributeWedgeMap& small_set, AttributesID key, u8 value) {
	u32 index = (small_set.count < AttributeWedgeMap::capacity) ? small_set.count++ : (AttributeWedgeMap::capacity - 1);
	small_set.keys[index]   = key;
	small_set.values[index] = value;
}

static u8 AttributeWedgeMapFind(AttributeWedgeMap& small_set, AttributesID key) {
	for (u32 i = 0; i < small_set.count; i += 1) {
		if (small_set.keys[i].index == key.index) return small_set.values[i];
	}
	
	return 0;
}

static void ResetAttributeWedgeMap(AttributeWedgeMap& small_set) {
	small_set.count = 0;
}


struct MeshDecimationState {
	// Edge quadrics accumulated on vertices.
	std::vector<Quadric> vertex_edge_quadrics;
	
	// Face quadrics accumulated on attributes.
	std::vector<QuadricWithAttributes> attribute_face_quadrics;
	
	EdgeDuplicateMap edge_duplicate_map;
	RemovedEdgeArray removed_edge_array;
	
	AttributeWedgeMap wedge_attribute_set;
	std::vector<QuadricWithAttributes> wedge_quadrics;
	std::vector<float> wedge_attributes;
	std::vector<AttributesID> wedge_attributes_ids;
};

struct EdgeSelectInfo {
	float min_error = FLT_MAX;
	Vector3 new_position;
};

void ComputeEdgeCollapseError(MeshView mesh, MeshDecimationState& state, EdgeID edge_id, EdgeSelectInfo* info = nullptr) {
	ResetAttributeWedgeMap(state.wedge_attribute_set);
	state.wedge_quadrics.clear();
	state.wedge_attributes_ids.clear();
	
	auto& edge = mesh[edge_id];
	
	// Wedges spanning collapsed edge must be unified. Manually set their wedge index to the same value and accumulate quadrics.
	// For reference see [Hugues Hoppe 1999] Section 5 Attribute Discontinuities, Figure 5.
	IterateCornerList<ElementType::Edge>(mesh, edge.corner_list_base, [&](CornerID corner_id) {
		auto& corner_0 = mesh[corner_id];
		auto& corner_1 = mesh[corner_0.corner_list_around[(u32)ElementType::Face].next];
		
		u8 wedge_index_0 = AttributeWedgeMapFind(state.wedge_attribute_set, corner_0.attributes_id);
		u8 wedge_index_1 = AttributeWedgeMapFind(state.wedge_attribute_set, corner_1.attributes_id);
		
		if (wedge_index_0 == 0 && wedge_index_1 == 0) {
			u8 index = (u8)(state.wedge_quadrics.size() + 1);
			AttributeWedgeMapAdd(state.wedge_attribute_set, corner_0.attributes_id, index);
			AttributeWedgeMapAdd(state.wedge_attribute_set, corner_1.attributes_id, index);
			
			state.wedge_attributes_ids.emplace_back(corner_0.attributes_id);
			auto& quadric = state.wedge_quadrics.emplace_back(state.attribute_face_quadrics[corner_0.attributes_id.index]);
			AccumulateQuadricWithAttributes(quadric, state.attribute_face_quadrics[corner_1.attributes_id.index]);
		} else if (wedge_index_0 == 0 && wedge_index_1 != 0) {
			AttributeWedgeMapAdd(state.wedge_attribute_set, corner_0.attributes_id, wedge_index_1);
			AccumulateQuadricWithAttributes(state.wedge_quadrics[wedge_index_1 - 1], state.attribute_face_quadrics[corner_0.attributes_id.index]);
		} else if (wedge_index_0 != 0 && wedge_index_1 == 0) {
			AttributeWedgeMapAdd(state.wedge_attribute_set, corner_1.attributes_id, wedge_index_0);
			AccumulateQuadricWithAttributes(state.wedge_quadrics[wedge_index_0 - 1], state.attribute_face_quadrics[corner_1.attributes_id.index]);
		}
	});
	
	auto accumulate_quadrics = [&](CornerID corner_id) {
		auto attribute_id = mesh[corner_id].attributes_id;
		
		if (AttributeWedgeMapFind(state.wedge_attribute_set, attribute_id) == 0) {
			AttributeWedgeMapAdd(state.wedge_attribute_set, attribute_id, (u8)(state.wedge_quadrics.size() + 1));
			state.wedge_quadrics.push_back(state.attribute_face_quadrics[attribute_id.index]);
			state.wedge_attributes_ids.emplace_back(attribute_id);
		}
	};
	
	auto& v0 = mesh[edge.vertex_0];
	auto& v1 = mesh[edge.vertex_1];
	
	IterateCornerList<ElementType::Vertex>(mesh, v0.corner_list_base, accumulate_quadrics);
	IterateCornerList<ElementType::Vertex>(mesh, v1.corner_list_base, accumulate_quadrics);
	
	if (info != nullptr) {
		auto edge_quadrics = state.vertex_edge_quadrics[edge.vertex_0.index];
		AccumulateQuadric(edge_quadrics, state.vertex_edge_quadrics[edge.vertex_1.index]);
		
		// Try a few different positions for the new vertex.
		compile_const u32 candidate_position_count = 3;
		Vector3 candidate_positions[candidate_position_count];
		candidate_positions[0] = v0.position;
		candidate_positions[1] = v1.position;
		candidate_positions[2] = (candidate_positions[0] + candidate_positions[1]) * 0.5f;
		
		QuadricWithAttributes total_quadric;
		AccumulateQuadric(total_quadric, edge_quadrics);
		
		u32 wedge_count = (u32)state.wedge_quadrics.size();
		for (u32 i = 0; i < wedge_count; i += 1) {
			 AccumulateQuadricWithAttributes(total_quadric, state.wedge_quadrics[i]);
		}
		
		// Override average with optimal position if it can be computed.
		ComputeOptimalVertexPosition(total_quadric, candidate_positions[2]);
		
		u32 valid_position_mask = ValidateEdgeCollapsePositions(mesh, edge, candidate_positions, candidate_position_count);
		for (u32 i = 0; i < candidate_position_count; i += 1) {
			auto& p = candidate_positions[i];
			
			float error = valid_position_mask & (1u << i) ? 0.f : inversion_error;
			if (error > info->min_error) continue;
			
			
			error += ComputeQuadricError(edge_quadrics, p);
			if (error > info->min_error) continue;
			
			for (u32 i = 0; i < wedge_count; i += 1) {
				error += ComputeQuadricErrorWithAttributes(state.wedge_quadrics[i], p);
			}
			if (error > info->min_error) continue;
			
			
			info->min_error = error;
			info->new_position = p;
		}
	}
}

struct EdgeCollapseHeap {
	std::vector<EdgeID> heap_index_to_edge_id;
	std::vector<u32>    edge_id_to_heap_index;
	std::vector<float>  edge_collapse_errors;
};

static u32 HeapChildIndex0(u32 node_index) { return node_index * 2 + 1; }
static u32 HeapChildIndex1(u32 node_index) { return node_index * 2 + 2; }
static u32 HeapParentIndex(u32 node_index) { return (node_index - 1) / 2; }

static void EdgeCollapseHeapSwapElements(EdgeCollapseHeap& heap, u32 node_index_0, u32 node_index_1) {
	auto node_0_edge_id = heap.heap_index_to_edge_id[node_index_0];
	auto node_1_edge_id = heap.heap_index_to_edge_id[node_index_1];
	auto node_0_error   = heap.edge_collapse_errors[node_index_0];
	auto node_1_error   = heap.edge_collapse_errors[node_index_1];
	
	heap.heap_index_to_edge_id[node_index_0] = node_1_edge_id;
	heap.edge_collapse_errors[node_index_0]  = node_1_error;
	
	heap.heap_index_to_edge_id[node_index_1] = node_0_edge_id;
	heap.edge_collapse_errors[node_index_1]  = node_0_error;
	
	heap.edge_id_to_heap_index[node_1_edge_id.index] = node_index_0;
	heap.edge_id_to_heap_index[node_0_edge_id.index] = node_index_1;
}

static void EdgeCollapseHeapSiftUp(EdgeCollapseHeap& heap, u32 node_index) {
	while (node_index) {
		u32 parent_node_index = HeapParentIndex(node_index);
		
		if (heap.edge_collapse_errors[node_index] < heap.edge_collapse_errors[parent_node_index]) {
			EdgeCollapseHeapSwapElements(heap, node_index, parent_node_index);
		}
		
		node_index = parent_node_index;
	}
}

static void EdgeCollapseHeapSiftDown(EdgeCollapseHeap& heap, u32 node_index) {
	u32 element_count = (u32)heap.edge_collapse_errors.size();
	while (HeapChildIndex0(node_index) < element_count) {
		u32 index_0 = HeapChildIndex0(node_index);
		u32 index_1 = HeapChildIndex1(node_index);
		
		u32 smallest_child_index = index_1 >= element_count ? index_0 : (heap.edge_collapse_errors[index_0] < heap.edge_collapse_errors[index_1] ? index_0 : index_1);
		
		if (heap.edge_collapse_errors[node_index] > heap.edge_collapse_errors[smallest_child_index]) {
			EdgeCollapseHeapSwapElements(heap, node_index, smallest_child_index);
		}
		
		node_index = smallest_child_index;
	}
}

static EdgeID EdgeCollapseHeapPop(EdgeCollapseHeap& heap) {
	auto edge_id = heap.heap_index_to_edge_id.front();

	if (heap.edge_collapse_errors.size() > 0) {
		heap.edge_collapse_errors[0]  = heap.edge_collapse_errors.back();
		heap.heap_index_to_edge_id[0] = heap.heap_index_to_edge_id.back();
	}
	
	heap.edge_id_to_heap_index[heap.heap_index_to_edge_id[0].index] = 0;
	heap.edge_id_to_heap_index[edge_id.index] = u32_max;

	heap.edge_collapse_errors.pop_back();
	heap.heap_index_to_edge_id.pop_back();
	
	EdgeCollapseHeapSiftDown(heap, 0);
	
	return edge_id;
}

static void EdgeCollapseHeapRemove(EdgeCollapseHeap& heap, u32 heap_index) {
	auto edge_id = heap.heap_index_to_edge_id[heap_index];
	
	bool sift_up = true;
	if (heap.edge_collapse_errors.size() > heap_index) {
		float prev_error = heap.edge_collapse_errors[heap_index];

		heap.edge_collapse_errors[heap_index]  = heap.edge_collapse_errors.back();
		heap.heap_index_to_edge_id[heap_index] = heap.heap_index_to_edge_id.back();

		float new_error = heap.edge_collapse_errors[heap_index];
		sift_up = new_error < prev_error;
	
	}
	
	heap.edge_id_to_heap_index[heap.heap_index_to_edge_id[heap_index].index] = heap_index;
	heap.edge_id_to_heap_index[edge_id.index] = u32_max;

	heap.edge_collapse_errors.pop_back();
	heap.heap_index_to_edge_id.pop_back();
	
	if (sift_up) {
		EdgeCollapseHeapSiftUp(heap, heap_index);
	} else {
		EdgeCollapseHeapSiftDown(heap, heap_index);
	}
}

static void EdgeCollapseHeapUpdate(EdgeCollapseHeap& heap, u32 node_index, float error) {
	bool sift_up = error < heap.edge_collapse_errors[node_index];
	
	heap.edge_collapse_errors[node_index] = error;
	
	if (sift_up) {
		EdgeCollapseHeapSiftUp(heap, node_index);
	} else {
		EdgeCollapseHeapSiftDown(heap, node_index);
	}
}

static void EdgeCollapseHeapInitialize(EdgeCollapseHeap& heap) {
	if (heap.edge_collapse_errors.size() == 0) return;
	
	u32 node_index = HeapParentIndex((u32)heap.edge_collapse_errors.size() - 1);
	
	for (u32 i = node_index; i > 0; i -= 1) {
		EdgeCollapseHeapSiftDown(heap, i);
	}
	EdgeCollapseHeapSiftDown(heap, 0);
}

#define ENABLE_EDGE_COLLAPSE_HEAP_VALIDATION 0
#if ENABLE_EDGE_COLLAPSE_HEAP_VALIDATION
static void EdgeCollapseHeapValidate(EdgeCollapseHeap& heap) {
	for (u32 i = 0; i < heap.edge_collapse_errors.size(); i += 1) {
		float e0 = heap.edge_collapse_errors[i];
		float e1 = HeapChildIndex0(i) < heap.edge_collapse_errors.size() ? heap.edge_collapse_errors[HeapChildIndex0(i)] : FLT_MAX;
		float e2 = HeapChildIndex1(i) < heap.edge_collapse_errors.size() ? heap.edge_collapse_errors[HeapChildIndex1(i)] : FLT_MAX;

		assert(e0 <= e1);
		assert(e0 <= e2);

		assert(heap.edge_id_to_heap_index[heap.heap_index_to_edge_id[i].index] == i);
	}
}
#endif // ENABLE_EDGE_COLLAPSE_HEAP_VALIDATION

static void InitializeMehsDecimationState(MeshView mesh, MeshDecimationState& state) {
	state.vertex_edge_quadrics.resize(mesh.vertex_count);
	state.attribute_face_quadrics.resize(mesh.attribute_count);
	state.wedge_quadrics.reserve(64);
	state.wedge_attributes.reserve(64 * attribute_stride_dwords);
	state.wedge_attributes_ids.reserve(64 * attribute_stride_dwords);
	state.edge_duplicate_map.keys_values.resize(ComputeHashTableSize(128u));
	
	memset(state.vertex_edge_quadrics.data(), 0, state.vertex_edge_quadrics.size() * sizeof(Quadric));
	memset(state.attribute_face_quadrics.data(), 0, state.attribute_face_quadrics.size() * sizeof(QuadricWithAttributes));
	
	{
		// ScopedTimer t("- Compute Face Quadrics");
		
		for (FaceID face_id = { 0 }; face_id.index < mesh.face_count; face_id.index += 1) {
			auto& face = mesh[face_id];
			if (face.corner_list_base.index == u32_max) continue;
			
			auto& c1 = mesh[face.corner_list_base];
			auto& c0 = mesh[c1.corner_list_around[(u32)ElementType::Face].prev];
			auto& c2 = mesh[c1.corner_list_around[(u32)ElementType::Face].next];
			
			auto p0 = mesh[c0.vertex_id].position;
			auto p1 = mesh[c1.vertex_id].position;
			auto p2 = mesh[c2.vertex_id].position;
			
			auto* a0 = mesh[c0.attributes_id];
			auto* a1 = mesh[c1.attributes_id];
			auto* a2 = mesh[c2.attributes_id];
			
			QuadricWithAttributes quadric;
			ComputeFaceQuadricWithAttributes(quadric, p0, p1, p2, a0, a1, a2);
			
			AccumulateQuadricWithAttributes(state.attribute_face_quadrics[c0.attributes_id.index], quadric);
			AccumulateQuadricWithAttributes(state.attribute_face_quadrics[c1.attributes_id.index], quadric);
			AccumulateQuadricWithAttributes(state.attribute_face_quadrics[c2.attributes_id.index], quadric);
		}
	}
	
	{
		// ScopedTimer t("- Compute Edge Quadrics");
		
		for (EdgeID edge_id = { 0 }; edge_id.index < mesh.edge_count; edge_id.index += 1) {
			auto& edge = mesh[edge_id];
			if (edge.corner_list_base.index == u32_max) continue;
			
			auto& c0 = mesh[edge.corner_list_base];
			auto& c1 = mesh[c0.corner_list_around[(u32)ElementType::Face].next];
			auto& c2 = mesh[c0.corner_list_around[(u32)ElementType::Face].prev];
			
			auto attributes_id_0 = c0.attributes_id;
			auto attributes_id_1 = c1.attributes_id;
			
			u32 edge_degree = 0;
			bool attribute_edge = false;
			IterateCornerList<ElementType::Edge>(mesh, edge.corner_list_base, [&](CornerID corner_id) {
				auto attributes_id = mesh[corner_id].attributes_id;
				attribute_edge |= (attributes_id.index != attributes_id_0.index && attributes_id.index != attributes_id_1.index);
				
				edge_degree += 1;
			});
			if (edge_degree != 1 && attribute_edge == false) continue;
			
			
			// (v0, v1) is the current edge.
			auto v0 = c0.vertex_id;
			auto v1 = c1.vertex_id;
			auto v2 = c2.vertex_id;
			
			auto p0 = mesh[v0].position;
			auto p1 = mesh[v1].position;
			auto p2 = mesh[v2].position;
			
			Quadric quadric;
			ComputeEdgeQuadric(quadric, p0, p1, p2, 1.f / edge_degree);
			
			AccumulateQuadric(state.vertex_edge_quadrics[v0.index], quadric);
			AccumulateQuadric(state.vertex_edge_quadrics[v1.index], quadric);
		}
	}
}

#define REPORT_DECIMATION_PROGRESS 0

static void DecimateMeshFaceGroup(MeshView mesh, MeshDecimationState& state, EdgeCollapseHeap& edge_collapse_heap, u32 target_face_count, u32 active_face_count) {
	s32 next_report_target = active_face_count + 1;
	
	u64 time0 = 0;
	u64 time1 = 0;
	
	float max_error = 0.f;
	while (active_face_count > target_face_count && edge_collapse_heap.edge_collapse_errors.size()) {
		if ((s32)active_face_count < next_report_target) {
#if REPORT_DECIMATION_PROGRESS
			printf("Remaining Faces: %u\n", active_face_count - target_face_count);
			next_report_target -= 10000;
#endif // REPORT_DECIMATION_PROGRESS
			
#if ENABLE_EDGE_COLLAPSE_HEAP_VALIDATION
			EdgeCollapseHeapValidate(edge_collapse_heap);
#endif // ENABLE_EDGE_COLLAPSE_HEAP_VALIDATION
		}
		
		u64 t0 = __rdtsc();
		
		// ~80% of the execution time.
		for (auto& [edge_key, edge_id] : state.edge_duplicate_map.keys_values) {
			if (edge_key == u64_max) continue;
			
			u32 heap_index = edge_collapse_heap.edge_id_to_heap_index[edge_id.index];
			if (heap_index == u32_max) continue;
			
			EdgeSelectInfo select_info;
			ComputeEdgeCollapseError(mesh, state, edge_id, &select_info);
			
			EdgeCollapseHeapUpdate(edge_collapse_heap, heap_index, select_info.min_error);
		}
		HashTableClear(state.edge_duplicate_map);
		
		u64 t1 = __rdtsc();
		time0 += (t1 - t0);
		
		auto edge_id = EdgeCollapseHeapPop(edge_collapse_heap);
		
		// 2% of the execution time
		EdgeSelectInfo select_info;
		ComputeEdgeCollapseError(mesh, state, edge_id, &select_info);
		
		max_error = max_error < select_info.min_error ? select_info.min_error : max_error;
		
		// 15% of the execution time
		auto [remaning_base_id, removed_face_count] = PerformEdgeCollapse(mesh, edge_id, state.edge_duplicate_map, state.removed_edge_array);
		active_face_count -= removed_face_count;
		
		for (auto edge_id : state.removed_edge_array) {
			u32 heap_index = edge_collapse_heap.edge_id_to_heap_index[edge_id.index];
			if (heap_index != u32_max) EdgeCollapseHeapRemove(edge_collapse_heap, heap_index);
		}
		
		{
			auto& edge = mesh[edge_id];
			mesh[edge.vertex_0].position = select_info.new_position;
			mesh[edge.vertex_1].position = select_info.new_position;
			
			Quadric quadric = state.vertex_edge_quadrics[edge.vertex_0.index];
			AccumulateQuadric(quadric, state.vertex_edge_quadrics[edge.vertex_1.index]);
			
			state.vertex_edge_quadrics[edge.vertex_0.index] = quadric;
			state.vertex_edge_quadrics[edge.vertex_1.index] = quadric;
		
#if ENABLE_ATTRIBUTE_SUPPORT
			if (remaning_base_id.index != u32_max) {
				u32 wedge_count = (u32)state.wedge_quadrics.size();
				state.wedge_attributes.resize(wedge_count * attribute_stride_dwords);
				
				for (u32 i = 0; i < wedge_count; i += 1) {
					auto* attributes = &state.wedge_attributes[i * attribute_stride_dwords];
					auto& wedge_quadric = state.wedge_quadrics[i];
					auto attributes_id = state.wedge_attributes_ids[i];
					
					state.attribute_face_quadrics[attributes_id.index] = wedge_quadric;
					if (ComputeWedgeAttributes(wedge_quadric, select_info.new_position, attributes)) {
						// TODO: Provide a way to pick which attributes should be normalized.
						auto* normal = (Vector3*)(attributes + 2);
						*normal = Normalize(*normal);
					} else {
						// Zero weight wedge. In theory there is no difference which attributes we take.
						memcpy(attributes, mesh[attributes_id], sizeof(u32) * attribute_stride_dwords);
					}
				}
				
				IterateCornerList<ElementType::Vertex>(mesh, remaning_base_id, [&](CornerID corner_id) {
					auto attributes_id = mesh[corner_id].attributes_id;
					
					u8 index = AttributeWedgeMapFind(state.wedge_attribute_set, attributes_id);
					if (index != 0) {
						memcpy(mesh[attributes_id], &state.wedge_attributes[(index - 1) * attribute_stride_dwords], sizeof(u32) * attribute_stride_dwords);
						mesh[corner_id].attributes_id = state.wedge_attributes_ids[index - 1];
					}
				});
			}
#endif // ENABLE_ATTRIBUTE_SUPPORT
		}
		
		u64 t2 = __rdtsc();
		
		time1 += (t2 - t0);
	}
	
#if REPORT_DECIMATION_PROGRESS
	printf("%.3f\n", (float)time0 / (float)time1);
	printf("Max Error: %e\n", max_error);
	printf("Face Count: %u/%u\n", active_face_count, target_face_count);
#endif // REPORT_DECIMATION_PROGRESS
}

//
// TODO:
// - Optimization of vertex location.
// - Scale mesh and attributes before simplification.
// - Memory-less quadrics.
// - Output an obj sequence for edge to verify edge collapses.
//
void DecimateMesh(MeshView mesh) {
	MeshDecimationState state;
	InitializeMehsDecimationState(mesh, state);
	
	
	EdgeCollapseHeap edge_collapse_heap;
	edge_collapse_heap.edge_collapse_errors.resize(mesh.edge_count);
	edge_collapse_heap.edge_id_to_heap_index.resize(mesh.edge_count);
	edge_collapse_heap.heap_index_to_edge_id.resize(mesh.edge_count);
	
	{
		// ScopedTimer t("- Rank Edge Collapses");
		
		for (EdgeID edge_id = { 0 }; edge_id.index < mesh.edge_count; edge_id.index += 1) {
			EdgeSelectInfo select_info;
			ComputeEdgeCollapseError(mesh, state, edge_id, &select_info);
			
			edge_collapse_heap.edge_collapse_errors[edge_id.index]  = select_info.min_error;
			edge_collapse_heap.edge_id_to_heap_index[edge_id.index] = edge_id.index;
			edge_collapse_heap.heap_index_to_edge_id[edge_id.index] = edge_id;
		}
		
		EdgeCollapseHeapInitialize(edge_collapse_heap);
	}
	
	u32 target_face_count = mesh.face_count / 138;
	u32 active_face_count = mesh.face_count;
	DecimateMeshFaceGroup(mesh, state, edge_collapse_heap, target_face_count, active_face_count);
}

// TODO: Reduce the number of the allocations done here.
// TODO: Simplify the code (maybe the caller can compute the neccessary group face counts, group edges, etc.)
static void DecimateMeshFaceGroups(MeshView mesh, ArrayView<u32> face_id_to_group_index, u32 group_count) {
	MeshDecimationState state;
	InitializeMehsDecimationState(mesh, state);
	
	std::vector<u32> vertex_group_indices;
	vertex_group_indices.resize(mesh.vertex_count, u32_max);
	
	std::vector<u32> group_face_counts;
	group_face_counts.resize(group_count);
	
	compile_const u32 vertex_group_index_locked = u32_max - 1;
	
	{
		for (FaceID face_id = { 0 }; face_id.index < mesh.face_count; face_id.index += 1) {
			auto& face = mesh[face_id];
			if (face.corner_list_base.index == u32_max) continue;
			
			u32 group_index = face_id_to_group_index[face_id.index];
			group_face_counts[group_index] += 1;
			
			IterateCornerList<ElementType::Face>(mesh, face.corner_list_base, [&](CornerID corner_id) {
				auto& corner = mesh[corner_id];
				u32& index = vertex_group_indices[corner.vertex_id.index];
				
				if (index == u32_max) {
					index = group_index;
				} else if (index != group_index) {
					index = vertex_group_index_locked; // Lock the vertex.
				}
			});
		}
	}
	
	std::vector<EdgeID> group_edge_ids;
	std::vector<u32> group_edge_counts;
	{
		group_edge_counts.resize(group_count);
		
		for (EdgeID edge_id = { 0 }; edge_id.index < mesh.edge_count; edge_id.index += 1) {
			auto& edge = mesh[edge_id];
			if (edge.corner_list_base.index == u32_max) continue;
			
			u32 group_index_0 = vertex_group_indices[edge.vertex_0.index];
			u32 group_index_1 = vertex_group_indices[edge.vertex_1.index];
			
			// TODO: Allow edge collapses when only one vertex is locked.
			bool edge_is_locked = (group_index_0 == vertex_group_index_locked) || (group_index_1 == vertex_group_index_locked);
			assert(edge_is_locked || group_index_0 == group_index_1);
			
			if (edge_is_locked == false) {
				group_edge_counts[group_index_0] += 1;
			}
		}
		
		u32 prefix_sum = 0;
		for (u32 i = 0; i < group_count; i += 1) {
			u32 count = group_edge_counts[i];
			group_edge_counts[i] = prefix_sum;
			prefix_sum += count;
		}
		
		group_edge_ids.resize(prefix_sum);
		for (EdgeID edge_id = { 0 }; edge_id.index < mesh.edge_count; edge_id.index += 1) {
			auto& edge = mesh[edge_id];
			if (edge.corner_list_base.index == u32_max) continue;
			
			u32 group_index_0 = vertex_group_indices[edge.vertex_0.index];
			u32 group_index_1 = vertex_group_indices[edge.vertex_1.index];
			
			// TODO: Allow edge collapses when only one vertex is locked.
			bool edge_is_locked = (group_index_0 == vertex_group_index_locked) || (group_index_1 == vertex_group_index_locked);
			assert(edge_is_locked || group_index_0 == group_index_1);
			
			if (edge_is_locked == false) {
				group_edge_ids[group_edge_counts[group_index_0]++] = edge_id;
			}
		}
	}
	
	EdgeCollapseHeap edge_collapse_heap;
	for (u32 i = 0; i < group_count; i += 1) {
		u32 edge_begin_index = i > 0 ? group_edge_counts[i - 1] : 0;
		u32 edge_end_index   = group_edge_counts[i];
		
		u32 face_count = group_face_counts[i];
		u32 edge_count = edge_end_index - edge_begin_index;
		
		edge_collapse_heap.edge_collapse_errors.resize(edge_count);
		edge_collapse_heap.edge_id_to_heap_index.resize(mesh.edge_count);
		edge_collapse_heap.heap_index_to_edge_id.resize(edge_count);
		
		memset(edge_collapse_heap.edge_id_to_heap_index.data(), 0xFF, edge_collapse_heap.edge_id_to_heap_index.size() * sizeof(u32));
		
		{
			// ScopedTimer t("- Rank Edge Collapses");
			
			for (u32 edge_index = edge_begin_index; edge_index < edge_end_index; edge_index += 1) {
				auto edge_id = group_edge_ids[edge_index];
				u32 local_edge_index = edge_index - edge_begin_index;
				
				EdgeSelectInfo select_info;
				ComputeEdgeCollapseError(mesh, state, edge_id, &select_info);
				
				edge_collapse_heap.edge_collapse_errors[local_edge_index]  = select_info.min_error;
				edge_collapse_heap.edge_id_to_heap_index[edge_id.index]    = local_edge_index;
				edge_collapse_heap.heap_index_to_edge_id[local_edge_index] = edge_id;
			}
			
			EdgeCollapseHeapInitialize(edge_collapse_heap);
		}
		
		u32 target_face_count = face_count / 2;
		u32 active_face_count = face_count;
		DecimateMeshFaceGroup(mesh, state, edge_collapse_heap, target_face_count, active_face_count);
		
		HashTableClear(state.edge_duplicate_map);
		state.removed_edge_array.clear();
	}
}


struct alignas(16) KdTreeElement {
	Vector3 position;
	u32 partition_index;
};
static_assert(sizeof(KdTreeElement) == 16);

struct KdTreeNode {
	union {
		float split;
		u32 index = 0;
	};
	
	u32 axis    : 2;
	u32 payload : 30; // Note that zero payload means that branch is pruned. This reduces the number of node visits by 5x.
	
	compile_const u32 leaf_axis = 3;
	compile_const u32 leaf_size = 64;
};

struct KdTree {
	Array<KdTreeElement> elements;
	Array<u32> element_indices;
	Array<KdTreeNode> nodes;
};

static u32 KdTreeSplit(const ArrayView<KdTreeElement>& elements, ArrayView<u32> indices, KdTreeNode& node) {
	Vector3 sum = { 0.f, 0.f, 0.f };
	Vector3 min = { +FLT_MAX, +FLT_MAX, +FLT_MAX };
	Vector3 max = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
	float indices_count = 0.f;
	
	for (u32 i = 0; i < indices.count; i += 1) {
		auto& element = elements[indices[i]];
		if (element.partition_index != u32_max) continue;
		
		for (u32 j = 0; j < 3; j += 1) {
			min[j] = min[j] < element.position[j] ? min[j] : element.position[j];
			max[j] = max[j] > element.position[j] ? max[j] : element.position[j];
		}
		sum = sum + element.position;
		indices_count += 1.f;
	}
	
	auto mean = (indices_count > 0.f) ? sum * (1.f / indices_count) : sum;
	auto extent = max - min;
	
	u32 split_axis = 0;
	for (u32 axis = 1; axis < 3; axis += 1) {
		if (extent[axis] > extent[split_axis]) split_axis = axis;
	}
	float split_position = mean[split_axis];
	
	
	u32 split_index = 0;
	for (u32 i = 0; i < indices.count; i += 1) {
		float position = elements[indices[i]].position[split_axis];
		std::swap(indices[split_index], indices[i]);
		if (position < split_position) split_index += 1;
	}
	
	node.split = split_position;
	node.axis  = split_axis;
	
	return split_index;
}

static u32 KdTreeBuildLeafNode(Array<KdTreeNode>& nodes, ArrayView<KdTreeElement> elements, const ArrayView<u32>& indices) {
	u32 leaf_node_index = nodes.count;
	
	u32 indices_count = indices.count;
	for (u32 i = 0; i < indices_count; i += 1) {
		KdTreeNode node;
		node.index   = indices[i];
		node.axis    = KdTreeNode::leaf_axis;
		node.payload = indices_count;
		ArrayAppend(nodes, node);
	}
	
	return leaf_node_index;
}

static u32 KdTreeBuildNode(Array<KdTreeNode>& nodes, ArrayView<KdTreeElement> elements, ArrayView<u32> indices) {
	if (indices.count <= KdTreeNode::leaf_size) {
		return KdTreeBuildLeafNode(nodes, elements, indices);
	}
	
	KdTreeNode node;
	u32 split_index = KdTreeSplit(elements, indices, node);
	
	// Faild to split the subtree. Create a single leaf to prevent infinite recursion.
	if (split_index == 0 || split_index >= indices.count) {
		return KdTreeBuildLeafNode(nodes, elements, indices);
	}
	
	u32 node_index = nodes.count;
	ArrayAppend(nodes, node);
	
	u32 node_index_0 = KdTreeBuildNode(nodes, elements, CreateArrayView(indices, 0, split_index));
	u32 node_index_1 = KdTreeBuildNode(nodes, elements, CreateArrayView(indices, split_index, indices.count));
	
	assert(node_index_0 == node_index + 1); // Left node is always the next node after the local root.
	assert(node_index_1 > node_index);      // Right node offset is non zero. Zero means branch is pruned.
	
	nodes[node_index].payload = node_index_1 - node_index;
	
	return node_index;
}

static void KdTreeReserve(KdTree& tree, Allocator& allocator) {
	ArrayResize(tree.element_indices, allocator, tree.elements.count);
	ArrayReserve(tree.nodes, allocator, tree.elements.count * 2);
}

static void KdTreeBuild(KdTree& tree, u32 begin_element_index, u32 end_element_index) {
	for (u32 i = 0; i < tree.element_indices.count; i += 1) {
		tree.element_indices[i] = i;
	}
	
	KdTreeBuildNode(tree.nodes, CreateArrayView(tree.elements, begin_element_index, end_element_index), CreateArrayView(tree.element_indices));
}

#define COUNT_KD_TREE_LOOKUPS 0
#define COUNT_KD_TREE_NODE_VISITS 0

#if COUNT_KD_TREE_NODE_VISITS
u32 kd_tree_node_visits = 0;
#endif // COUNT_KD_TREE_NODE_VISITS

static bool KdTreeFindClosestActiveElement(KdTree& kd_tree, const Vector3& point, u32& closest_index, float& min_distance, u32 root = 0) {
	auto& node = kd_tree.nodes[root];
	
#if COUNT_KD_TREE_NODE_VISITS
	kd_tree_node_visits += 1;
#endif // COUNT_KD_TREE_NODE_VISITS
	
	// Prune empty branches of the tree for better traversal performance.
	bool should_prune = true;
	
	if (node.axis == KdTreeNode::leaf_axis) {
		u32 child_count = node.payload;
		
		for (u32 i = 0; i < child_count; i += 1) {
#if COUNT_KD_TREE_NODE_VISITS
			kd_tree_node_visits += 1;
#endif // COUNT_KD_TREE_NODE_VISITS
			
			u32 index = kd_tree.nodes[root + i].index;
			auto& element = kd_tree.elements[index];
			
			if (element.partition_index != u32_max) continue; // Element is already used, i.e. it's inactive for the sake of search.
			should_prune = false; // Don't prune the branch if we have at least one active leaf element.
			
			auto point_to_element = element.position - point;
			
			auto distance = DotProduct(point_to_element, point_to_element);
			if (distance < min_distance) {
				min_distance = distance;
				closest_index = index;
			}
		}
	} else if (node.payload != 0) {
		// Visit the closest node first.
		float delta  = point[node.axis] - node.split;
		u32 offset_0 = delta <= 0.f ? 1 : node.payload; // Left node is always the next node after the local root.
		u32 offset_1 = delta <= 0.f ? node.payload : 1; // Right node offset is non zero. Zero means branch is pruned.
		
		bool prune_lh = KdTreeFindClosestActiveElement(kd_tree, point, closest_index, min_distance, root + offset_0);
		bool prune_rh = false;
		
		if ((delta * delta) <= min_distance) {
			prune_rh = KdTreeFindClosestActiveElement(kd_tree, point, closest_index, min_distance, root + offset_1);
		} else {
			prune_rh = (kd_tree.nodes[root + offset_1].payload == 0);
		}
		should_prune = prune_lh && prune_rh;
	}
	
	if (should_prune) node.payload = 0;
	
	return should_prune;
}

compile_const u32 meshlet_max_vertex_count = 128;
compile_const u32 meshlet_max_face_count   = 128;
compile_const u32 meshlet_max_face_degree  = 3;
compile_const u32 meshlet_group_max_meshlets = 32;

struct alignas(16) Meshlet {
	Vector3 aabb_min;
	u32 current_group_index = 0;
	Vector3 aabb_max;
	u32 coarser_group_index = 0;
};

static void KdTreeBuildElementsForFaces(MeshView mesh, Allocator& allocator, Array<KdTreeElement>& elements) {
	ArrayResize(elements, allocator, mesh.face_count);
	
	for (FaceID face_id = { 0 }; face_id.index < mesh.face_count; face_id.index += 1) {
		auto& element = elements[face_id.index];
		auto& face    = mesh[face_id];
		
		if (face.corner_list_base.index != u32_max) {
			Vector3 position = { 0.f, 0.f, 0.f };
			float face_degree = 0.f;
			IterateCornerList<ElementType::Face>(mesh, face.corner_list_base, [&](CornerID corner_id) {
				position = position + mesh[mesh[corner_id].vertex_id].position;
				face_degree += 1.f;
			});
			
			element.position = position * (1.f / face_degree);
			element.partition_index = u32_max - 1; // Face elements are inactive by default. We mark them as active per group when generating meshlets.
		} else {
			element.partition_index = u32_max - 1;
		}
	}
}

static void KdTreeBuildElementsForMeshlets(ArrayView<Meshlet> meshlets, Allocator& allocator, Array<KdTreeElement>& elements) {
	ArrayResize(elements, allocator, meshlets.count);
	
	for (u32 meshlet_index = 0; meshlet_index < meshlets.count; meshlet_index += 1) {
		auto& element = elements[meshlet_index];
		auto& meshlet = meshlets[meshlet_index];
		
		auto aabb_min_ps = _mm_load_ps(&meshlet.aabb_min.x);
		auto aabb_max_ps = _mm_load_ps(&meshlet.aabb_max.x);
		auto center_ps   = _mm_mul_ps(_mm_add_ps(aabb_min_ps, aabb_max_ps), _mm_set_ps1(0.5f));
		
		_mm_store_ps(&element.position.x, center_ps);
		element.partition_index = u32_max;
	}
}


struct MeshletAdjacencyInfo {
	u32 meshlet_index     = 0;
	u32 shared_edge_count = 0;
};

struct MeshletAdjacency {
	ArrayView<u32> prefix_sum;
	ArrayView<MeshletAdjacencyInfo> infos;
};

struct MeshletBuildResult {
	ArrayView<FaceID>  meshlet_faces;
	ArrayView<u32>     meshlet_face_prefix_sum;
	ArrayView<Meshlet> meshlets;
	
	MeshletAdjacency meshlet_adjacency;
};

static MeshletAdjacency BuildMeshletAdjacency(MeshView mesh, Allocator& allocator, ArrayView<u32> meshlet_face_prefix_sum, ArrayView<FaceID> meshlet_faces, ArrayView<KdTreeElement> kd_tree_elements);

static void BuildMeshletsForGroup(
	MeshView mesh,
	KdTree kd_tree,
	Array<u8> vertex_usage_map,
	Array<FaceID>& meshlet_faces,
	Array<u32>& meshlet_face_prefix_sum,
	Array<Meshlet>& meshlets) {
	
	compile_const u32 candidates_per_face = 4;
	FixedSizeArray<AttributesID, meshlet_max_face_count * candidates_per_face> meshlet_vertices;
	FixedSizeArray<FaceID, meshlet_max_vertex_count + meshlet_max_face_degree> meshlet_candidate_elements;
	
	auto meshlet_aabb_min_ps  = _mm_set_ps1(+FLT_MAX);
	auto meshlet_aabb_max_ps  = _mm_set_ps1(-FLT_MAX);
	u32  meshlet_vertex_count = 0;
	u32  meshlet_face_count   = 0;
	
#if COUNT_KD_TREE_LOOKUPS
	u32 kd_tree_lookup_count = 0;
#endif // COUNT_KD_TREE_LOOKUPS
	
	while (true) {
		u32 best_candidate_face_index = u32_max;
		u32 second_best_candidate_face_index  = u32_max;
		float smallest_surface_area = FLT_MAX;
		
		for (u32 i = 0; i < meshlet_candidate_elements.count;) {
			auto face_id = meshlet_candidate_elements[i];
			
			auto& element = kd_tree.elements[face_id.index];
			if (element.partition_index != u32_max) {
				ArrayEraseSwap(meshlet_candidate_elements, i);
				continue;
			}
			auto position = _mm_load_ps(&element.position.x);
			
			auto new_aabb_min_ps = _mm_min_ps(meshlet_aabb_min_ps, position);
			auto new_aabb_max_ps = _mm_max_ps(meshlet_aabb_max_ps, position);
			auto new_aabb_extent_ps = _mm_sub_ps(new_aabb_max_ps, new_aabb_min_ps);
			
			// Half AABB surface area x*y + y*z + z*x
			float surface_area = _mm_cvtss_f32(_mm_dp_ps(new_aabb_extent_ps, _mm_permute_ps(new_aabb_extent_ps, 0b00'10'01), 0x7F));
			
			if (smallest_surface_area > surface_area) {
				smallest_surface_area = surface_area;
				second_best_candidate_face_index = best_candidate_face_index;
				best_candidate_face_index = i;
			}
			
			i += 1;
		}
		
		auto best_face_id        = best_candidate_face_index        != u32_max ? meshlet_candidate_elements[best_candidate_face_index]        : FaceID{ u32_max };
		auto second_best_face_id = second_best_candidate_face_index != u32_max ? meshlet_candidate_elements[second_best_candidate_face_index] : FaceID{ u32_max };
		
		if (best_candidate_face_index != u32_max) {
			ArrayEraseSwap(meshlet_candidate_elements, best_candidate_face_index);
		}
		
		if (best_face_id.index == u32_max) {
			alignas(16) float center[4];
			if (meshlet_face_count) {
				auto center_ps = _mm_mul_ps(_mm_add_ps(meshlet_aabb_max_ps, meshlet_aabb_min_ps), _mm_set_ps1(0.5f));
				_mm_store_ps(center, center_ps);
			} else {
				_mm_store_ps(center, _mm_setzero_ps());
			}
			
			float min_distance = FLT_MAX;
			KdTreeFindClosestActiveElement(kd_tree, *(Vector3*)center, best_face_id.index, min_distance);
			
#if COUNT_KD_TREE_LOOKUPS
			kd_tree_lookup_count += 1;
#endif // COUNT_KD_TREE_LOOKUPS
		}
		
		if (best_face_id.index == u32_max) {
			break;
		}
		
		
		u32 new_vertex_count = 0;
		IterateCornerList<ElementType::Face>(mesh, mesh[best_face_id].corner_list_base, [&](CornerID corner_id) {
			auto& corner = mesh[corner_id];
			u8 vertex_index = vertex_usage_map[corner.attributes_id.index];
			if (vertex_index == 0xFF) {
				vertex_usage_map[corner.attributes_id.index] = meshlet_vertex_count + new_vertex_count;
				ArrayAppend(meshlet_vertices, corner.attributes_id);
				new_vertex_count += 1;
			}
			
			IterateCornerList<ElementType::Edge>(mesh, corner_id, [&](CornerID corner_id) {
				auto face_id = mesh[corner_id].face_id;
				
				if (kd_tree.elements[face_id.index].partition_index == u32_max && meshlet_candidate_elements.count < meshlet_candidate_elements.capacity) {
					ArrayAppend(meshlet_candidate_elements, face_id);
				}
			});
		});
		
		
		if ((meshlet_vertex_count + new_vertex_count > meshlet_max_vertex_count) || (meshlet_face_count + 1 > meshlet_max_face_count)) {
			// TODO: Create AABB over vertices.
			Meshlet meshlet;
			_mm_store_ps(&meshlet.aabb_min.x, meshlet_aabb_min_ps);
			_mm_store_ps(&meshlet.aabb_max.x, meshlet_aabb_max_ps);
			
			meshlet_vertex_count = 0;
			meshlet_face_count   = 0;
			meshlet_aabb_min_ps  = _mm_set_ps1(+FLT_MAX);
			meshlet_aabb_max_ps  = _mm_set_ps1(-FLT_MAX);
			
			for (auto attributes_id : meshlet_vertices) {
				vertex_usage_map[attributes_id.index] = 0xFF;
			}
			
			ArrayAppend(meshlet_face_prefix_sum, meshlet_faces.count);
			ArrayAppend(meshlets, meshlet);
			
			meshlet_vertices.count = 0;
			meshlet_candidate_elements.count = 0;
			
			if (second_best_face_id.index != u32_max) {
				ArrayAppend(meshlet_candidate_elements, second_best_face_id);
			}
		}
		
		ArrayAppend(meshlet_faces, best_face_id);
		meshlet_vertex_count += new_vertex_count;
		meshlet_face_count   += 1;
		
		auto& element = kd_tree.elements[best_face_id.index];
		element.partition_index = meshlet_face_prefix_sum.count;
		
		auto position = _mm_load_ps(&element.position.x);
		meshlet_aabb_min_ps = _mm_min_ps(meshlet_aabb_min_ps, position);
		meshlet_aabb_max_ps = _mm_max_ps(meshlet_aabb_max_ps, position);
	}
	
	if (meshlet_face_count) {
		// TODO: Create AABB over vertices.
		Meshlet meshlet;
		_mm_store_ps(&meshlet.aabb_min.x, meshlet_aabb_min_ps);
		_mm_store_ps(&meshlet.aabb_max.x, meshlet_aabb_max_ps);

		for (auto attributes_id : meshlet_vertices) {
			vertex_usage_map[attributes_id.index] = 0xFF;
		}
		
		meshlet_face_count = 0;
		ArrayAppend(meshlet_face_prefix_sum, meshlet_faces.count);
		ArrayAppend(meshlets, meshlet);
	}
}
	

static MeshletBuildResult BuildMeshlets(MeshView mesh, Allocator& allocator, ArrayView<u32> face_id_to_group_index, u32 group_count) {
	KdTree kd_tree;
	KdTreeBuildElementsForFaces(mesh, allocator, kd_tree.elements);
	KdTreeReserve(kd_tree, allocator);
	
	Array<u8> vertex_usage_map;
	ArrayResizeMemset(vertex_usage_map, allocator, mesh.attribute_count, 0xFF);
	
	// TODO: Use an irregular 2D array.
	Array<FaceID> meshlet_faces;
	Array<u32> meshlet_face_prefix_sum;
	Array<Meshlet> meshlets;
	Array<u32> group_face_prefix_sum;
	
	ArrayReserve(meshlet_faces, allocator, mesh.face_count);
	ArrayReserve(meshlet_face_prefix_sum, allocator, mesh.face_count);
	ArrayReserve(meshlets, allocator, mesh.face_count);
	ArrayResizeMemset(group_face_prefix_sum, allocator, group_count, 0);
	
	{
		for (u32 i = 0; i < face_id_to_group_index.count; i += 1) {
			auto face = mesh.faces[i];
			if (face.corner_list_base.index == u32_max) continue;
			
			u32 group_index = face_id_to_group_index[i];
			group_face_prefix_sum[group_index] += 1;
		}
		
		u32 prefix_sum = 0;
		for (u32 i = 0; i < group_count; i += 1) {
			u32 count = group_face_prefix_sum[i];
			group_face_prefix_sum[i] = prefix_sum;
			prefix_sum += count;
		}
		
		for (u32 i = 0; i < face_id_to_group_index.count; i += 1) {
			auto face = mesh.faces[i];
			if (face.corner_list_base.index == u32_max) continue;
			
			u32 group_index = face_id_to_group_index[i];
			u32 element_index = group_face_prefix_sum[group_index]++;
			
			kd_tree.element_indices[element_index] = i;
		}
	}
	
	u32 begin_element_index = 0;
	for (u32 i = 0; i < group_count; i += 1) {
		u32 end_element_index = group_face_prefix_sum[i];
		
		kd_tree.nodes.count = 0;
		auto element_indices = CreateArrayView(kd_tree.element_indices, begin_element_index, end_element_index);
		auto elements        = CreateArrayView(kd_tree.elements);
		
		for (u32 index : element_indices) {
			// Mark elements of the current group as eligible for meshlet generation.
			elements[index].partition_index = u32_max;
		}
		
		KdTreeBuildNode(kd_tree.nodes, elements, element_indices);
	
		BuildMeshletsForGroup(
			mesh,
			kd_tree,
			vertex_usage_map,
			meshlet_faces,
			meshlet_face_prefix_sum,
			meshlets
		);
		
		assert(meshlet_faces.count == end_element_index);
		
		begin_element_index = end_element_index;
	}
	
	assert(meshlet_faces.count == mesh.face_count);
	
	MeshletBuildResult result;
	result.meshlet_faces           = CreateArrayView(meshlet_faces);
	result.meshlet_face_prefix_sum = CreateArrayView(meshlet_face_prefix_sum);
	result.meshlets                = CreateArrayView(meshlets);
	result.meshlet_adjacency       = BuildMeshletAdjacency(mesh, allocator, result.meshlet_face_prefix_sum, result.meshlet_faces, CreateArrayView(kd_tree.elements));
	
#if COUNT_KD_TREE_NODE_VISITS
	printf("BuildMeshlets: kd_tree_node_visits: %u\n", kd_tree_node_visits);
#endif // COUNT_KD_TREE_NODE_VISITS
	
#if COUNT_KD_TREE_LOOKUPS
	printf("BuildMeshlets: kd_tree_lookup_count: %u\n", kd_tree_lookup_count);
#endif // COUNT_KD_TREE_LOOKUPS
	
	return result;
}

static MeshletAdjacency BuildMeshletAdjacency(MeshView mesh, Allocator& allocator, ArrayView<u32> meshlet_face_prefix_sum, ArrayView<FaceID> meshlet_faces, ArrayView<KdTreeElement> kd_tree_elements) {
	Array<u32> meshlet_adjacency_info_indices;
	ArrayResizeMemset(meshlet_adjacency_info_indices, allocator, meshlet_face_prefix_sum.count, 0xFF);
	
	Array<u32> meshlet_adjacency_prefix_sum;
	Array<MeshletAdjacencyInfo> meshlet_adjacency_infos;
	ArrayReserve(meshlet_adjacency_prefix_sum, allocator, meshlet_face_prefix_sum.count);
	// Must be the last allocation on the memory block stack as it might grow.
	ArrayReserve(meshlet_adjacency_infos, allocator, meshlet_face_prefix_sum.count * 8);
	
	u32 face_begin_index = 0;
	for (u32 meshlet_index = 0; meshlet_index < meshlet_face_prefix_sum.count; meshlet_index += 1) {
		u32 face_end_index = meshlet_face_prefix_sum[meshlet_index];
		
		// At least reserve one meshlet per face edge. Do this upfront instead of adding code in the inner loop to improve performance.
		compile_const u32 reserve_size = meshlet_max_face_count * meshlet_max_face_degree;
		if (meshlet_adjacency_infos.count + reserve_size >= meshlet_adjacency_infos.capacity) {
			ArrayReserve(meshlet_adjacency_infos, allocator, ArrayComputeNewCapacity(meshlet_adjacency_infos.capacity, meshlet_adjacency_infos.capacity + reserve_size));
		}
		
		u32 begin_adjacency_info_index = meshlet_adjacency_infos.count;
		for (u32 face_index = face_begin_index; face_index < face_end_index; face_index += 1) {
			auto face_id = meshlet_faces[face_index];
			
			IterateCornerList<ElementType::Face>(mesh, mesh[face_id].corner_list_base, [&](CornerID corner_id) {
				IterateCornerList<ElementType::Edge>(mesh, corner_id, [&](CornerID corner_id) {
					auto other_face_id = mesh[corner_id].face_id;
					u32 other_meshlet_index = kd_tree_elements[other_face_id.index].partition_index;
					if (other_meshlet_index == meshlet_index) return;
					
					assert(other_meshlet_index != u32_max); // Face isn't a part of any meshlet.
					
					u32 adjacency_info_index = meshlet_adjacency_info_indices[other_meshlet_index];
					if (adjacency_info_index == u32_max) {
						adjacency_info_index = meshlet_adjacency_infos.count;
						
						// Enough memory should be reserved upfront. Reserving directly in this loop
						// slows down adjacency search by 40% even if we never need to grow the array.
						if (meshlet_adjacency_infos.count >= meshlet_adjacency_infos.capacity) return;
						
						meshlet_adjacency_info_indices[other_meshlet_index] = meshlet_adjacency_infos.count;
						
						MeshletAdjacencyInfo info;
						info.meshlet_index     = other_meshlet_index;
						info.shared_edge_count = 1;
						ArrayAppend(meshlet_adjacency_infos, info);
					} else {
						meshlet_adjacency_infos[adjacency_info_index].shared_edge_count += 1;
					}
				});
			});
		}
		u32 end_adjacency_info_index = meshlet_adjacency_infos.count;
		
		ArrayAppend(meshlet_adjacency_prefix_sum, end_adjacency_info_index);
		
		for (u32 adjacency_info_index = begin_adjacency_info_index; adjacency_info_index < end_adjacency_info_index; adjacency_info_index += 1) {
			meshlet_adjacency_info_indices[meshlet_adjacency_infos[adjacency_info_index].meshlet_index] = u32_max;
		}
		
		face_begin_index = face_end_index;
	}
	
	MeshletAdjacency meshlet_adjacency;
	meshlet_adjacency.prefix_sum = CreateArrayView(meshlet_adjacency_prefix_sum);
	meshlet_adjacency.infos      = CreateArrayView(meshlet_adjacency_infos);
	
	return meshlet_adjacency;
}

static u32 CountMeshletGroupSharedEdges(MeshletAdjacency meshlet_adjacency, Array<KdTreeElement> kd_tree_elements, u32 meshlet_index, u32 targent_group_index) {
	u32 meshlet_begin_index = meshlet_index > 0 ? meshlet_adjacency.prefix_sum[meshlet_index - 1] : 0;
	u32 meshlet_end_index   = meshlet_adjacency.prefix_sum[meshlet_index];
	
	u32 shared_edge_count = 0;
	for (u32 adjacency_info_index = meshlet_begin_index; adjacency_info_index < meshlet_end_index; adjacency_info_index += 1) {
		auto adjacency_info = meshlet_adjacency.infos[adjacency_info_index];
		
		auto& element = kd_tree_elements[adjacency_info.meshlet_index];
		if (element.partition_index == targent_group_index) {
			shared_edge_count += adjacency_info.shared_edge_count;
		}
	}
	
	return shared_edge_count;
}

struct MeshletGroupBuildResult {
	ArrayView<u32> meshlet_indices;
	ArrayView<u32> prefix_sum;
};

static MeshletGroupBuildResult BuildMeshletGroups(MeshView mesh, Allocator& allocator, ArrayView<Meshlet> meshlets, MeshletAdjacency meshlet_adjacency) {
	KdTree kd_tree;
	KdTreeBuildElementsForMeshlets(meshlets, allocator, kd_tree.elements);
	KdTreeReserve(kd_tree, allocator);
	KdTreeBuild(kd_tree, 0, kd_tree.elements.count);
	
	FixedSizeArray<u32, meshlet_group_max_meshlets> meshlet_group;
	
	Array<u32> meshlet_group_meshlet_indices;
	Array<u32> meshlet_group_prefix_sum;
	ArrayReserve(meshlet_group_meshlet_indices, allocator, meshlets.count);
	ArrayReserve(meshlet_group_prefix_sum, allocator, (meshlets.count + meshlet_group_max_meshlets - 1) / meshlet_group_max_meshlets);
	
	auto meshlet_group_aabb_min_ps = _mm_set_ps1(+FLT_MAX);
	auto meshlet_group_aabb_max_ps = _mm_set_ps1(-FLT_MAX);
	
#if COUNT_KD_TREE_LOOKUPS
	u32 kd_tree_lookup_count = 0;
#endif // COUNT_KD_TREE_LOOKUPS
	
	while (true) {
		u32 best_candidate_meshlet_index = u32_max;
		u32 max_shared_edge_count = 0;
		
		for (u32 i = 0; i < meshlet_group.count; i += 1) {
			u32 meshlet_index = meshlet_group[i];
			
			u32 meshlet_begin_index = meshlet_index > 0 ? meshlet_adjacency.prefix_sum[meshlet_index - 1] : 0;
			u32 meshlet_end_index   = meshlet_adjacency.prefix_sum[meshlet_index];
			
			for (u32 adjacency_info_index = meshlet_begin_index; adjacency_info_index < meshlet_end_index; adjacency_info_index += 1) {
				auto adjacency_info = meshlet_adjacency.infos[adjacency_info_index];
				
				auto& element = kd_tree.elements[adjacency_info.meshlet_index];
				if (element.partition_index != u32_max) continue; // Meshlet is already assigned to a group.
				
				u32 shared_edge_count = CountMeshletGroupSharedEdges(meshlet_adjacency, kd_tree.elements, adjacency_info.meshlet_index, meshlet_group_prefix_sum.count);
				
				if (max_shared_edge_count < shared_edge_count) {
					max_shared_edge_count = shared_edge_count;
					best_candidate_meshlet_index = adjacency_info.meshlet_index;
				}
			}
		}
		
		if (best_candidate_meshlet_index == u32_max) {
			alignas(16) float center[4];
			if (meshlet_group.count) {
				auto center_ps = _mm_mul_ps(_mm_add_ps(meshlet_group_aabb_max_ps, meshlet_group_aabb_min_ps), _mm_set_ps1(0.5f));
				_mm_store_ps(center, center_ps);
			} else {
				_mm_store_ps(center, _mm_setzero_ps());
			}
			
			float min_distance = FLT_MAX;
			KdTreeFindClosestActiveElement(kd_tree, *(Vector3*)center, best_candidate_meshlet_index, min_distance);
			
#if COUNT_KD_TREE_LOOKUPS
			kd_tree_lookup_count += 1;
#endif // COUNT_KD_TREE_LOOKUPS
		}
		
		if (best_candidate_meshlet_index == u32_max) {
			break;
		}
		
		if (meshlet_group.count >= meshlet_group.capacity) {
			meshlet_group_aabb_min_ps = _mm_set_ps1(+FLT_MAX);
			meshlet_group_aabb_max_ps = _mm_set_ps1(-FLT_MAX);
			
			ArrayAppend(meshlet_group_prefix_sum, meshlet_group_meshlet_indices.count);
			meshlet_group.count = 0;
		}
		
		ArrayAppend(meshlet_group_meshlet_indices, best_candidate_meshlet_index);
		ArrayAppend(meshlet_group, best_candidate_meshlet_index);
		
		auto& element = kd_tree.elements[best_candidate_meshlet_index];
		element.partition_index = meshlet_group_prefix_sum.count;
		
		auto position = _mm_load_ps(&element.position.x);
		meshlet_group_aabb_min_ps = _mm_min_ps(meshlet_group_aabb_min_ps, position);
		meshlet_group_aabb_max_ps = _mm_max_ps(meshlet_group_aabb_max_ps, position);
	}
	
	if (meshlet_group.count) {
		meshlet_group.count = 0;
		ArrayAppend(meshlet_group_prefix_sum, meshlet_group_meshlet_indices.count);
	}
	
	assert(meshlet_group_meshlet_indices.count == meshlets.count);
	
	MeshletGroupBuildResult result;
	result.meshlet_indices = CreateArrayView(meshlet_group_meshlet_indices);
	result.prefix_sum      = CreateArrayView(meshlet_group_prefix_sum);
	
#if COUNT_KD_TREE_NODE_VISITS
	printf("BuildMeshletGroups: kd_tree_node_visits: %u\n", kd_tree_node_visits);
#endif // COUNT_KD_TREE_NODE_VISITS
	
#if COUNT_KD_TREE_LOOKUPS
	printf("BuildMeshletGroups: kd_tree_lookup_count: %u\n", kd_tree_lookup_count);
#endif // COUNT_KD_TREE_LOOKUPS
	
	return result;
}

static void BuildFaceIdToGroupIndexMap(MeshView mesh, Allocator& allocator, MeshletBuildResult meshlet_build_result, MeshletGroupBuildResult meshlet_group_build_result, ArrayView<u32> face_id_to_group_index) {
	assert(face_id_to_group_index.count >= mesh.face_count);
	
	u32 group_meshlet_begin_index = 0;
	for (u32 group_index = 0; group_index < meshlet_group_build_result.prefix_sum.count; group_index += 1) {
		u32 group_meshlet_end_index = meshlet_group_build_result.prefix_sum[group_index];
		
		for (u32 group_meshlet_index = group_meshlet_begin_index; group_meshlet_index < group_meshlet_end_index; group_meshlet_index += 1) {
			u32 meshlet_index = meshlet_group_build_result.meshlet_indices[group_meshlet_index];
			
			u32 face_begin_index = meshlet_index > 0 ? meshlet_build_result.meshlet_face_prefix_sum[meshlet_index - 1] : 0;
			u32 face_end_index   = meshlet_build_result.meshlet_face_prefix_sum[meshlet_index];
			for (u32 face_index = face_begin_index; face_index < face_end_index; face_index += 1) {
				auto face_id = meshlet_build_result.meshlet_faces[face_index];
				face_id_to_group_index[face_id.index] = group_index;
			}
		}
		
		group_meshlet_begin_index = group_meshlet_end_index;
	}
}

template<typename ElementID>
static u32 CreateMeshElementRemap(MeshView mesh, Array<ElementID> old_element_id_to_new_element_id) {
	u32 old_element_count = old_element_id_to_new_element_id.count;
	u32 new_element_count = 0;
	
	for (ElementID old_element_id = { 0 }; old_element_id.index < old_element_count; old_element_id.index += 1) {
		auto element = mesh[old_element_id];
		
		ElementID new_element_id = { u32_max };
		if (element.corner_list_base.index != u32_max) {
			new_element_id = { new_element_count };
			new_element_count += 1;
		
			mesh[new_element_id] = element;
		}
		old_element_id_to_new_element_id[old_element_id.index] = new_element_id;
	}
	
	return new_element_count;
}

static void CompactMesh(MeshView& mesh, Array<u32>& face_id_to_group_index, Allocator& allocator) {
	Array<FaceID> old_face_id_to_new_face_id;
	ArrayResize(old_face_id_to_new_face_id, allocator, mesh.face_count);
	
	Array<EdgeID> old_edge_id_to_new_edge_id;
	ArrayResize(old_edge_id_to_new_edge_id, allocator, mesh.edge_count);
	
	u32 new_face_count = CreateMeshElementRemap<FaceID>(mesh, old_face_id_to_new_face_id);
	u32 new_edge_count = CreateMeshElementRemap<EdgeID>(mesh, old_edge_id_to_new_edge_id);
	
	mesh.face_count = new_face_count;
	mesh.edge_count = new_edge_count;
	
	for (u32 i = 0; i < mesh.corner_count; i += 1) {
		auto& corner = mesh.corners[i];
		if (corner.face_id.index != u32_max) {
			corner.face_id = old_face_id_to_new_face_id[corner.face_id.index];
		}
		
		if (corner.edge_id.index != u32_max) {
			corner.edge_id = old_edge_id_to_new_edge_id[corner.edge_id.index];
		}
	}
	
	for (u32 i = 0; i < old_face_id_to_new_face_id.count; i += 1) {
		auto new_face_id = old_face_id_to_new_face_id[i];
		if (new_face_id.index == u32_max) continue;
		
		face_id_to_group_index[new_face_id.index] = face_id_to_group_index[i];
	}
	face_id_to_group_index.count = new_face_count;
}

void BuildVirtualGeometry(MeshView& mesh) {
	//
	// Virtual Geometry TODO:
	//
	// 1. Build mesh data structure (faces, edges, vertices, corners).
	// 2. Build meshlets from meshlet group faces.
	// 3. Build meshlet groups.
	// 4. Decimate meshlet groups.
	//  - Make sure locked vertex data is preserved and respected.
	//  - Make sure all groups are simplified by 1/2
	// 5. Output meshlet data.
	//  - For each meshlet there should be two errors. One for high quality version of the group, one for low quality.
	//
	
	Allocator allocator;
	
	Array<u32> face_id_to_group_index;
	ArrayResizeMemset(face_id_to_group_index, allocator, mesh.face_count, 0);
	u32 group_count = 1;
	
	for (u32 level = 0; level < 16; level += 1) {
		u32 allocator_high_water = allocator.memory_block_count;
		
		auto meshlet_build_result       = BuildMeshlets(mesh, allocator, CreateArrayView(face_id_to_group_index), group_count);
		auto meshlet_group_build_result = BuildMeshletGroups(mesh, allocator, meshlet_build_result.meshlets, meshlet_build_result.meshlet_adjacency);
		group_count = meshlet_group_build_result.prefix_sum.count;
		
		BuildFaceIdToGroupIndexMap(mesh, allocator, meshlet_build_result, meshlet_group_build_result, CreateArrayView(face_id_to_group_index));
		
		DecimateMeshFaceGroups(mesh, CreateArrayView(face_id_to_group_index), group_count);
		
		// Compact the mesh after decimation to remove unused faces and edges. TODO: Ensure source mesh is compacted and replace face and edge validity checks with asserts.
		CompactMesh(mesh, face_id_to_group_index, allocator);
		
		AllocatorFreeMemoryBlocks(allocator, allocator_high_water);
		
		if (mesh.face_count <= 1024) break;
	}
	
	AllocatorFreeMemoryBlocks(allocator);
}

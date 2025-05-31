#include "MeshDecimation.h"

//
// References:
// - Michael Garland, Paul S. Heckbert. 1997. Surface Simplification Using Quadric Error Metrics.
// - Thomas Wang. 1997. Integer Hash Function.
// - Hugues Hoppe. 1999. New Quadric Metric for Simplifying Meshes with Appearance Attributes.
// - Hugues Hoppe, Steve Marschner. 2000. Efficient Minimization of New Quadric Metric for Simplifying Meshes with Appearance Attributes.
// - Matthias Teschner, Bruno Heidelberger, Matthias Muller, Danat Pomeranets, Markus Gross. 2003. Optimized Spatial Hashing for Collision Detection of Deformable Objects.
// - Brian Karis, Rune Stubbe, Graham Wihlidal. 2021. Nanite A Deep Dive.
// - HSUEH-TI DEREK LIU, XIAOTING ZHANG, CEM YUKSEL. 2024. Simplifying Triangle Meshes in the Wild.
// - Arseny Kapoulkine. 2025. Meshoptimizer library. https://github.com/zeux/meshoptimizer. See license in THIRD_PARTY_LICENSES.md.
//

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <type_traits>


#if !defined(VGT_CACHE_LINE_SIZE)
#define VGT_CACHE_LINE_SIZE 64
#endif // !defined(VGT_CACHE_LINE_SIZE)


#if defined(_MSC_VER)
#define never_inline_function __declspec(noinline)
#define always_inline_function __forceinline
#else // !defined(_MSC_VER)
#define never_inline_function
#define always_inline_function
#endif // !defined(_MSC_VER)

#define compile_const constexpr static const


namespace VirtualGeometryTools
{

using u8  = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;

static_assert(sizeof(u8)  == 1);
static_assert(sizeof(u32) == 4);
static_assert(sizeof(u64) == 8);

compile_const u8  u8_max  = (u8)0xFF;
compile_const u32 u32_max = (u32)0xFFFF'FFFF;
compile_const u64 u64_max = (u64)0xFFFF'FFFF'FFFF'FFFF;

using Vector3 = VgtVector3;

compile_const u32 meshlet_max_vertex_count = 254;
compile_const u32 meshlet_max_face_degree  = 3;

compile_const u32 meshlet_max_face_count = 128;
compile_const u32 meshlet_min_face_count = 32;
compile_const float discontinuous_meshlet_max_expansion = 2.f; // Sqrt of the maximum AABB expansion when adding a discontinuous face to a meshlet.

compile_const u32 meshlet_group_max_meshlet_count = VGT_MESHLET_GROUP_SIZE;
compile_const u32 meshlet_group_min_meshlet_count = meshlet_group_max_meshlet_count / 2;
compile_const float discontinuous_meshlet_group_max_expansion = 4.f; // Sqrt of the maximum AABB expansion when adding a discontinuous meshlet to a group.

compile_const u32 virtual_geometry_max_levels_of_details = 16;


// Based on [Kapoulkine 2025] and [Teschner 2003].
static u32 ComputePositionHash(const Vector3& v) {
	const u32* key = (const u32*)&v;
	
	u32 x = key[0];
	u32 y = key[1];
	u32 z = key[2];
	
	// Replace negative zero with zero.
	x = (x == 0x80000000) ? 0 : x;
	y = (y == 0x80000000) ? 0 : y;
	z = (z == 0x80000000) ? 0 : z;
	
	// Scramble bits to make sure that integer coordinates have entropy in lower bits.
	x ^= x >> 17;
	y ^= y >> 17;
	z ^= z >> 17;
	
	// Optimized Spatial Hashing for Collision Detection of Deformable Objects.
	return (x * 73856093) ^ (y * 19349663) ^ (z * 83492791);
}

// Based on [Wang 1997]. 64 bit to 32 bit Hash Functions.
static u32 ComputeEdgeKeyHash(u64 key) {
	key = (~key) + (key << 18); // key = (key << 18) - key - 1;
	key = key ^ (key >> 31);
	key = key * 21; // key = (key + (key << 2)) + (key << 4);
	key = key ^ (key >> 11);
	key = key + (key << 6);
	key = key ^ (key >> 22);
	
	return (u32)key;
}


enum struct ElementType : u32 {
	Vertex = 0,
	Edge   = 1,
	Face   = 2,
	
	Count
};

struct CornerID {
	u32 index = 0;
};

struct FaceID {
	u32 index = 0;
	
	compile_const ElementType element_type = ElementType::Face;
};

struct VertexID {
	u32 index = 0;
	
	compile_const ElementType element_type = ElementType::Vertex;
};

struct EdgeID {
	u32 index = 0;
	
	compile_const ElementType element_type = ElementType::Edge;
};

struct AttributesID {
	u32 index = 0;
};

struct Face {
	CornerID corner_list_base; // Corner list around a face.
	u32 geometry_index = 0;
};

struct Edge {
	union {
		struct {
			VertexID vertex_0;
			VertexID vertex_1;
		};
		u64 edge_key = 0;
	};
	
	CornerID corner_list_base; // Corner list around an edge.
};

struct CornerListIDs {
	CornerID next;
	CornerID prev;
};

struct Corner {
	CornerListIDs corner_list_around[(u32)ElementType::Count];
	
	FaceID face_id;
	EdgeID edge_id;
	
	VertexID vertex_id;
	AttributesID attributes_id;
};
static_assert(sizeof(Corner) == 40, "Invalid Corner size.");

struct alignas(16) Vertex {
	Vector3 position;
	CornerID corner_list_base; // Corner list around a vertex.
};


struct alignas(VGT_CACHE_LINE_SIZE) MeshView {
	Face*   faces      = nullptr;
	Edge*   edges      = nullptr;
	Vertex* vertices   = nullptr;
	Corner* corners    = nullptr;
	float*  attributes = nullptr;
	
	u32 face_count      = 0;
	u32 edge_count      = 0;
	u32 vertex_count    = 0;
	u32 corner_count    = 0;
	u32 attribute_count = 0;
	u32 attribute_stride_dwords = 0;
	
	Face&   operator[] (FaceID face_id)             { return faces[face_id.index]; }
	Edge&   operator[] (EdgeID edge_id)             { return edges[edge_id.index]; }
	Vertex& operator[] (VertexID vertex_id)         { return vertices[vertex_id.index]; }
	Corner& operator[] (CornerID corner_id)         { return corners[corner_id.index]; }
	float*  operator[] (AttributesID attributes_id) { return attributes + attributes_id.index * attribute_stride_dwords; }
};
static_assert(sizeof(MeshView) == 64);


static u64 PackEdgeKey(VertexID vertex_id_0, VertexID vertex_id_1) {
	// Always pack VertexIDs in ascending order to ensure that PackEdgeKey(A, B) == PackEdgeKey(B, A) and they hash to the same value.
	return vertex_id_1.index > vertex_id_0.index ?
		((u64)vertex_id_1.index << 32) | (u64)vertex_id_0.index :
		((u64)vertex_id_0.index << 32) | (u64)vertex_id_1.index;
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
static ElementID CornerListMerge(MeshView mesh, ElementID element_0, ElementID element_1) {
	auto base_id_0 = mesh[element_0].corner_list_base;
	auto base_id_1 = mesh[element_1].corner_list_base;
	
	compile_const ElementType element_type_t = ElementID::element_type;
	compile_const u32 element_type = (u32)element_type_t;
	
	auto remaining_element_id = ElementID{ u32_max };
	if (base_id_0.index != u32_max && base_id_1.index != u32_max) {
		IterateCornerList<element_type_t>(mesh, base_id_1, [&](CornerID corner_id) {
			if constexpr (element_type_t == ElementType::Vertex) {
				mesh[corner_id].vertex_id = element_0;
			
				// TODO: This can be iteration over just incoming and outgoing edges of a corner.
				IterateCornerList<ElementType::Face>(mesh, corner_id, [&](CornerID corner_id) {
					auto& edge = mesh[mesh[corner_id].edge_id];
					if (edge.vertex_0.index == element_1.index) edge.vertex_0 = element_0;
					if (edge.vertex_1.index == element_1.index) edge.vertex_1 = element_0;
					VGT_ASSERT(edge.vertex_0.index != edge.vertex_1.index);
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
		
		remaining_element_id = element_0;
	} else if (base_id_0.index != u32_max) {
		remaining_element_id = element_0;
	} else if (base_id_1.index != u32_max) {
		remaining_element_id = element_1;
	}
	
	return remaining_element_id;
}

struct Allocator {
	compile_const u32 max_memory_block_count = 48;
	
	VgtAllocatorCallbacks callbacks;
	
	u32 memory_block_count = 0;
	void* memory_blocks[max_memory_block_count] = {};
};

static void* AllocateMemoryBlock(Allocator& allocator, void* old_memory_block, u64 size_bytes) {
	VGT_ASSERT(old_memory_block != nullptr || allocator.memory_block_count < Allocator::max_memory_block_count);
	VGT_ASSERT(old_memory_block == nullptr || allocator.memory_block_count > 0);
	VGT_ASSERT(old_memory_block == nullptr || allocator.memory_blocks[allocator.memory_block_count - 1] == old_memory_block);
	
	void* memory_block = allocator.callbacks.realloc(old_memory_block, size_bytes, allocator.callbacks.user_data);
	u32 memory_block_index = old_memory_block ? allocator.memory_block_count - 1 : allocator.memory_block_count++;
	
	allocator.memory_blocks[memory_block_index] = memory_block;
	
	return memory_block;
}

static void InitializeAllocator(Allocator& allocator, const VgtAllocatorCallbacks* callbacks) {
	if (callbacks) {
		allocator.callbacks = *callbacks;
	} else {
		allocator.callbacks.realloc = [](void* old_memory_block, u64 size_bytes, void*) { return realloc(old_memory_block, size_bytes); };
	}
}

static void AllocatorFreeMemoryBlocks(Allocator& allocator, u32 last_memory_block_index = 0) {
	for (u32 i = allocator.memory_block_count; i > last_memory_block_index; i -= 1) {
		allocator.callbacks.realloc(allocator.memory_blocks[i - 1], 0, allocator.callbacks.user_data);
	}
	allocator.memory_block_count = last_memory_block_index;
}

static u32 AllocatorFindMemoryBlock(Allocator& allocator, void* old_memory_block) {
	for (u32 i = allocator.memory_block_count; i > 0; i -= 1) {
		if (allocator.memory_blocks[i - 1] == old_memory_block) return i - 1;
	}
	return u32_max;	
}


#define DECLARE_ARRAY_OPERATORS() \
	T& operator[] (u32 index) { VGT_ASSERT(index < count); return data[index]; } \
	const T& operator[] (u32 index) const { VGT_ASSERT(index < count); return data[index]; } \
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
	
	DECLARE_ARRAY_OPERATORS()
};
static_assert(sizeof(Array<u32>) == 16);

template<typename T, u32 compile_time_capacity>
struct FixedSizeArray {
	using ValueType = T;
	compile_const u32 capacity = compile_time_capacity;
	
	T data[capacity] = {};
	u32 count = 0;

	DECLARE_ARRAY_OPERATORS()
};
static_assert(sizeof(FixedSizeArray<u32, 1>) == 8);

template<typename T>
struct ArrayView {
	using ValueType = T;
	
	T* data = nullptr;
	u32 count = 0;
	
	DECLARE_ARRAY_OPERATORS()
};
static_assert(sizeof(ArrayView<u32>) == 16);

#undef DECLARE_ARRAY_OPERATORS

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
always_inline_function static void ArrayAppend(ArrayT& array, const typename ArrayT::ValueType& value) {
	VGT_ASSERT(array.count < array.capacity);
	array.data[array.count++] = value;
}

template<typename T>
never_inline_function static void ArrayGrow(Array<T>& array, Allocator& allocator, u32 new_capacity) {
	u32 old_memory_block_index = AllocatorFindMemoryBlock(allocator, array.data);
	
	void* memory_block = allocator.callbacks.realloc(array.data, new_capacity * sizeof(T), allocator.callbacks.user_data);
	u32 memory_block_index = old_memory_block_index != u32_max ? old_memory_block_index : allocator.memory_block_count++;
	
	allocator.memory_blocks[memory_block_index] = memory_block;
	
	array.data     = (T*)memory_block;
	array.capacity = new_capacity;
}

template<typename T>
static void ArrayAppendMaybeGrow(Array<T>& array, Allocator& allocator, const T& value) {
	if (array.count >= array.capacity) ArrayGrow(array, allocator, ArrayComputeNewCapacity(array.capacity, array.count + 1));
	
	array.data[array.count++] = value;
}

template<typename ArrayT>
static void ArrayEraseSwap(ArrayT& array, u32 index) {
	VGT_ASSERT(index < array.count);
	
	array.data[index] = array.data[array.count - 1];
	array.count -= 1;
}

template<typename ArrayT>
static typename ArrayT::ValueType& ArrayLastElement(ArrayT& array) {
	VGT_ASSERT(array.count != 0);
	return array.data[array.count - 1];
}

template<typename ArrayT>
static ArrayView<typename ArrayT::ValueType> CreateArrayView(ArrayT array, u32 begin_index, u32 end_index) {
	return { array.data + begin_index, end_index - begin_index };
}

template<typename ArrayT>
static ArrayView<typename ArrayT::ValueType> CreateArrayView(ArrayT& array) {
	return { array.data, array.count };
}

//
// Based on [Kapoulkine 2025].
// See also https://fgiesen.wordpress.com/2015/02/22/triangular-numbers-mod-2n/
//
struct VertexHashTable {
	Array<VertexID> vertex_ids;
};

static VertexID HashTableAddOrFind(VertexHashTable& table, Array<Vertex>& vertices, const Vector3& position) {
	u32 table_size = table.vertex_ids.count;
	u32 mod_mask   = table_size - 1u;
	
	u32 hash  = ComputePositionHash(position);
	u32 index = (hash & mod_mask);
	
	for (u32 i = 0; i <= mod_mask; i += 1) {
		auto vertex_id = table.vertex_ids[index];
		
		if (vertex_id.index == u32_max) {
			auto new_vertex_id = VertexID{ vertices.count };
			table.vertex_ids[index] = new_vertex_id;
			
			Vertex vertex;
			vertex.position = position;
			vertex.corner_list_base.index = u32_max;
			ArrayAppend(vertices, vertex);
			
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
	Array<EdgeID> edge_ids;
};

static EdgeID HashTableAddOrFind(EdgeHashTable& table, Array<Edge>& edges, u64 edge_key) {
	u32 table_size = table.edge_ids.count;
	u32 mod_mask   = table_size - 1u;
	
	u32 hash  = ComputeEdgeKeyHash(edge_key);
	u32 index = (hash & mod_mask);
	
	for (u32 i = 0; i <= mod_mask; i += 1) {
		auto edge_id = table.edge_ids[index];
		
		if (edge_id.index == u32_max) {
			auto new_edge_id = EdgeID{ edges.count };
			table.edge_ids[index] = new_edge_id;
			
			Edge edge;
			edge.edge_key = edge_key;
			edge.corner_list_base.index = u32_max;
			ArrayAppend(edges, edge);
			
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
	
	KeyValue* keys_values = nullptr;
	u32 capacity = 0;
	u32 count    = 0;
	
	KeyValue* begin() { return keys_values; }
	KeyValue* end()   { return keys_values + capacity; }
};


static void HashTableClear(EdgeDuplicateMap& table) {
	memset(table.keys_values, 0xFF, table.capacity * sizeof(EdgeDuplicateMap::KeyValue));
	table.count = 0;
}

static EdgeID HashTableAddOrFind(EdgeDuplicateMap& table, Allocator& heap_allocator, u64 edge_key, EdgeID edge_id);

never_inline_function static void HashTableGrow(EdgeDuplicateMap& table, Allocator& heap_allocator, u32 new_capacity) {
	auto* old_keys_values = table.keys_values;
	u32 old_memory_block_index = AllocatorFindMemoryBlock(heap_allocator, old_keys_values);
	
	u32 old_capacity = table.capacity;
	void* memory_block = heap_allocator.callbacks.realloc(nullptr, new_capacity * sizeof(EdgeDuplicateMap::KeyValue), heap_allocator.callbacks.user_data);
	u32 memory_block_index = old_memory_block_index != u32_max ? old_memory_block_index : heap_allocator.memory_block_count++;
	
	heap_allocator.memory_blocks[memory_block_index] = memory_block;
	
	table.keys_values = (EdgeDuplicateMap::KeyValue*)memory_block;
	table.capacity    = new_capacity;
	
	HashTableClear(table);
	
	for (auto [key, value] : ArrayView<EdgeDuplicateMap::KeyValue>{ old_keys_values, old_capacity }) {
		if (key != u64_max) HashTableAddOrFind(table, heap_allocator, key, value);
	}
	
	heap_allocator.callbacks.realloc(old_keys_values, 0, heap_allocator.callbacks.user_data);
}

static EdgeID HashTableAddOrFind(EdgeDuplicateMap& table, Allocator& heap_allocator, u64 edge_key, EdgeID edge_id) {
	u32 table_size = table.capacity;

	compile_const u32 load_factor_percent = 85;
	if ((table.count + 1) * 100 >= table_size * load_factor_percent) {
		HashTableGrow(table, heap_allocator, table.capacity * 2);
		table_size = table.capacity;
	}
	
	u32 mod_mask = table_size - 1u;
	
	u32 hash  = ComputeEdgeKeyHash(edge_key);
	u32 index = (hash & mod_mask);
	
	for (u32 i = 0; i <= mod_mask; i += 1) {
		auto key_value = table.keys_values[index];
		
		if (key_value.edge_key == u64_max) {
			table.count += 1;
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

static u32 ComputeHashTableSize(u32 max_element_count) {
	u32 hash_table_size = 1;
	
	while (hash_table_size < max_element_count + max_element_count / 4) {
		hash_table_size = hash_table_size * 2;
	}
	
	return hash_table_size;
}


static MeshView BuildEditableMesh(Allocator& allocator, const VgtTriangleGeometryDesc* geometry_descs, u32 geometry_desc_count, u32 vertex_stride_bytes) {
	u32 vertex_stride_dwords    = vertex_stride_bytes / sizeof(u32);
	u32 attribute_stride_dwords = vertex_stride_dwords - 3;
	
	if (vertex_stride_dwords < 3 || attribute_stride_dwords > VGT_MAX_ATTRIBUTE_STRIDE_DWORDS) return {};
	
	
	u32 vertices_count = 0;
	u32 indices_count  = 0;
	
	for (u32 geometry_index = 0; geometry_index < geometry_desc_count; geometry_index += 1) {
		auto& desc = geometry_descs[geometry_index];
		vertices_count += desc.vertex_count;
		indices_count  += desc.index_count;
	}
	u32 triangle_count = (indices_count / 3);
	
	
	Array<Face>   faces;
	Array<Edge>   edges;
	Array<Vertex> vertices;
	Array<Corner> corners;
	Array<float>  attributes;
	ArrayReserve(vertices, allocator, vertices_count);
	ArrayResize(corners, allocator, indices_count);
	ArrayReserve(faces, allocator, triangle_count);
	ArrayReserve(edges, allocator, indices_count);
	ArrayResize(attributes, allocator, vertices_count * attribute_stride_dwords);
	
	MeshView mesh;
	mesh.faces           = faces.data;
	mesh.edges           = edges.data;
	mesh.vertices        = vertices.data;
	mesh.corners         = corners.data;
	mesh.attributes      = attributes.data;
	mesh.face_count      = faces.capacity;
	mesh.edge_count      = edges.capacity;
	mesh.vertex_count    = vertices.capacity;
	mesh.corner_count    = corners.count;
	mesh.attribute_count = attributes.count / attribute_stride_dwords;
	mesh.attribute_stride_dwords = attribute_stride_dwords;
	
	u32 allocator_high_water = allocator.memory_block_count;
	
	Array<VertexID> src_vertex_index_to_vertex_id;
	ArrayResize(src_vertex_index_to_vertex_id, allocator, vertices_count);
	
	VertexHashTable vertex_table;
	ArrayResizeMemset(vertex_table.vertex_ids, allocator, ComputeHashTableSize(vertices_count), 0xFF);
	
	EdgeHashTable edge_table;
	ArrayResizeMemset(edge_table.edge_ids, allocator, ComputeHashTableSize(indices_count), 0xFF);
	
	for (u32 geometry_index = 0, base_vertex_index = 0; geometry_index < geometry_desc_count; geometry_index += 1) {
		auto& desc = geometry_descs[geometry_index];
		
		for (u32 geometry_vertex_index = 0; geometry_vertex_index < desc.vertex_count; geometry_vertex_index += 1) {
			u32 vertex_index = base_vertex_index + geometry_vertex_index;
			
			auto* vertex = &desc.vertices[geometry_vertex_index * vertex_stride_dwords];
			memcpy(mesh[AttributesID{ vertex_index }], vertex + 3, attribute_stride_dwords * sizeof(u32));
			
			auto vertex_id = HashTableAddOrFind(vertex_table, vertices, *(Vector3*)vertex);
			src_vertex_index_to_vertex_id[vertex_index] = vertex_id;
		}
		base_vertex_index += desc.vertex_count;
	}
	mesh.vertex_count = vertices.count;
	
	for (u32 geometry_index = 0, base_vertex_index = 0; geometry_index < geometry_desc_count; geometry_index += 1) {
		auto& desc = geometry_descs[geometry_index];
		u32 geometry_triangle_count = (desc.index_count / 3);
		
		for (u32 triangle_index = 0; triangle_index < geometry_triangle_count; triangle_index += 1) {
			u32 indices[3] = {
				base_vertex_index + desc.indices[triangle_index * 3 + 0],
				base_vertex_index + desc.indices[triangle_index * 3 + 1],
				base_vertex_index + desc.indices[triangle_index * 3 + 2],
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
			
			
			auto face_id = FaceID{ faces.count };
			
			Face face;
			face.corner_list_base.index = u32_max;
			face.geometry_index = geometry_index;
			ArrayAppend(faces, face);
			
			for (u32 corner_index = 0; corner_index < 3; corner_index += 1) {
				auto corner_id = CornerID{ face_id.index * 3 + corner_index };
				
				auto edge_id = HashTableAddOrFind(edge_table, edges, edge_keys[corner_index]);
				
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
		base_vertex_index += desc.vertex_count;
	}
	mesh.edge_count   = edges.count;
	mesh.face_count   = faces.count;
	mesh.corner_count = faces.count * 3;
	
	AllocatorFreeMemoryBlocks(allocator, allocator_high_water);
	
	return mesh;
}


struct EdgeCollapseResult {
	VertexID remaining_vertex_id;
	u32 removed_face_count = 0;
};

static EdgeCollapseResult PerformEdgeCollapse(MeshView mesh, EdgeID edge_id, Allocator& heap_allocator, EdgeDuplicateMap& edge_duplicate_map, Array<EdgeID>& removed_edge_array) {
	auto& edge = mesh[edge_id];
	
	VGT_ASSERT(edge.vertex_0.index != edge.vertex_1.index);
	auto& vertex_0 = mesh[edge.vertex_0];
	auto& vertex_1 = mesh[edge.vertex_1];

	VGT_ASSERT(edge.corner_list_base.index != u32_max);
	VGT_ASSERT(vertex_0.corner_list_base.index != u32_max);
	VGT_ASSERT(vertex_1.corner_list_base.index != u32_max);
	VGT_ASSERT(vertex_0.corner_list_base.index != vertex_1.corner_list_base.index);
	
	removed_edge_array.count = 0;
	u32 removed_face_count = 0;
	IterateCornerList<ElementType::Edge>(mesh, edge.corner_list_base, [&](CornerID corner_id) {
		auto& corner = mesh[corner_id];
		auto& face   = mesh[corner.face_id];
		
		IterateCornerList<ElementType::Face>(mesh, face.corner_list_base, [&](CornerID corner_id) {
			CornerListRemove<ElementType::Vertex>(mesh, corner_id);
			bool edge_removed = CornerListRemove<ElementType::Edge>(mesh, corner_id);
			bool face_removed = CornerListRemove<ElementType::Face>(mesh, corner_id);
			
			if (edge_removed) ArrayAppendMaybeGrow(removed_edge_array, heap_allocator, mesh[corner_id].edge_id);
			removed_face_count += (u32)face_removed;
		});
	});
	
	auto remaining_vertex_id = CornerListMerge<VertexID>(mesh, edge.vertex_0, edge.vertex_1);

	if (remaining_vertex_id.index != u32_max) {
		auto remaining_base_id = mesh[remaining_vertex_id].corner_list_base;
		
		IterateCornerList<ElementType::Vertex>(mesh, remaining_base_id, [&](CornerID corner_id) {
			// TODO: This can be iteration over just incoming and outgoing edges of a corner.
			IterateCornerList<ElementType::Face>(mesh, corner_id, [&](CornerID corner_id) {
				auto edge_id_1 = mesh[corner_id].edge_id;
				auto& edge_1 = mesh[edge_id_1];
				
				auto edge_id_0 = HashTableAddOrFind(edge_duplicate_map, heap_allocator, PackEdgeKey(edge_1.vertex_0, edge_1.vertex_1), edge_id_1);
				if (edge_id_0.index != edge_id_1.index) {
					CornerListMerge<EdgeID>(mesh, edge_id_0, edge_id_1);
					ArrayAppendMaybeGrow(removed_edge_array, heap_allocator, edge_id_1);
				}
			});
		});
		
		IterateCornerList<ElementType::Vertex>(mesh, remaining_base_id, [&](CornerID corner_id_0) {
			IterateCornerList<ElementType::Face>(mesh, corner_id_0, [&](CornerID corner_id_1) {
				if (remaining_base_id.index == corner_id_1.index) return;
				
				IterateCornerList<ElementType::Vertex>(mesh, corner_id_1, [&](CornerID corner_id_2) {
					if (corner_id_1.index == corner_id_2.index) return;
					
					IterateCornerList<ElementType::Face>(mesh, corner_id_2, [&](CornerID corner_id) {
						auto edge_id = mesh[corner_id].edge_id;
						auto& edge = mesh[edge_id];
						
						HashTableAddOrFind(edge_duplicate_map, heap_allocator, PackEdgeKey(edge.vertex_0, edge.vertex_1), edge_id);
					});
				});
			});
		});
	}
	
	
	EdgeCollapseResult result;
	result.remaining_vertex_id = remaining_vertex_id;
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
static_assert(sizeof(Quadric) == sizeof(float) * 11);

struct QuadricAttributeGradient {
	Vector3 g = { 0.f, 0.f, 0.f };
	float   d = 0.f;
};

struct QuadricWithAttributes : Quadric {
#if VGT_ENABLE_ATTRIBUTE_SUPPORT
	QuadricAttributeGradient attributes[VGT_MAX_ATTRIBUTE_STRIDE_DWORDS]; // Note that this array is used as a 'flexible array'.
	
	// Quadric with attributes cannot be copied by value as it's variable size (i.e. it might be missing trailing attributes).
	QuadricWithAttributes& operator= (const QuadricWithAttributes&) = delete;
	QuadricWithAttributes& operator= (QuadricWithAttributes&&) = delete;
#endif // VGT_ENABLE_ATTRIBUTE_SUPPORT
};
static_assert(sizeof(QuadricWithAttributes) == sizeof(Quadric) + sizeof(QuadricAttributeGradient) * VGT_MAX_ATTRIBUTE_STRIDE_DWORDS);


//
// Matt Pharr's blog. 2019. Accurate Differences of Products with Kahan's Algorithm.
//
always_inline_function static float DifferenceOfProducts(float a, float b, float c, float d) {
	float cd = c * d;
	float err = fmaf(c, -d,  cd);
	float dop = fmaf(a,  b, -cd);
	return dop + err;
}

always_inline_function static Vector3 operator+ (const Vector3& lh, const Vector3& rh) { return Vector3{ lh.x + rh.x, lh.y + rh.y, lh.z + rh.z }; }
always_inline_function static Vector3 operator- (const Vector3& lh, const Vector3& rh) { return Vector3{ lh.x - rh.x, lh.y - rh.y, lh.z - rh.z }; }
always_inline_function static Vector3 operator* (const Vector3& lh, float rh) { return Vector3{ lh.x * rh, lh.y * rh, lh.z * rh }; }
always_inline_function static Vector3 operator+ (const Vector3& lh, float rh) { return Vector3{ lh.x + rh, lh.y + rh, lh.z + rh }; }
always_inline_function static Vector3 operator- (const Vector3& lh, float rh) { return Vector3{ lh.x - rh, lh.y - rh, lh.z - rh }; }
always_inline_function static float DotProduct(const Vector3& lh, const Vector3& rh) { return lh.x * rh.x + lh.y * rh.y + lh.z * rh.z; }
always_inline_function static float Length(const Vector3& v) { return sqrtf(DotProduct(v, v)); }
always_inline_function static Vector3 Normalize(const Vector3& v) { float length = Length(v); return length < FLT_EPSILON ? v : v * (1.f / length); }
always_inline_function static Vector3 CrossProduct(const Vector3& lh, const Vector3& rh) { return Vector3{ lh.y * rh.z - lh.z * rh.y, lh.z * rh.x - lh.x * rh.z, lh.x * rh.y - lh.y * rh.x }; }

always_inline_function static Vector3 VectorMax(const Vector3& lh, const Vector3& rh) {
	Vector3 result;
	result.x = lh.x > rh.x ? lh.x : rh.x;
	result.y = lh.y > rh.y ? lh.y : rh.y;
	result.z = lh.z > rh.z ? lh.z : rh.z;
	return result;
}

always_inline_function static Vector3 VectorMin(const Vector3& lh, const Vector3& rh) {
	Vector3 result;
	result.x = lh.x < rh.x ? lh.x : rh.x;
	result.y = lh.y < rh.y ? lh.y : rh.y;
	result.z = lh.z < rh.z ? lh.z : rh.z;
	return result;
}

template<typename T>
always_inline_function T Clamp(T value, T min, T max) {
	if (value < min) value = min;
	if (value > max) value = max;
	return value;
}


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

static void AccumulateQuadricWithAttributes(QuadricWithAttributes& accumulator, const QuadricWithAttributes& quadric, u32 attribute_stride_dwords) {
	AccumulateQuadric(accumulator, quadric);
	
#if VGT_ENABLE_ATTRIBUTE_SUPPORT
	for (u32 i = 0; i < attribute_stride_dwords; i += 1) {
		auto& attribute_accumulator = accumulator.attributes[i];
		auto& attribute_quadric     = quadric.attributes[i];

		attribute_accumulator.g.x += attribute_quadric.g.x;
		attribute_accumulator.g.y += attribute_quadric.g.y;
		attribute_accumulator.g.z += attribute_quadric.g.z;
		attribute_accumulator.d   += attribute_quadric.d;
	}
#endif // VGT_ENABLE_ATTRIBUTE_SUPPORT
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
	auto normal                = normal_length < FLT_EPSILON ? normal_direction : normal_direction * (1.f / normal_length);
	auto distance_to_triangle  = -DotProduct(normal, p1);
	
	ComputePlanarQuadric(quadric, normal, distance_to_triangle, DotProduct(p10, p10) * weight);
}

static void ComputeFaceQuadricWithAttributes(QuadricWithAttributes& quadric, const Vector3& p0, const Vector3& p1, const Vector3& p2, float* a0, float* a1, float* a2, float* attribute_weights, u32 attribute_stride_dwords) {
	auto p10 = p1 - p0;
	auto p20 = p2 - p0;
	
	auto scaled_normal       = CrossProduct(p10, p20);
	auto twice_triangle_area = Length(scaled_normal);
	auto n                   = twice_triangle_area < FLT_EPSILON ? scaled_normal : scaled_normal * (1.f / twice_triangle_area);
	
	float weight = twice_triangle_area * 0.5f;
	ComputePlanarQuadric(quadric, n, -DotProduct(n, p0), weight);
	
	
#if VGT_ENABLE_ATTRIBUTE_SUPPORT
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
		float attribute_weight = attribute_weights[i];
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
#endif // VGT_ENABLE_ATTRIBUTE_SUPPORT
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

static float ComputeQuadricErrorWithAttributes(const QuadricWithAttributes& q, const Vector3& p, u32 attribute_stride_dwords) {
	//
	// error = p^T * A * p + 2 * b * v + c
	//
	//                           ( a00,   a01,   a02,  -g0.x, -gi.x)   (p.x)
	// (p.x, p.y, p.z, s0, si) * ( a01,   a11,   a12,  -g0.y, -gi.y) * (p.y) + 2 * (b, -d0, -di) * (p, s0, si) + (c + d0^2 + di^2)
	//                           ( a02,   a12,   a22,  -g0.z, -gi.z)   (p.z)
	//                           (-g0.x, -g0.y, -g0.z,  q.w,   0.0 )   (s0 )
	//                           (-gi.x, -gi.y, -gi.z,  0.0,   q.w )   (si )
	// 
	float weighted_error = ComputeQuadricError(q, p);
	
#if VGT_ENABLE_ATTRIBUTE_SUPPORT
	if (q.weight < FLT_EPSILON) return weighted_error;
	
	float rcp_weight = 1.f / q.weight;
	for (u32 i = 0; i < attribute_stride_dwords; i += 1) {
		auto g = q.attributes[i].g;
		auto d = q.attributes[i].d;
		
		float s = (DotProduct(g, p) + d) * rcp_weight;
		
		//
		// Simplified by replacing first three lines with a dot product, and substituting -DotProduct(g, p) for (d - s * q.weight).
		// Note that d0^2 + di^2 are added to c directly in ComputeFaceQuadricWithAttributes.
		//
		// p.x * (-g.x * s) +
		// p.y * (-g.y * s) +
		// p.z * (-g.z * s) +
		// s * (-DotProduct(g, p) + s * q.weight) +
		// -2.f * d * s;
		//
		float weighted_attribute_error = s * s * -q.weight;
		
		weighted_error += weighted_attribute_error;
	}
#endif // VGT_ENABLE_ATTRIBUTE_SUPPORT
	
	return fabsf(weighted_error);
}

#if VGT_ENABLE_ATTRIBUTE_SUPPORT
// Attribute computation for zero weight quadrics should be handled by the caller.
static bool ComputeWedgeAttributes(const QuadricWithAttributes& q, const Vector3& p, float* attributes, float* rcp_attribute_weights, u32 attribute_stride_dwords) {
	if (q.weight < FLT_EPSILON) return false;
	
	float rcp_weight = 1.f / q.weight;
	for (u32 i = 0; i < attribute_stride_dwords; i += 1) {
		auto g = q.attributes[i].g;
		auto d = q.attributes[i].d;
		
		float s = (DotProduct(g, p) + d) * rcp_weight;
		
		attributes[i] = s * rcp_attribute_weights[i];
	}
	
	return true;
}
#endif // VGT_ENABLE_ATTRIBUTE_SUPPORT

static bool ComputeOptimalVertexPosition(const QuadricWithAttributes& quadric, Vector3& optimal_position, u32 attribute_stride_dwords) {
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
	
#if VGT_ENABLE_ATTRIBUTE_SUPPORT
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
		h0 += (g.x * d);
		h1 += (g.y * d);
		h2 += (g.z * d);
	}
#endif // VGT_ENABLE_ATTRIBUTE_SUPPORT
	
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

static void AttributeWedgeMapAdd(AttributeWedgeMap& small_set, AttributesID key, u32 value) {
	u32 index = (small_set.count < AttributeWedgeMap::capacity) ? small_set.count++ : (AttributeWedgeMap::capacity - 1);
	small_set.keys[index]   = key;
	small_set.values[index] = (u8)value;
}

static u32 AttributeWedgeMapFind(AttributeWedgeMap& small_set, AttributesID key) {
	for (u32 i = 0; i < small_set.count; i += 1) {
		if (small_set.keys[i].index == key.index) return small_set.values[i];
	}
	
	return u32_max;
}

struct QuadricWithAttributesArray {
	void* data   = nullptr;
	u32 count    = 0;
	u32 capacity = 0;
	u32 data_stride_bytes = 0;
	
	QuadricWithAttributes& operator[] (u32 index) { VGT_ASSERT(index < count); return *(QuadricWithAttributes*)((u8*)data + index * data_stride_bytes); }
};

// TODO: Reuse existing array functions.
never_inline_function static void ArrayGrow(QuadricWithAttributesArray& array, Allocator& allocator, u32 new_capacity) {
	u32 old_memory_block_index = AllocatorFindMemoryBlock(allocator, array.data);
	
	void* memory_block = allocator.callbacks.realloc(array.data, new_capacity * array.data_stride_bytes + sizeof(QuadricWithAttributes), allocator.callbacks.user_data);
	u32 memory_block_index = old_memory_block_index != u32_max ? old_memory_block_index : allocator.memory_block_count++;
	
	allocator.memory_blocks[memory_block_index] = memory_block;
	
	array.data     = memory_block;
	array.capacity = new_capacity;
}

// TODO: Reuse existing array functions.
static void ArrayAppendMaybeGrow(QuadricWithAttributesArray& array, Allocator& allocator, const QuadricWithAttributes& value) {
	if (array.count >= array.capacity) ArrayGrow(array, allocator, ArrayComputeNewCapacity(array.capacity, array.count + 1));
	
	memcpy(&array[array.count++], &value, array.data_stride_bytes);
}

// TODO: Reuse existing array functions.
static void ArrayReserve(QuadricWithAttributesArray& array, Allocator& allocator, u32 capacity, u32 attribute_stride_dwords) {
	array.data_stride_bytes = sizeof(Quadric) + sizeof(QuadricAttributeGradient) * attribute_stride_dwords;
	array.data     = AllocateMemoryBlock(allocator, array.data, capacity * array.data_stride_bytes + sizeof(QuadricWithAttributes));
	array.capacity = capacity;
}

// TODO: Reuse existing array functions.
static void ArrayResizeMemset(QuadricWithAttributesArray& array, Allocator& allocator, u32 new_count, u32 attribute_stride_dwords, u8 pattern) { // Fills new elements with a byte pattern.
	ArrayReserve(array, allocator, new_count, attribute_stride_dwords);
	array.count = new_count;
	memset(array.data, pattern, new_count * array.data_stride_bytes);
}


struct alignas(VGT_CACHE_LINE_SIZE) MeshDecimationState {
	// Edge quadrics accumulated on vertices.
	Array<Quadric> vertex_edge_quadrics;
	
	// Face quadrics accumulated on attributes.
	QuadricWithAttributesArray attribute_face_quadrics;
	
	EdgeDuplicateMap edge_duplicate_map;
	Array<EdgeID>    removed_edge_array;
	
	QuadricWithAttributesArray wedge_quadrics;
	Array<AttributesID> wedge_attributes_ids;
	AttributeWedgeMap wedge_attribute_set;
	
	alignas(16) float attribute_weights[VGT_MAX_ATTRIBUTE_STRIDE_DWORDS];
	alignas(16) float rcp_attribute_weights[VGT_MAX_ATTRIBUTE_STRIDE_DWORDS];
	
	float position_weight;
	float rcp_position_weight;
};

struct EdgeCollapseError {
	float min_error;
	Vector3 new_position;
};

static EdgeCollapseError ComputeEdgeCollapseError(MeshView mesh, Allocator& heap_allocator, MeshDecimationState& state, EdgeID edge_id) {
	state.wedge_quadrics.count       = 0;
	state.wedge_attributes_ids.count = 0;
	state.wedge_attribute_set.count  = 0;
	
	auto& edge = mesh[edge_id];
	u32 attribute_stride_dwords = mesh.attribute_stride_dwords;
	
	// Wedges spanning collapsed edge must be unified. Manually set their wedge index to the same value and accumulate quadrics.
	// For reference see [Hugues Hoppe 1999] Section 5 Attribute Discontinuities, Figure 5.
	IterateCornerList<ElementType::Edge>(mesh, edge.corner_list_base, [&](CornerID corner_id) {
		auto& corner_0 = mesh[corner_id];
		auto& corner_1 = mesh[corner_0.corner_list_around[(u32)ElementType::Face].next];
		
		u32 wedge_index_0 = AttributeWedgeMapFind(state.wedge_attribute_set, corner_0.attributes_id);
		u32 wedge_index_1 = AttributeWedgeMapFind(state.wedge_attribute_set, corner_1.attributes_id);
		
		if (wedge_index_0 == u32_max && wedge_index_1 == u32_max) {
			u32 wedge_index = state.wedge_quadrics.count;
			AttributeWedgeMapAdd(state.wedge_attribute_set, corner_0.attributes_id, wedge_index);
			AttributeWedgeMapAdd(state.wedge_attribute_set, corner_1.attributes_id, wedge_index);
			
			ArrayAppendMaybeGrow(state.wedge_attributes_ids, heap_allocator, corner_0.attributes_id);
			ArrayAppendMaybeGrow(state.wedge_quadrics,       heap_allocator, state.attribute_face_quadrics[corner_0.attributes_id.index]);
			
			AccumulateQuadricWithAttributes(state.wedge_quadrics[wedge_index], state.attribute_face_quadrics[corner_1.attributes_id.index], attribute_stride_dwords);
		} else if (wedge_index_0 == u32_max && wedge_index_1 != u32_max) {
			AttributeWedgeMapAdd(state.wedge_attribute_set, corner_0.attributes_id, wedge_index_1);
			AccumulateQuadricWithAttributes(state.wedge_quadrics[wedge_index_1], state.attribute_face_quadrics[corner_0.attributes_id.index], attribute_stride_dwords);
		} else if (wedge_index_0 != u32_max && wedge_index_1 == u32_max) {
			AttributeWedgeMapAdd(state.wedge_attribute_set, corner_1.attributes_id, wedge_index_0);
			AccumulateQuadricWithAttributes(state.wedge_quadrics[wedge_index_0], state.attribute_face_quadrics[corner_1.attributes_id.index], attribute_stride_dwords);
		}
	});
	
	auto accumulate_quadrics = [&](CornerID corner_id) {
		auto attribute_id = mesh[corner_id].attributes_id;
		
		if (AttributeWedgeMapFind(state.wedge_attribute_set, attribute_id) == u32_max) {
			AttributeWedgeMapAdd(state.wedge_attribute_set,  attribute_id,   state.wedge_quadrics.count);
			ArrayAppendMaybeGrow(state.wedge_quadrics,       heap_allocator, state.attribute_face_quadrics[attribute_id.index]);
			ArrayAppendMaybeGrow(state.wedge_attributes_ids, heap_allocator, attribute_id);
		}
	};
	
	auto& v0 = mesh[edge.vertex_0];
	auto& v1 = mesh[edge.vertex_1];
	
	IterateCornerList<ElementType::Vertex>(mesh, v0.corner_list_base, accumulate_quadrics);
	IterateCornerList<ElementType::Vertex>(mesh, v1.corner_list_base, accumulate_quadrics);
	
	EdgeCollapseError collapse_error;
	collapse_error.min_error = FLT_MAX;
	{
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
		
		u32 wedge_count = state.wedge_quadrics.count;
		for (u32 i = 0; i < wedge_count; i += 1) {
			AccumulateQuadricWithAttributes(total_quadric, state.wedge_quadrics[i], attribute_stride_dwords);
		}
		
		Vector3 optimal_position;
		if (ComputeOptimalVertexPosition(total_quadric, optimal_position, attribute_stride_dwords)) {
			// Override average position with optimal position if it can be computed.
			candidate_positions[2] = optimal_position * state.rcp_position_weight;
		}
		
		u32 valid_position_mask = ValidateEdgeCollapsePositions(mesh, edge, candidate_positions, candidate_position_count);
		for (u32 i = 0; i < candidate_position_count; i += 1) {
			float error = valid_position_mask & (1u << i) ? 0.f : total_quadric.weight;
			if (error > collapse_error.min_error) continue;
			
			auto p = candidate_positions[i] * state.position_weight;
			error += ComputeQuadricError(edge_quadrics, p);
			if (error > collapse_error.min_error) continue;
			
			for (u32 i = 0; i < wedge_count; i += 1) {
				error += ComputeQuadricErrorWithAttributes(state.wedge_quadrics[i], p, attribute_stride_dwords);
			}
			if (error > collapse_error.min_error) continue;
			
			
			collapse_error.min_error    = error;
			collapse_error.new_position = candidate_positions[i];
		}
	}
	
	return collapse_error;
}

struct EdgeCollapseHeap {
	Array<EdgeID> heap_index_to_edge_id;
	Array<u32>    edge_id_to_heap_index;
	Array<float>  edge_collapse_errors;
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
	u32 element_count = heap.edge_collapse_errors.count;
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
	VGT_ASSERT(heap.edge_collapse_errors.count != 0);
	
	auto edge_id = heap.heap_index_to_edge_id[0];

	heap.edge_collapse_errors[0]  = ArrayLastElement(heap.edge_collapse_errors);
	heap.heap_index_to_edge_id[0] = ArrayLastElement(heap.heap_index_to_edge_id);
	
	heap.edge_id_to_heap_index[heap.heap_index_to_edge_id[0].index] = 0;
	heap.edge_id_to_heap_index[edge_id.index] = u32_max;

	heap.edge_collapse_errors.count  -= 1;
	heap.heap_index_to_edge_id.count -= 1;
	
	EdgeCollapseHeapSiftDown(heap, 0);
	
	return edge_id;
}

static void EdgeCollapseHeapRemove(EdgeCollapseHeap& heap, u32 heap_index) {
	auto edge_id = heap.heap_index_to_edge_id[heap_index];
	
	bool sift_up = true;
	if (heap.edge_collapse_errors.count > heap_index) {
		float prev_error = heap.edge_collapse_errors[heap_index];

		heap.edge_collapse_errors[heap_index]  = ArrayLastElement(heap.edge_collapse_errors);
		heap.heap_index_to_edge_id[heap_index] = ArrayLastElement(heap.heap_index_to_edge_id);

		float new_error = heap.edge_collapse_errors[heap_index];
		sift_up = new_error < prev_error;
	}
	
	heap.edge_id_to_heap_index[heap.heap_index_to_edge_id[heap_index].index] = heap_index;
	heap.edge_id_to_heap_index[edge_id.index] = u32_max;

	heap.edge_collapse_errors.count  -= 1;
	heap.heap_index_to_edge_id.count -= 1;
	
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
	if (heap.edge_collapse_errors.count == 0) return;
	
	u32 node_index = HeapParentIndex(heap.edge_collapse_errors.count - 1);
	
	for (u32 i = node_index; i > 0; i -= 1) {
		EdgeCollapseHeapSiftDown(heap, i);
	}
	EdgeCollapseHeapSiftDown(heap, 0);
}

static void InitializeMeshDecimationState(MeshView mesh, float* attribute_weights, Allocator& allocator, Allocator& heap_allocator, MeshDecimationState& state) {
	ArrayResizeMemset(state.vertex_edge_quadrics, allocator, mesh.vertex_count, 0);
	ArrayResizeMemset(state.attribute_face_quadrics, allocator, mesh.attribute_count, mesh.attribute_stride_dwords, 0);
	
	ArrayReserve(state.wedge_quadrics,       heap_allocator, 64, mesh.attribute_stride_dwords);
	ArrayReserve(state.wedge_attributes_ids, heap_allocator, 64);
	ArrayReserve(state.removed_edge_array,   heap_allocator, 64);
	HashTableGrow(state.edge_duplicate_map,  heap_allocator, ComputeHashTableSize(128u));
	
	
	for (u32 i = 0; i < VGT_MAX_ATTRIBUTE_STRIDE_DWORDS; i += 1) {
		float attribute_weight = attribute_weights && i < mesh.attribute_stride_dwords ? attribute_weights[i] : 0.f;
		
		if (attribute_weight > FLT_EPSILON) {
			state.attribute_weights[i]     = attribute_weight;
			state.rcp_attribute_weights[i] = 1.f / attribute_weight;
		} else {
			state.attribute_weights[i]     = 1.f;
			state.rcp_attribute_weights[i] = 1.f;
		}
	}
	
	
	{
		float twice_mesh_surface_area = 0.f;
		for (FaceID face_id = { 0 }; face_id.index < mesh.face_count; face_id.index += 1) {
			auto& face = mesh[face_id];
			VGT_ASSERT(face.corner_list_base.index != u32_max);
			
			auto& c1 = mesh[face.corner_list_base];
			auto& c0 = mesh[c1.corner_list_around[(u32)ElementType::Face].prev];
			auto& c2 = mesh[c1.corner_list_around[(u32)ElementType::Face].next];
			
			auto p0 = mesh[c0.vertex_id].position;
			auto p1 = mesh[c1.vertex_id].position;
			auto p2 = mesh[c2.vertex_id].position;
			
			twice_mesh_surface_area += Length(CrossProduct(p1 - p0, p2 - p0));
		}
		
		//
		// Scale the mesh such that average face surface area is equal to 1.0.
		// See [Karis 2021] for reference.
		//
		float face_surface_area       = mesh.face_count ? twice_mesh_surface_area * 0.5f / (float)mesh.face_count : 1.f;
		float rcp_mesh_position_scale = sqrtf(face_surface_area);
		float mesh_position_scale     = 1.f / rcp_mesh_position_scale;
		
		bool is_valid_position_scale = (rcp_mesh_position_scale > FLT_EPSILON);
		state.position_weight     = is_valid_position_scale ? mesh_position_scale     : 1.f;
		state.rcp_position_weight = is_valid_position_scale ? rcp_mesh_position_scale : 1.f;
	}
	
	{
		float position_weight = state.position_weight;
		u32 attribute_stride_dwords = mesh.attribute_stride_dwords;
		for (FaceID face_id = { 0 }; face_id.index < mesh.face_count; face_id.index += 1) {
			auto& face = mesh[face_id];
			VGT_ASSERT(face.corner_list_base.index != u32_max);
			
			auto& c1 = mesh[face.corner_list_base];
			auto& c0 = mesh[c1.corner_list_around[(u32)ElementType::Face].prev];
			auto& c2 = mesh[c1.corner_list_around[(u32)ElementType::Face].next];
			
			auto p0 = mesh[c0.vertex_id].position * position_weight;
			auto p1 = mesh[c1.vertex_id].position * position_weight;
			auto p2 = mesh[c2.vertex_id].position * position_weight;
			
			auto* a0 = mesh[c0.attributes_id];
			auto* a1 = mesh[c1.attributes_id];
			auto* a2 = mesh[c2.attributes_id];
			
			QuadricWithAttributes quadric;
			ComputeFaceQuadricWithAttributes(quadric, p0, p1, p2, a0, a1, a2, state.attribute_weights, attribute_stride_dwords);
			
			AccumulateQuadricWithAttributes(state.attribute_face_quadrics[c0.attributes_id.index], quadric, attribute_stride_dwords);
			AccumulateQuadricWithAttributes(state.attribute_face_quadrics[c1.attributes_id.index], quadric, attribute_stride_dwords);
			AccumulateQuadricWithAttributes(state.attribute_face_quadrics[c2.attributes_id.index], quadric, attribute_stride_dwords);
		}
	}
	
	{
		float position_weight = state.position_weight;
		for (EdgeID edge_id = { 0 }; edge_id.index < mesh.edge_count; edge_id.index += 1) {
			auto& edge = mesh[edge_id];
			VGT_ASSERT(edge.corner_list_base.index != u32_max);
			
			auto& c0 = mesh[edge.corner_list_base];
			auto& c1 = mesh[c0.corner_list_around[(u32)ElementType::Face].next];
			auto& c2 = mesh[c0.corner_list_around[(u32)ElementType::Face].prev];
			
			auto attributes_id_0 = c0.attributes_id;
			auto attributes_id_1 = c1.attributes_id;
			
			u32 geometry_index_0 = mesh[c0.face_id].geometry_index;
			
			u32 edge_degree = 0;
			bool attribute_edge = false;
			IterateCornerList<ElementType::Edge>(mesh, edge.corner_list_base, [&](CornerID corner_id) {
				auto& corner = mesh[corner_id];
				
				auto attributes_id = corner.attributes_id;
				attribute_edge |= (attributes_id.index != attributes_id_0.index) && (attributes_id.index != attributes_id_1.index);
				
				u32 geometry_index = mesh[corner.face_id].geometry_index;
				attribute_edge |= (geometry_index != geometry_index_0);
				
				edge_degree += 1;
			});
			if (edge_degree != 1 && attribute_edge == false) continue;
			
			
			// (v0, v1) is the current edge.
			auto v0 = c0.vertex_id;
			auto v1 = c1.vertex_id;
			auto v2 = c2.vertex_id;
			
			auto p0 = mesh[v0].position * position_weight;
			auto p1 = mesh[v1].position * position_weight;
			auto p2 = mesh[v2].position * position_weight;
			
			Quadric quadric;
			ComputeEdgeQuadric(quadric, p0, p1, p2, 1.f / edge_degree);
			
			AccumulateQuadric(state.vertex_edge_quadrics[v0.index], quadric);
			AccumulateQuadric(state.vertex_edge_quadrics[v1.index], quadric);
		}
	}
}

static float DecimateMeshFaceGroup(
	MeshView mesh,
	Allocator& heap_allocator,
	MeshDecimationState& state,
	EdgeCollapseHeap& edge_collapse_heap,
	VgtNormalizeVertexAttributes normalize_vertex_attributes,
	u8* changed_vertex_mask,
	u32 target_face_count,
	u32 active_face_count,
	float target_error_limit = FLT_MAX) {
	
	target_error_limit = target_error_limit * state.position_weight;
	target_error_limit = target_error_limit * target_error_limit;
	
	float max_error = 0.f;
	while (active_face_count > target_face_count && edge_collapse_heap.edge_collapse_errors.count) {
		// ~80% of the execution time.
		for (auto& [edge_key, edge_id] : state.edge_duplicate_map) {
			if (edge_key == u64_max) continue;
			
			u32 heap_index = edge_collapse_heap.edge_id_to_heap_index[edge_id.index];
			if (heap_index == u32_max) continue;
			
			auto collapse_error = ComputeEdgeCollapseError(mesh, heap_allocator, state, edge_id);
			
			EdgeCollapseHeapUpdate(edge_collapse_heap, heap_index, collapse_error.min_error);
		}
		HashTableClear(state.edge_duplicate_map);
		
		
		auto edge_id = EdgeCollapseHeapPop(edge_collapse_heap);
		
		// 2% of the execution time
		auto collapse_error = ComputeEdgeCollapseError(mesh, heap_allocator, state, edge_id);
		if (collapse_error.min_error > target_error_limit) break;
		
		max_error = max_error < collapse_error.min_error ? collapse_error.min_error : max_error;
		
		// 15% of the execution time
		auto [remaining_vertex_id, removed_face_count] = PerformEdgeCollapse(mesh, edge_id, heap_allocator, state.edge_duplicate_map, state.removed_edge_array);
		active_face_count -= removed_face_count;
		
		for (auto edge_id : state.removed_edge_array) {
			u32 heap_index = edge_collapse_heap.edge_id_to_heap_index[edge_id.index];
			if (heap_index != u32_max) EdgeCollapseHeapRemove(edge_collapse_heap, heap_index);
		}
		
		// Update vertices.
		{
			auto& edge = mesh[edge_id];
			mesh[edge.vertex_0].position = collapse_error.new_position;
			mesh[edge.vertex_1].position = collapse_error.new_position;
			
			Quadric quadric = state.vertex_edge_quadrics[edge.vertex_0.index];
			AccumulateQuadric(quadric, state.vertex_edge_quadrics[edge.vertex_1.index]);
			
			state.vertex_edge_quadrics[edge.vertex_0.index] = quadric;
			state.vertex_edge_quadrics[edge.vertex_1.index] = quadric;
		}
		
#if VGT_ENABLE_ATTRIBUTE_SUPPORT
		// Update attributes.
		if (remaining_vertex_id.index != u32_max) {
			u32 wedge_count = state.wedge_quadrics.count;
			u32 attribute_stride_dwords = mesh.attribute_stride_dwords;
			auto weighted_new_position = collapse_error.new_position * state.position_weight;
			
			for (u32 i = 0; i < wedge_count; i += 1) {
				auto& wedge_quadric = state.wedge_quadrics[i];
				auto  attributes_id = state.wedge_attributes_ids[i];
				auto* attributes    = mesh[attributes_id];
				
				memcpy(&state.attribute_face_quadrics[attributes_id.index], &wedge_quadric, state.attribute_face_quadrics.data_stride_bytes);
				if (ComputeWedgeAttributes(wedge_quadric, weighted_new_position, attributes, state.rcp_attribute_weights, attribute_stride_dwords) && normalize_vertex_attributes) {
					normalize_vertex_attributes(attributes);
				}
			}
				
			if (changed_vertex_mask) {
				changed_vertex_mask[remaining_vertex_id.index] = 0xFF;
			}
			
			IterateCornerList<ElementType::Vertex>(mesh, mesh[remaining_vertex_id].corner_list_base, [&](CornerID corner_id) {
				u32 index = AttributeWedgeMapFind(state.wedge_attribute_set, mesh[corner_id].attributes_id);
				if (index != u32_max) mesh[corner_id].attributes_id = state.wedge_attributes_ids[index];
			});
		}
#endif // VGT_ENABLE_ATTRIBUTE_SUPPORT
	}
	
	HashTableClear(state.edge_duplicate_map);
	state.removed_edge_array.count = 0;
	
	return sqrtf(max_error) * state.rcp_position_weight;
}

static void DecimateMeshFaceGroups(
	MeshView mesh,
	Allocator& allocator,
	Allocator& heap_allocator,
	VgtNormalizeVertexAttributes normalize_vertex_attributes,
	float* attribute_weights,
	Array<FaceID> meshlet_group_faces,
	Array<u32> meshlet_group_face_prefix_sum,
	Array<VgtErrorMetric> meshlet_group_error_metrics,
	Array<u8> changed_vertex_mask) {
	
	u32 allocator_high_water = allocator.memory_block_count;
	u32 heap_allocator_high_water = heap_allocator.memory_block_count;
	
	MeshDecimationState state;
	InitializeMeshDecimationState(mesh, attribute_weights, allocator, heap_allocator, state);
	
	compile_const u32 vertex_group_index_locked = u32_max - 1;

	Array<u32> vertex_group_indices;
	ArrayResizeMemset(vertex_group_indices, allocator, mesh.vertex_count, 0xFF);
	
	{
		u32 begin_face_index = 0;
		for (u32 group_index = 0; group_index < meshlet_group_face_prefix_sum.count; group_index += 1) {
			u32 end_face_index = meshlet_group_face_prefix_sum[group_index];
			
			for (u32 face_index = begin_face_index; face_index < end_face_index; face_index += 1) {
				auto face_id = meshlet_group_faces[face_index];
				auto& face = mesh[face_id];
				VGT_ASSERT(face.corner_list_base.index != u32_max);
				
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
			
			begin_face_index = end_face_index;
		}
	}
	
	Array<EdgeID> meshlet_group_edge_ids;
	Array<u32> meshlet_group_edge_prefix_sum;
	u32 max_edge_count = 0;
	{
		ArrayResizeMemset(meshlet_group_edge_prefix_sum, allocator, meshlet_group_face_prefix_sum.count, 0);
		
		for (EdgeID edge_id = { 0 }; edge_id.index < mesh.edge_count; edge_id.index += 1) {
			auto& edge = mesh[edge_id];
			VGT_ASSERT(edge.corner_list_base.index != u32_max);
			
			u32 group_index_0 = vertex_group_indices[edge.vertex_0.index];
			u32 group_index_1 = vertex_group_indices[edge.vertex_1.index];
			
			// TODO: Allow edge collapses when only one vertex is locked.
			bool edge_is_locked = (group_index_0 == vertex_group_index_locked) || (group_index_1 == vertex_group_index_locked);
			VGT_ASSERT(edge_is_locked || group_index_0 == group_index_1);
			
			if (edge_is_locked == false) {
				meshlet_group_edge_prefix_sum[group_index_0] += 1;
			}
		}
		
		u32 prefix_sum = 0;
		for (u32 i = 0; i < meshlet_group_face_prefix_sum.count; i += 1) {
			u32 count = meshlet_group_edge_prefix_sum[i];
			max_edge_count = count > max_edge_count ? count : max_edge_count;
			
			meshlet_group_edge_prefix_sum[i] = prefix_sum;
			prefix_sum += count;
		}
		
		ArrayResize(meshlet_group_edge_ids, allocator, prefix_sum);
		for (EdgeID edge_id = { 0 }; edge_id.index < mesh.edge_count; edge_id.index += 1) {
			auto& edge = mesh[edge_id];
			VGT_ASSERT(edge.corner_list_base.index != u32_max);
			
			u32 group_index_0 = vertex_group_indices[edge.vertex_0.index];
			u32 group_index_1 = vertex_group_indices[edge.vertex_1.index];
			
			// TODO: Allow edge collapses when only one vertex is locked.
			bool edge_is_locked = (group_index_0 == vertex_group_index_locked) || (group_index_1 == vertex_group_index_locked);
			VGT_ASSERT(edge_is_locked || group_index_0 == group_index_1);
			
			if (edge_is_locked == false) {
				meshlet_group_edge_ids[meshlet_group_edge_prefix_sum[group_index_0]++] = edge_id;
			}
		}
	}
	
	
	EdgeCollapseHeap edge_collapse_heap;
	ArrayResize(edge_collapse_heap.edge_collapse_errors,  allocator, max_edge_count);
	ArrayResize(edge_collapse_heap.edge_id_to_heap_index, allocator, mesh.edge_count);
	ArrayResize(edge_collapse_heap.heap_index_to_edge_id, allocator, max_edge_count);
	
	u32 begin_edge_index = 0;
	u32 begin_face_index = 0;
	for (u32 group_index = 0; group_index < meshlet_group_face_prefix_sum.count; group_index += 1) {
		u32 end_edge_index = meshlet_group_edge_prefix_sum[group_index];
		u32 end_face_index = meshlet_group_face_prefix_sum[group_index];
		
		u32 edge_count = end_edge_index - begin_edge_index;
		u32 face_count = end_face_index - begin_face_index;
		
		
		edge_collapse_heap.edge_collapse_errors.count  = edge_count;
		edge_collapse_heap.edge_id_to_heap_index.count = mesh.edge_count;
		edge_collapse_heap.heap_index_to_edge_id.count = edge_count;
		
		memset(edge_collapse_heap.edge_id_to_heap_index.data, 0xFF, edge_collapse_heap.edge_id_to_heap_index.count * sizeof(u32));
		
		for (u32 edge_index = begin_edge_index; edge_index < end_edge_index; edge_index += 1) {
			auto edge_id = meshlet_group_edge_ids[edge_index];
			u32 local_edge_index = edge_index - begin_edge_index;
			
			auto collapse_error = ComputeEdgeCollapseError(mesh, heap_allocator, state, edge_id);
			
			edge_collapse_heap.edge_collapse_errors[local_edge_index]  = collapse_error.min_error;
			edge_collapse_heap.edge_id_to_heap_index[edge_id.index]    = local_edge_index;
			edge_collapse_heap.heap_index_to_edge_id[local_edge_index] = edge_id;
		}
		
		EdgeCollapseHeapInitialize(edge_collapse_heap);
		
		
		u32 target_face_count = face_count / 2;
		u32 active_face_count = face_count;
		float decimation_error = DecimateMeshFaceGroup(
			mesh,
			heap_allocator,
			state,
			edge_collapse_heap,
			normalize_vertex_attributes,
			changed_vertex_mask.data,
			target_face_count,
			active_face_count
		);
		
		auto& error_metric = meshlet_group_error_metrics[group_index];
		error_metric.error = error_metric.error > decimation_error ? error_metric.error : decimation_error;
		
		begin_edge_index = end_edge_index;
		begin_face_index = end_face_index;
	}
	
	AllocatorFreeMemoryBlocks(allocator, allocator_high_water);
	AllocatorFreeMemoryBlocks(heap_allocator, heap_allocator_high_water);
}

//
// KdTree implementation is based on [Kapoulkine 2025]. KdTree is used to accelerate spatial lookup queries during meshlet and meshlet group builds.
//
struct alignas(16) KdTreeElement {
	Vector3 position;
	
	// During meshlet build partition index is used either as geometry_index (is_active_element == 1) or meshlet_index (is_active_element == 0).
	// During meshlet group build partition index is used as meshlet_group_index (is_active_element == 0).
	u32 is_active_element : 1;
	u32 partition_index   : 31;
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

// TODO: Experiment with splitting elements by geometry index. This would allow us to move element filtering higher up the tree.
static u32 KdTreeSplit(const ArrayView<KdTreeElement>& elements, ArrayView<u32> indices, KdTreeNode& node) {
	Vector3 sum = { 0.f, 0.f, 0.f };
	Vector3 min = { +FLT_MAX, +FLT_MAX, +FLT_MAX };
	Vector3 max = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
	float indices_count = 0.f;
	
	for (u32 i = 0; i < indices.count; i += 1) {
		auto& element = elements[indices[i]];
		
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
		u32 index = indices[i];
		float position = elements[index].position[split_axis];

		// Swap(indices[i], indices[split_index]);
		indices[i] = indices[split_index];
		indices[split_index] = index;

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
	
	VGT_ASSERT(node_index_0 == node_index + 1); // Left node is always the next node after the local root.
	VGT_ASSERT(node_index_1 > node_index);      // Right node offset is non zero. Zero means branch is pruned.
	
	nodes[node_index].payload = node_index_1 - node_index;
	
	return node_index;
}

static void KdTreeBuild(KdTree& tree, Allocator& allocator) {
	ArrayResize(tree.element_indices, allocator, tree.elements.count);
	ArrayReserve(tree.nodes, allocator, tree.elements.count * 2);
	
	for (u32 i = 0; i < tree.element_indices.count; i += 1) {
		tree.element_indices[i] = i;
	}
	
	KdTreeBuildNode(tree.nodes, CreateArrayView(tree.elements), CreateArrayView(tree.element_indices));
}

#define COUNT_KD_TREE_LOOKUPS 0
#define COUNT_KD_TREE_NODE_VISITS 0

#if COUNT_KD_TREE_NODE_VISITS
u32 kd_tree_node_visits = 0;
#endif // COUNT_KD_TREE_NODE_VISITS

static bool KdTreeFindClosestActiveElement(KdTree& kd_tree, const Vector3& point, u32 geometry_index, u32& closest_index, float& min_distance, u32 root = 0) {
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
			
			// Element is already used, i.e. it's inactive for the sake of search.
			if (element.is_active_element == 0) continue;
			
			// Don't prune the branch if we have at least one active leaf element.
			should_prune = false;
			
			// Element is coming from a wrong geometry.
			if (geometry_index != u32_max && element.partition_index != geometry_index) continue;
			
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
		
		bool prune_lh = KdTreeFindClosestActiveElement(kd_tree, point, geometry_index, closest_index, min_distance, root + offset_0);
		bool prune_rh = false;
		
		if ((delta * delta) <= min_distance) {
			prune_rh = KdTreeFindClosestActiveElement(kd_tree, point, geometry_index, closest_index, min_distance, root + offset_1);
		} else {
			prune_rh = (kd_tree.nodes[root + offset_1].payload == 0);
		}
		should_prune = prune_lh && prune_rh;
	}
	
	if (should_prune) node.payload = 0;
	
	return should_prune;
}

static void KdTreeBuildElementsForFaces(MeshView mesh, Allocator& allocator, Array<KdTreeElement>& elements) {
	ArrayResize(elements, allocator, mesh.face_count);
	
	for (FaceID face_id = { 0 }; face_id.index < mesh.face_count; face_id.index += 1) {
		auto& element = elements[face_id.index];
		auto& face    = mesh[face_id];
		
		VGT_ASSERT(face.corner_list_base.index != u32_max);
		
		Vector3 position = { 0.f, 0.f, 0.f };
		float face_degree = 0.f;
		IterateCornerList<ElementType::Face>(mesh, face.corner_list_base, [&](CornerID corner_id) {
			position = position + mesh[mesh[corner_id].vertex_id].position;
			face_degree += 1.f;
		});
		
		element.position          = position * (1.f / face_degree);
		element.is_active_element = 0; // Face elements are inactive by default. We mark them as active per group when generating meshlets.
		element.partition_index   = face.geometry_index;
	}
}

static void KdTreeBuildElementsForMeshlets(ArrayView<VgtMeshlet> meshlets, Allocator& allocator, Array<KdTreeElement>& elements) {
	ArrayResize(elements, allocator, meshlets.count);
	
	for (u32 meshlet_index = 0; meshlet_index < meshlets.count; meshlet_index += 1) {
		auto& element = elements[meshlet_index];
		auto& meshlet = meshlets[meshlet_index];
		
		element.position = (meshlet.aabb_min + meshlet.aabb_max) * 0.5f;
		element.is_active_element = 1;
		element.partition_index   = 0;
	}
}

static VgtSphereBounds ComputeSphereBoundsUnion(ArrayView<VgtSphereBounds> source_sphere_bounds) {
	return VgtComputeSphereBoundsUnion(source_sphere_bounds.data, source_sphere_bounds.count);
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
	ArrayView<VgtMeshlet> meshlets;
	
	ArrayView<u8>       meshlet_triangles;
	ArrayView<CornerID> meshlet_corners;
	ArrayView<u32>      meshlet_corner_prefix_sum;
	
	MeshletAdjacency meshlet_adjacency;
};

static MeshletAdjacency BuildMeshletAdjacency(MeshView mesh, Allocator& allocator, ArrayView<u32> meshlet_face_prefix_sum, ArrayView<FaceID> meshlet_faces, ArrayView<KdTreeElement> kd_tree_elements);

//
// Based on [Kapoulkine 2025].
//
static void BuildMeshletsForFaceGroup(
	MeshView mesh,
	KdTree kd_tree,
	u32 meshlet_target_face_count,
	u32 meshlet_target_vertex_count,
	Array<u8> vertex_usage_map,
	Array<FaceID>& meshlet_faces,
	Array<u8>& meshlet_triangles,
	Array<u32>& meshlet_face_prefix_sum,
	Array<CornerID>& meshlet_corners,
	Array<u32>& meshlet_corner_prefix_sum) {
	
	compile_const u32 candidates_per_face = 4;
	FixedSizeArray<AttributesID, meshlet_max_face_count * meshlet_max_face_degree> meshlet_vertices;
	FixedSizeArray<FaceID, meshlet_max_face_count * candidates_per_face> meshlet_candidate_elements;
	
	auto meshlet_aabb_min = Vector3{ +FLT_MAX, +FLT_MAX, +FLT_MAX };
	auto meshlet_aabb_max = Vector3{ -FLT_MAX, -FLT_MAX, -FLT_MAX };
	u32  meshlet_vertex_count = 0;
	u32  meshlet_face_count   = 0;
	u32  meshlet_geometry_index = u32_max;
	
#if COUNT_KD_TREE_LOOKUPS
	u32 kd_tree_lookup_count = 0;
#endif // COUNT_KD_TREE_LOOKUPS
	
	while (true) {
		u32 best_candidate_face_index = u32_max;
		float smallest_distance_to_face = FLT_MAX;
		
		auto bounds_center = (meshlet_aabb_max + meshlet_aabb_min) * 0.5f;
		
		for (u32 i = 0; i < meshlet_candidate_elements.count;) {
			auto face_id = meshlet_candidate_elements[i];
			
			auto& element = kd_tree.elements[face_id.index];
			if (element.is_active_element == 0) {
				ArrayEraseSwap(meshlet_candidate_elements, i);
				continue;
			}
			
			auto bounds_center_to_face_center = (bounds_center - element.position);
			float distance_to_face = DotProduct(bounds_center_to_face_center, bounds_center_to_face_center);
			
			if (smallest_distance_to_face > distance_to_face) {
				smallest_distance_to_face = distance_to_face;
				best_candidate_face_index = i;
			}
			
			u32 new_vertex_count = 0;
			IterateCornerList<ElementType::Face>(mesh, mesh[face_id].corner_list_base, [&](CornerID corner_id) {
				auto& corner = mesh[corner_id];
				u8 vertex_index = vertex_usage_map[corner.attributes_id.index];
				if (vertex_index == 0xFF) new_vertex_count += 1;
			});
			
			if (new_vertex_count == 0) {
				best_candidate_face_index = i;
				break;
			}
			
			i += 1;
		}
		
		auto best_face_id = FaceID{ u32_max };
		if (best_candidate_face_index != u32_max) {
			best_face_id = meshlet_candidate_elements[best_candidate_face_index];
			ArrayEraseSwap(meshlet_candidate_elements, best_candidate_face_index);
		}
		
		bool restart_meshlet = false;
		bool kd_tree_is_empty = false;
		if (best_face_id.index == u32_max) {
			auto center = meshlet_face_count ? bounds_center : Vector3{ 0.f, 0.f, 0.f };
			
			float min_distance = FLT_MAX;
			kd_tree_is_empty = KdTreeFindClosestActiveElement(kd_tree, center, meshlet_geometry_index, best_face_id.index, min_distance);
			
			if (best_face_id.index != u32_max && meshlet_face_count >= meshlet_min_face_count) {
				auto& element = kd_tree.elements[best_face_id.index];
				
				auto new_aabb_min = VectorMin(meshlet_aabb_min, element.position);
				auto new_aabb_max = VectorMax(meshlet_aabb_max, element.position);
				auto new_aabb_extent = (new_aabb_max     - new_aabb_min);
				auto old_aabb_extent = (meshlet_aabb_max - meshlet_aabb_min);
				
				float new_radius = DotProduct(new_aabb_extent, new_aabb_extent);
				float old_radius = DotProduct(old_aabb_extent, old_aabb_extent);
				
				restart_meshlet = (new_radius > old_radius * discontinuous_meshlet_max_expansion);
			}
			
#if COUNT_KD_TREE_LOOKUPS
			kd_tree_lookup_count += 1;
#endif // COUNT_KD_TREE_LOOKUPS
		}
		
		if (best_face_id.index == u32_max && kd_tree_is_empty) {
			break;
		}
		
		
		u32 new_vertex_count = 0;
		if (best_face_id.index != u32_max) {
			IterateCornerList<ElementType::Face>(mesh, mesh[best_face_id].corner_list_base, [&](CornerID corner_id) {
				auto& corner = mesh[corner_id];
				u8 vertex_index = vertex_usage_map[corner.attributes_id.index];
				if (vertex_index == 0xFF) new_vertex_count += 1;
			});
		}
		
		if (restart_meshlet || best_face_id.index == u32_max || (meshlet_vertex_count + new_vertex_count > meshlet_target_vertex_count) || (meshlet_face_count + 1 > meshlet_target_face_count)) {
			VGT_ASSERT(meshlet_face_count   <= meshlet_target_face_count);
			VGT_ASSERT(meshlet_vertex_count <= meshlet_target_vertex_count);
			
			for (auto attributes_id : meshlet_vertices) {
				vertex_usage_map[attributes_id.index] = 0xFF;
			}
			meshlet_vertices.count = 0;
			
			meshlet_vertex_count = 0;
			meshlet_face_count   = 0;
			meshlet_geometry_index = u32_max;
			
			meshlet_aabb_min = Vector3{ +FLT_MAX, +FLT_MAX, +FLT_MAX };
			meshlet_aabb_max = Vector3{ -FLT_MAX, -FLT_MAX, -FLT_MAX };
			
			ArrayAppend(meshlet_face_prefix_sum, meshlet_faces.count);
			ArrayAppend(meshlet_corner_prefix_sum, meshlet_corners.count);
			
			meshlet_candidate_elements.count = 0;
		}
		
		if (best_face_id.index == u32_max) continue;
		
		u32 best_face_geometry_index = mesh[best_face_id].geometry_index;
		VGT_ASSERT(meshlet_face_count == 0 || meshlet_geometry_index == best_face_geometry_index);
		
		new_vertex_count = 0;
		IterateCornerList<ElementType::Face>(mesh, mesh[best_face_id].corner_list_base, [&](CornerID corner_id) {
			auto& corner = mesh[corner_id];
			u8 vertex_index = vertex_usage_map[corner.attributes_id.index];
			if (vertex_index == 0xFF) {
				vertex_index = meshlet_vertex_count + new_vertex_count;
				vertex_usage_map[corner.attributes_id.index] = vertex_index;
				
				new_vertex_count += 1;
				
				ArrayAppend(meshlet_vertices, corner.attributes_id);
				ArrayAppend(meshlet_corners, corner_id);
			}
			
			ArrayAppend(meshlet_triangles, vertex_index);
			
			IterateCornerList<ElementType::Vertex>(mesh, corner_id, [&](CornerID corner_id) {
				auto face_id = mesh[corner_id].face_id;
				auto& element = kd_tree.elements[face_id.index];
				
				if ((face_id.index != best_face_id.index) &&
					(element.is_active_element != 0) &&
					(element.partition_index   == best_face_geometry_index) && 
					(meshlet_candidate_elements.count < meshlet_candidate_elements.capacity)) {
					ArrayAppend(meshlet_candidate_elements, face_id);
				}
			});
		});
		
		ArrayAppend(meshlet_faces, best_face_id);
		meshlet_vertex_count += new_vertex_count;
		meshlet_face_count   += 1;
		
		auto& element = kd_tree.elements[best_face_id.index];
		element.partition_index   = meshlet_face_prefix_sum.count;
		element.is_active_element = 0;
		
		meshlet_geometry_index = best_face_geometry_index;
		
		meshlet_aabb_min = VectorMin(meshlet_aabb_min, element.position);
		meshlet_aabb_max = VectorMax(meshlet_aabb_max, element.position);
	}
	
	if (meshlet_face_count) {
		VGT_ASSERT(meshlet_face_count   <= meshlet_target_face_count);
		VGT_ASSERT(meshlet_vertex_count <= meshlet_target_vertex_count);
		
		for (auto attributes_id : meshlet_vertices) {
			vertex_usage_map[attributes_id.index] = 0xFF;
		}
		meshlet_vertices.count = 0;
		
		ArrayAppend(meshlet_face_prefix_sum, meshlet_faces.count);
		ArrayAppend(meshlet_corner_prefix_sum, meshlet_corners.count);
	}
	
#if COUNT_KD_TREE_NODE_VISITS
	printf("BuildMeshletsForFaceGroup: kd_tree_node_visits: %u\n", kd_tree_node_visits);
#endif // COUNT_KD_TREE_NODE_VISITS
	
#if COUNT_KD_TREE_LOOKUPS
	printf("BuildMeshletsForFaceGroup: kd_tree_lookup_count: %u\n", kd_tree_lookup_count);
#endif // COUNT_KD_TREE_LOOKUPS
}

// Note that FaceIDs inside groups are going to be scrambled inside groups during KdTree build. This leaves prefix sum in a valid, but different state.
static MeshletBuildResult BuildMeshletsForFaceGroups(
	MeshView mesh,
	Allocator& allocator,
	Array<FaceID> meshlet_group_faces,
	Array<u32> meshlet_group_face_prefix_sum,
	Array<VgtErrorMetric> meshlet_group_error_metrics,
	u32 meshlet_target_face_count,
	u32 meshlet_target_vertex_count,
	u32 meshlet_group_base_index) {
	
	KdTree kd_tree;
	KdTreeBuildElementsForFaces(mesh, allocator, kd_tree.elements);
	ArrayReserve(kd_tree.nodes, allocator, kd_tree.elements.count * 2);
	
	// Use meshlet group faces as source element indices to build meshlets in groups.
	// KdTree elements during meshlet generation are just faces, so element indices and FaceIDs are the same.
	kd_tree.element_indices.data     = (u32*)meshlet_group_faces.data;
	kd_tree.element_indices.count    = meshlet_group_faces.count;
	kd_tree.element_indices.capacity = meshlet_group_faces.capacity;
	static_assert(sizeof(FaceID) == sizeof(u32));
	VGT_ASSERT(kd_tree.elements.count == meshlet_group_faces.count);
	
	
	Array<u8> vertex_usage_map;
	ArrayResizeMemset(vertex_usage_map, allocator, mesh.attribute_count, 0xFF);
	
	Array<FaceID> meshlet_faces;
	Array<CornerID> meshlet_corners;
	Array<u8> meshlet_triangles;
	ArrayReserve(meshlet_faces, allocator, mesh.face_count);
	ArrayReserve(meshlet_corners, allocator, mesh.face_count * meshlet_max_face_degree);
	ArrayReserve(meshlet_triangles, allocator, mesh.face_count * meshlet_max_face_degree);
	
	Array<u32> meshlet_face_prefix_sum;
	Array<u32> meshlet_corner_prefix_sum;
	ArrayReserve(meshlet_face_prefix_sum, allocator, mesh.face_count);
	ArrayReserve(meshlet_corner_prefix_sum, allocator, mesh.face_count);
	
	Array<u32> meshlet_group_meshlet_prefix_sum;
	ArrayReserve(meshlet_group_meshlet_prefix_sum, allocator, meshlet_group_face_prefix_sum.count);
	
	u32 begin_element_index = 0;
	for (u32 group_index = 0; group_index < meshlet_group_face_prefix_sum.count; group_index += 1) {
		u32 end_element_index = meshlet_group_face_prefix_sum[group_index];
		
		kd_tree.nodes.count = 0;
		auto element_indices = CreateArrayView(kd_tree.element_indices, begin_element_index, end_element_index);
		
		for (u32 index : element_indices) {
			// Mark elements of the current group as eligible for meshlet generation.
			kd_tree.elements[index].is_active_element = 1;
		}
		
		KdTreeBuildNode(kd_tree.nodes, CreateArrayView(kd_tree.elements), element_indices);
		
		BuildMeshletsForFaceGroup(
			mesh,
			kd_tree,
			meshlet_target_face_count,
			meshlet_target_vertex_count,
			vertex_usage_map,
			meshlet_faces,
			meshlet_triangles,
			meshlet_face_prefix_sum,
			meshlet_corners,
			meshlet_corner_prefix_sum
		);
		
		ArrayAppend(meshlet_group_meshlet_prefix_sum, meshlet_face_prefix_sum.count);
		
		VGT_ASSERT(meshlet_faces.count == end_element_index);
		
		begin_element_index = end_element_index;
	}
	
	VGT_ASSERT(meshlet_faces.count == mesh.face_count);
	
	
	Array<VgtMeshlet> meshlets;
	ArrayResize(meshlets, allocator, meshlet_corner_prefix_sum.count);
	
	u32 begin_corner_index = 0;
	for (u32 meshlet_index = 0; meshlet_index < meshlet_corner_prefix_sum.count; meshlet_index += 1) {
		u32 end_corner_index = meshlet_corner_prefix_sum[meshlet_index];
		
		auto meshlet_aabb_min = Vector3{ +FLT_MAX, +FLT_MAX, +FLT_MAX };
		auto meshlet_aabb_max = Vector3{ -FLT_MAX, -FLT_MAX, -FLT_MAX };
		
		FixedSizeArray<VgtSphereBounds, meshlet_max_vertex_count> vertex_sphere_bounds;
		VGT_ASSERT(end_corner_index - begin_corner_index <= meshlet_max_vertex_count);
		
		for (u32 corner_index = begin_corner_index; corner_index < end_corner_index; corner_index += 1) {
			auto corner_id = meshlet_corners[corner_index];
			auto vertex_id = mesh[corner_id].vertex_id;
			
			auto position = mesh[vertex_id].position;
			meshlet_aabb_min = VectorMin(meshlet_aabb_min, position);
			meshlet_aabb_max = VectorMax(meshlet_aabb_max, position);
			
			VgtSphereBounds bounds;
			bounds.center = position;
			bounds.radius = 0.f;
			ArrayAppend(vertex_sphere_bounds, bounds);
		}
		
		auto& meshlet = meshlets[meshlet_index];
		meshlet.aabb_min = meshlet_aabb_min;
		meshlet.aabb_max = meshlet_aabb_max;
		
		auto meshlet_sphere_bounds = ComputeSphereBoundsUnion(CreateArrayView(vertex_sphere_bounds));
		meshlet.geometric_sphere_bounds = meshlet_sphere_bounds;
		
		// Will be overridden in the loop below if we have source meshlet_group_error_metrics.
		// For the level zero we don't have them, so this error metric will be kept as is.
		meshlet.current_level_error_metric.bounds = meshlet_sphere_bounds;
		meshlet.current_level_error_metric.error  = 0.f;
		meshlet.current_level_meshlet_group_index = u32_max;
		
		// All meshlets faces are guaranteed to come from the same geometry.
		meshlet.geometry_index = mesh[mesh[meshlet_corners[begin_corner_index]].face_id].geometry_index;
		
		begin_corner_index = end_corner_index;
	}
	
	u32 meshlet_begin_index = 0;
	for (u32 group_index = 0; group_index < meshlet_group_error_metrics.count; group_index += 1) {
		u32 meshlet_end_index = meshlet_group_meshlet_prefix_sum[group_index];
		
		auto source_meshlet_group_error_metric = meshlet_group_error_metrics[group_index];
		for (u32 meshlet_index = meshlet_begin_index; meshlet_index < meshlet_end_index; meshlet_index += 1) {
			auto& meshlet = meshlets[meshlet_index];
			meshlet.current_level_error_metric        = source_meshlet_group_error_metric;
			meshlet.current_level_meshlet_group_index = group_index + meshlet_group_base_index;
		}

		meshlet_begin_index = meshlet_end_index;
	}
	
	
	MeshletBuildResult result;
	result.meshlet_faces             = CreateArrayView(meshlet_faces);
	result.meshlet_face_prefix_sum   = CreateArrayView(meshlet_face_prefix_sum);
	result.meshlets                  = CreateArrayView(meshlets);
	result.meshlet_triangles         = CreateArrayView(meshlet_triangles);
	result.meshlet_corners           = CreateArrayView(meshlet_corners);
	result.meshlet_corner_prefix_sum = CreateArrayView(meshlet_corner_prefix_sum);
	result.meshlet_adjacency         = BuildMeshletAdjacency(mesh, allocator, result.meshlet_face_prefix_sum, result.meshlet_faces, CreateArrayView(kd_tree.elements));
	
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
	
	u32 begin_face_index = 0;
	for (u32 meshlet_index = 0; meshlet_index < meshlet_face_prefix_sum.count; meshlet_index += 1) {
		u32 end_face_index = meshlet_face_prefix_sum[meshlet_index];
		
		// At least reserve one meshlet per face edge. Do this upfront instead of adding code in the inner loop to improve performance.
		compile_const u32 reserve_size = meshlet_max_face_count * meshlet_max_face_degree;
		if (meshlet_adjacency_infos.count + reserve_size >= meshlet_adjacency_infos.capacity) {
			ArrayReserve(meshlet_adjacency_infos, allocator, ArrayComputeNewCapacity(meshlet_adjacency_infos.capacity, meshlet_adjacency_infos.capacity + reserve_size));
		}
		
		u32 begin_adjacency_info_index = meshlet_adjacency_infos.count;
		for (u32 face_index = begin_face_index; face_index < end_face_index; face_index += 1) {
			auto face_id = meshlet_faces[face_index];
			
			IterateCornerList<ElementType::Face>(mesh, mesh[face_id].corner_list_base, [&](CornerID corner_id) {
				IterateCornerList<ElementType::Edge>(mesh, corner_id, [&](CornerID corner_id) {
					auto other_face_id = mesh[corner_id].face_id;
					u32 other_meshlet_index = kd_tree_elements[other_face_id.index].partition_index;
					if (other_meshlet_index == meshlet_index) return;
					
					VGT_ASSERT(kd_tree_elements[other_face_id.index].is_active_element == 0); // Face isn't a part of any meshlet.
					
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
		
		begin_face_index = end_face_index;
	}
	
	MeshletAdjacency meshlet_adjacency;
	meshlet_adjacency.prefix_sum = CreateArrayView(meshlet_adjacency_prefix_sum);
	meshlet_adjacency.infos      = CreateArrayView(meshlet_adjacency_infos);
	
	return meshlet_adjacency;
}

// Compute a fraction of edges that is shared with the target meshlet group for a given meshlet.
static float CountMeshletGroupSharedEdges(MeshletAdjacency meshlet_adjacency, Array<KdTreeElement> kd_tree_elements, u32 meshlet_index, u32 targent_group_index) {
	u32 meshlet_begin_index = meshlet_index > 0 ? meshlet_adjacency.prefix_sum[meshlet_index - 1] : 0;
	u32 meshlet_end_index   = meshlet_adjacency.prefix_sum[meshlet_index];
	
	u32 shared_edge_count = 0;
	u32 total_edge_count  = 0;
	for (u32 adjacency_info_index = meshlet_begin_index; adjacency_info_index < meshlet_end_index; adjacency_info_index += 1) {
		auto adjacency_info = meshlet_adjacency.infos[adjacency_info_index];
		
		auto& element = kd_tree_elements[adjacency_info.meshlet_index];
		total_edge_count += adjacency_info.shared_edge_count;
		
		if (element.is_active_element == 0 && element.partition_index == targent_group_index) {
			shared_edge_count += adjacency_info.shared_edge_count;
		}
	}
	
	return total_edge_count ? (float)shared_edge_count / (float)total_edge_count : 0.f;
}

struct MeshletGroupBuildResult {
	ArrayView<u32> meshlet_indices;
	ArrayView<u32> prefix_sum;
};

static MeshletGroupBuildResult BuildMeshletGroups(MeshView mesh, Allocator& allocator, ArrayView<VgtMeshlet> meshlets, MeshletAdjacency meshlet_adjacency, ArrayView<u32> meshlet_face_prefix_sum) {
	KdTree kd_tree;
	KdTreeBuildElementsForMeshlets(meshlets, allocator, kd_tree.elements);
	KdTreeBuild(kd_tree, allocator);
	
	FixedSizeArray<u32, meshlet_group_max_meshlet_count> meshlet_group;
	
	Array<u32> meshlet_group_meshlet_indices;
	Array<u32> meshlet_group_prefix_sum;
	ArrayReserve(meshlet_group_meshlet_indices, allocator, meshlets.count);
	ArrayReserve(meshlet_group_prefix_sum, allocator, (meshlets.count + meshlet_group_min_meshlet_count - 1) / meshlet_group_min_meshlet_count);
	
	auto meshlet_group_aabb_min = Vector3{ +FLT_MAX, +FLT_MAX, +FLT_MAX };
	auto meshlet_group_aabb_max = Vector3{ -FLT_MAX, -FLT_MAX, -FLT_MAX };
	
#if COUNT_KD_TREE_LOOKUPS
	u32 kd_tree_lookup_count = 0;
#endif // COUNT_KD_TREE_LOOKUPS
	
	while (true) {
		u32 best_candidate_meshlet_index = u32_max;
		float max_shared_edge_count = 0.f;
		
		for (u32 i = 0; i < meshlet_group.count; i += 1) {
			u32 meshlet_index = meshlet_group[i];
			
			u32 meshlet_begin_index = meshlet_index > 0 ? meshlet_adjacency.prefix_sum[meshlet_index - 1] : 0;
			u32 meshlet_end_index   = meshlet_adjacency.prefix_sum[meshlet_index];
			
			for (u32 adjacency_info_index = meshlet_begin_index; adjacency_info_index < meshlet_end_index; adjacency_info_index += 1) {
				auto adjacency_info = meshlet_adjacency.infos[adjacency_info_index];
				
				auto& element = kd_tree.elements[adjacency_info.meshlet_index];
				if (element.is_active_element == 0) continue; // Meshlet is already assigned to a group.
				
				//
				// Use the fraction of shared edges as the heuristic for grouping meshlets. This ensures that mesh decimation
				// can collapse as many edges as possible. As a side effect this helps to alter group boundaries across levels.
				//
				// For reference see [Karis 2021].
				//
				float shared_edge_count = CountMeshletGroupSharedEdges(meshlet_adjacency, kd_tree.elements, adjacency_info.meshlet_index, meshlet_group_prefix_sum.count);
				
				u32 begin_face_index = adjacency_info.meshlet_index ? meshlet_face_prefix_sum[adjacency_info.meshlet_index - 1] : 0;
				u32 end_face_index   = meshlet_face_prefix_sum[adjacency_info.meshlet_index];

				if (max_shared_edge_count < shared_edge_count) {
					max_shared_edge_count = shared_edge_count;
					best_candidate_meshlet_index = adjacency_info.meshlet_index;
				}
			}
		}
		
		bool restart_meshlet_group = false;
		if (best_candidate_meshlet_index == u32_max) {
			auto center = meshlet_group.count ? (meshlet_group_aabb_max + meshlet_group_aabb_min) * 0.5f : Vector3{ 0.f, 0.f, 0.f };
			
			float min_distance = FLT_MAX;
			KdTreeFindClosestActiveElement(kd_tree, center, u32_max, best_candidate_meshlet_index, min_distance);
			
			if (best_candidate_meshlet_index != u32_max && meshlet_group.count >= meshlet_group_min_meshlet_count) {
				auto& element = kd_tree.elements[best_candidate_meshlet_index];
				
				auto new_aabb_min = VectorMin(meshlet_group_aabb_min, element.position);
				auto new_aabb_max = VectorMax(meshlet_group_aabb_max, element.position);
				auto new_aabb_extent = (new_aabb_max           - new_aabb_min);
				auto old_aabb_extent = (meshlet_group_aabb_max - meshlet_group_aabb_min);
				
				float new_radius = DotProduct(new_aabb_extent, new_aabb_extent);
				float old_radius = DotProduct(old_aabb_extent, old_aabb_extent);
				
				restart_meshlet_group = (new_radius > old_radius * discontinuous_meshlet_group_max_expansion);
			}
			
#if COUNT_KD_TREE_LOOKUPS
			kd_tree_lookup_count += 1;
#endif // COUNT_KD_TREE_LOOKUPS
		}
		
		if (best_candidate_meshlet_index == u32_max) {
			break;
		}
		
		if (restart_meshlet_group || meshlet_group.count >= meshlet_group.capacity) {
			meshlet_group_aabb_min = Vector3{ +FLT_MAX, +FLT_MAX, +FLT_MAX };
			meshlet_group_aabb_max = Vector3{ -FLT_MAX, -FLT_MAX, -FLT_MAX };
			
			ArrayAppend(meshlet_group_prefix_sum, meshlet_group_meshlet_indices.count);
			meshlet_group.count = 0;
		}
		
		ArrayAppend(meshlet_group_meshlet_indices, best_candidate_meshlet_index);
		ArrayAppend(meshlet_group, best_candidate_meshlet_index);
		
		auto& element = kd_tree.elements[best_candidate_meshlet_index];
		element.partition_index   = meshlet_group_prefix_sum.count;
		element.is_active_element = 0;
		
		meshlet_group_aabb_min = VectorMin(meshlet_group_aabb_min, element.position);
		meshlet_group_aabb_max = VectorMax(meshlet_group_aabb_max, element.position);
	}
	
	if (meshlet_group.count) {
		meshlet_group.count = 0;
		ArrayAppend(meshlet_group_prefix_sum, meshlet_group_meshlet_indices.count);
	}
	
	VGT_ASSERT(meshlet_group_meshlet_indices.count == meshlets.count);
	
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

static void ConvertMeshletGroupsToFaceGroups(
	MeshView mesh,
	Allocator& allocator,
	MeshletBuildResult meshlet_build_result,
	MeshletGroupBuildResult meshlet_group_build_result,
	Array<FaceID>& meshlet_group_faces,
	Array<u32>& meshlet_group_face_prefix_sum,
	Array<VgtErrorMetric>& meshlet_group_error_metrics) {
	
	VGT_ASSERT(meshlet_group_faces.capacity           >= mesh.face_count);
	VGT_ASSERT(meshlet_group_face_prefix_sum.capacity >= mesh.face_count);
	VGT_ASSERT(meshlet_group_error_metrics.capacity   >= mesh.face_count);
	
	meshlet_group_faces.count           = 0;
	meshlet_group_face_prefix_sum.count = 0;
	meshlet_group_error_metrics.count   = 0;
	
	u32 group_meshlet_begin_index = 0;
	for (u32 group_index = 0; group_index < meshlet_group_build_result.prefix_sum.count; group_index += 1) {
		u32 group_meshlet_end_index = meshlet_group_build_result.prefix_sum[group_index];
		
		FixedSizeArray<VgtSphereBounds, meshlet_group_max_meshlet_count> meshlet_error_sphere_bounds;
		float max_error = 0.f;
		
		for (u32 group_meshlet_index = group_meshlet_begin_index; group_meshlet_index < group_meshlet_end_index; group_meshlet_index += 1) {
			u32 meshlet_index = meshlet_group_build_result.meshlet_indices[group_meshlet_index];
			
			u32 begin_face_index = meshlet_index > 0 ? meshlet_build_result.meshlet_face_prefix_sum[meshlet_index - 1] : 0;
			u32 end_face_index   = meshlet_build_result.meshlet_face_prefix_sum[meshlet_index];
			for (u32 face_index = begin_face_index; face_index < end_face_index; face_index += 1) {
				ArrayAppend(meshlet_group_faces, meshlet_build_result.meshlet_faces[face_index]);
			}
			
			auto& meshlet = meshlet_build_result.meshlets[meshlet_index];
			ArrayAppend(meshlet_error_sphere_bounds, meshlet.current_level_error_metric.bounds);
			max_error = meshlet.current_level_error_metric.error > max_error ? meshlet.current_level_error_metric.error : max_error;
		}
		ArrayAppend(meshlet_group_face_prefix_sum, meshlet_group_faces.count);
		
		//
		// Coarser level meshlets should have at least the same error as their children (finer representation).
		// This error metric might get increased during meshlet group decimation.
		//
		// See [Karis 2021] for reference on monotonic error metric.
		//
		VgtErrorMetric meshlet_group_minimum_error_metric;
		meshlet_group_minimum_error_metric.bounds = ComputeSphereBoundsUnion(CreateArrayView(meshlet_error_sphere_bounds));
		meshlet_group_minimum_error_metric.error  = max_error;
		ArrayAppend(meshlet_group_error_metrics, meshlet_group_minimum_error_metric);
		
		group_meshlet_begin_index = group_meshlet_end_index;
	}
}

static void BuildMeshletsAndMeshletGroups(Allocator& heap_allocator, MeshletBuildResult meshlet_build_result, MeshletGroupBuildResult meshlet_group_build_result, Array<VgtErrorMetric> meshlet_group_error_metrics, Array<VgtMeshletGroup>& meshlet_groups, Array<VgtMeshlet>& meshlets, u32 level_index) {
	if (meshlet_groups.capacity == 0) {
		ArrayReserve(meshlet_groups, heap_allocator, meshlet_group_build_result.prefix_sum.count * 4);
	}
	
	if (meshlets.capacity == 0) {
		ArrayReserve(meshlets, heap_allocator, meshlet_build_result.meshlets.count * 4);
	}
	
	u32 meshlet_group_base_index = meshlet_groups.count;
	
	u32 group_meshlet_begin_index = 0;
	for (u32 group_index = 0; group_index < meshlet_group_build_result.prefix_sum.count; group_index += 1) {
		u32 group_meshlet_end_index = meshlet_group_build_result.prefix_sum[group_index];
		
		auto meshlet_group_aabb_min = Vector3{ +FLT_MAX, +FLT_MAX, +FLT_MAX };
		auto meshlet_group_aabb_max = Vector3{ -FLT_MAX, -FLT_MAX, -FLT_MAX };
		
		FixedSizeArray<VgtSphereBounds, meshlet_group_max_meshlet_count> meshlet_sphere_bounds;
		auto meshlet_group_error_metric = meshlet_group_error_metrics[group_index];
		
		u32 begin_meshlet_index = meshlets.count;
		for (u32 group_meshlet_index = group_meshlet_begin_index; group_meshlet_index < group_meshlet_end_index; group_meshlet_index += 1) {
			u32 meshlet_index = meshlet_group_build_result.meshlet_indices[group_meshlet_index];
			
			auto& meshlet = meshlet_build_result.meshlets[meshlet_index];
			meshlet.coarser_level_error_metric        = meshlet_group_error_metric;
			meshlet.coarser_level_meshlet_group_index = group_index + meshlet_group_base_index;
			
			meshlet_group_aabb_min = VectorMin(meshlet_group_aabb_min, meshlet.aabb_min);
			meshlet_group_aabb_max = VectorMax(meshlet_group_aabb_max, meshlet.aabb_max);
			
			ArrayAppend(meshlet_sphere_bounds, meshlet.geometric_sphere_bounds);
			
			ArrayAppendMaybeGrow(meshlets, heap_allocator, meshlet);
		}
		u32 end_meshlet_index = meshlets.count;
		
		VgtMeshletGroup meshlet_group;
		meshlet_group.aabb_min = meshlet_group_aabb_min;
		meshlet_group.aabb_max = meshlet_group_aabb_max;
		
		meshlet_group.geometric_sphere_bounds = ComputeSphereBoundsUnion(CreateArrayView(meshlet_sphere_bounds));
		meshlet_group.error_metric            = meshlet_group_error_metric;
		meshlet_group.begin_meshlet_index     = begin_meshlet_index;
		meshlet_group.end_meshlet_index       = end_meshlet_index;
		meshlet_group.level_of_detail_index   = level_index;
		
		ArrayAppendMaybeGrow(meshlet_groups, heap_allocator, meshlet_group);
		
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

struct MeshElementRemap {
	Array<FaceID> old_face_id_to_new_face_id;
	Array<EdgeID> old_edge_id_to_new_edge_id;
};

static MeshElementRemap CompactMesh(MeshView& mesh, Allocator& allocator) {
	Array<FaceID> old_face_id_to_new_face_id;
	ArrayResize(old_face_id_to_new_face_id, allocator, mesh.face_count);
	
	Array<EdgeID> old_edge_id_to_new_edge_id;
	ArrayResize(old_edge_id_to_new_edge_id, allocator, mesh.edge_count);
	
	mesh.face_count = CreateMeshElementRemap<FaceID>(mesh, old_face_id_to_new_face_id);
	mesh.edge_count = CreateMeshElementRemap<EdgeID>(mesh, old_edge_id_to_new_edge_id);
	
	// Remap mesh corners.
	for (u32 i = 0; i < mesh.corner_count; i += 1) {
		auto& corner = mesh.corners[i];
		if (corner.face_id.index != u32_max) {
			corner.face_id = old_face_id_to_new_face_id[corner.face_id.index];
		}
		
		if (corner.edge_id.index != u32_max) {
			corner.edge_id = old_edge_id_to_new_edge_id[corner.edge_id.index];
		}
	}
	
	MeshElementRemap remap;
	remap.old_face_id_to_new_face_id = old_face_id_to_new_face_id;
	remap.old_edge_id_to_new_edge_id = old_edge_id_to_new_edge_id;

	return remap;
}

static void CompactMeshletGroupFaces(MeshElementRemap remap, Array<FaceID>& meshlet_group_faces, Array<u32>& meshlet_group_face_prefix_sum) {
	auto old_face_id_to_new_face_id = remap.old_face_id_to_new_face_id;
	
	u32 new_prefix_sum = 0;
	
	u32 begin_face_index = 0;
	for (u32 group_index = 0; group_index < meshlet_group_face_prefix_sum.count; group_index += 1) {
		u32 end_face_index = meshlet_group_face_prefix_sum[group_index];
		
		for (u32 face_index = begin_face_index; face_index < end_face_index; face_index += 1) {
			auto old_face_id = meshlet_group_faces[face_index];
			auto new_face_id = old_face_id_to_new_face_id[old_face_id.index];
			if (new_face_id.index == u32_max) continue;
			
			meshlet_group_faces[new_prefix_sum++] = new_face_id;
		}
		meshlet_group_face_prefix_sum[group_index] = new_prefix_sum;
		
		begin_face_index = end_face_index;
	}
	
	meshlet_group_faces.count = new_prefix_sum;
}

// TODO: We could write this data directly into the output buffers from BuildMeshletsForFaceGroups.
static void BuildMeshletVertexAndIndexBuffers(
	MeshView mesh,
	Allocator& heap_allocator,
	MeshletBuildResult meshlet_build_result,
	Array<u32> attributes_id_to_vertex_index,
	Array<u32>& meshlet_vertex_indices,
	Array<VgtMeshletTriangle>& meshlet_triangles) {
	
	auto meshlet_corner_prefix_sum = meshlet_build_result.meshlet_corner_prefix_sum;
	
	if (meshlet_vertex_indices.count + meshlet_build_result.meshlet_corners.count > meshlet_vertex_indices.capacity) {
		ArrayGrow(meshlet_vertex_indices, heap_allocator, ArrayComputeNewCapacity(meshlet_vertex_indices.capacity, meshlet_vertex_indices.count + meshlet_build_result.meshlet_corners.count));
	}
	
	u32 new_meshlets_triangle_count = (meshlet_build_result.meshlet_triangles.count / 3);
	if (meshlet_triangles.count + new_meshlets_triangle_count > meshlet_triangles.capacity) {
		ArrayGrow(meshlet_triangles, heap_allocator, ArrayComputeNewCapacity(meshlet_triangles.capacity, meshlet_triangles.count + new_meshlets_triangle_count));
	}
	
	u32 begin_corner_index = 0;
	for (u32 meshlet_index = 0; meshlet_index < meshlet_corner_prefix_sum.count; meshlet_index += 1) {
		u32 end_corner_index = meshlet_corner_prefix_sum[meshlet_index];
		
		u32 begin_vertex_indices_index = meshlet_vertex_indices.count;
		for (u32 corner_index = begin_corner_index; corner_index < end_corner_index; corner_index += 1) {
			auto corner_id = meshlet_build_result.meshlet_corners[corner_index];
			auto attributes_id = mesh[corner_id].attributes_id;
			
			u32 vertex_index = attributes_id_to_vertex_index[attributes_id.index];
			ArrayAppend(meshlet_vertex_indices, vertex_index);
		}
		u32 end_vertex_indices_index = meshlet_vertex_indices.count;

		auto& meshlet = meshlet_build_result.meshlets[meshlet_index];
		meshlet.begin_vertex_indices_index = begin_vertex_indices_index;
		meshlet.end_vertex_indices_index   = end_vertex_indices_index;
		
		begin_corner_index = end_corner_index;
	}
	
	auto meshlet_face_prefix_sum = meshlet_build_result.meshlet_face_prefix_sum;
	
	u32 begin_face_index = 0;
	for (u32 meshlet_index = 0; meshlet_index < meshlet_face_prefix_sum.count; meshlet_index += 1) {
		u32 end_face_index = meshlet_face_prefix_sum[meshlet_index];
		
		u32 begin_meshlet_triangles_index = meshlet_triangles.count;
		for (u32 face_index = begin_face_index; face_index < end_face_index; face_index += 1) {
			VgtMeshletTriangle triangle;
			triangle.i0 = meshlet_build_result.meshlet_triangles[face_index * 3 + 0];
			triangle.i1 = meshlet_build_result.meshlet_triangles[face_index * 3 + 1];
			triangle.i2 = meshlet_build_result.meshlet_triangles[face_index * 3 + 2];

			ArrayAppend(meshlet_triangles, triangle);
		}
		u32 end_meshlet_triangles_index = meshlet_triangles.count;
		
		auto& meshlet = meshlet_build_result.meshlets[meshlet_index];
		meshlet.begin_meshlet_triangles_index = begin_meshlet_triangles_index;
		meshlet.end_meshlet_triangles_index   = end_meshlet_triangles_index;
		
		begin_face_index = end_face_index;
	}
}

static void AppendChangedVertices(
	MeshView mesh,
	Allocator& allocator,
	Allocator& heap_allocator,
	Array<u8> changed_vertex_mask,
	Array<u32> attributes_id_to_vertex_index,
	Array<float>& vertices) {
	
	Array<VertexID> attributes_id_to_vertex_id;
	ArrayResizeMemset(attributes_id_to_vertex_id, allocator, mesh.attribute_count, 0xFF);
	
	for (VertexID vertex_id = { 0 }; vertex_id.index < mesh.vertex_count; vertex_id.index += 1) {
		auto& vertex = mesh[vertex_id];
		if (vertex.corner_list_base.index == u32_max)  continue;
		if (changed_vertex_mask[vertex_id.index] == 0) continue;
		
		IterateCornerList<ElementType::Vertex>(mesh, vertex.corner_list_base, [&](CornerID corner_id) {
			auto attributes_ids = mesh[corner_id].attributes_id;
			attributes_id_to_vertex_id[attributes_ids.index] = vertex_id;
		});
	}
	memset(changed_vertex_mask.data, 0, changed_vertex_mask.count);

	u32 changed_vertex_count = 0;
	for (AttributesID attributes_id = { 0 }; attributes_id.index < mesh.attribute_count; attributes_id.index += 1) {
		auto vertex_id = attributes_id_to_vertex_id[attributes_id.index];
		changed_vertex_count += vertex_id.index != u32_max ? 1 : 0;
	}
	
	u32 attribute_stride_dwords = mesh.attribute_stride_dwords;
	u32 vertex_stride_dwords    = (attribute_stride_dwords + 3);
	u32 new_array_size          = vertices.count + changed_vertex_count * vertex_stride_dwords;
	
	if (new_array_size > vertices.capacity) {
		ArrayGrow(vertices, heap_allocator, new_array_size * 3 / 2);
	}
	
	u32 output_vertex_index = vertices.count / vertex_stride_dwords;
	vertices.count = new_array_size;
	
	for (AttributesID attributes_id = { 0 }; attributes_id.index < mesh.attribute_count; attributes_id.index += 1) {
		auto vertex_id = attributes_id_to_vertex_id[attributes_id.index];
		if (vertex_id.index == u32_max) continue;
		
		attributes_id_to_vertex_index[attributes_id.index] = output_vertex_index;
		
		auto* vertex = &vertices[output_vertex_index * vertex_stride_dwords];
		memcpy(vertex + 0, &mesh[vertex_id].position, sizeof(Vector3));
		memcpy(vertex + 3, mesh[attributes_id], attribute_stride_dwords * sizeof(u32));
		output_vertex_index += 1;
	}

	VGT_ASSERT(vertices.count == output_vertex_index * vertex_stride_dwords);
}

} // namespace VirtualGeometryTools


void VgtBuildVirtualGeometry(const VgtVirtualGeometryBuildInputs* inputs, VgtVirtualGeometryBuildResult* result, const VgtSystemCallbacks* callbacks) {
	using namespace VirtualGeometryTools;
	
	VGT_ASSERT(inputs);
	VGT_ASSERT(result);
	
	Allocator allocator;
	InitializeAllocator(allocator, callbacks ? &callbacks->temp_allocator : nullptr);

	Allocator heap_allocator;
	InitializeAllocator(heap_allocator, callbacks ? &callbacks->heap_allocator : nullptr);
	
	auto mesh = BuildEditableMesh(allocator, inputs->mesh.geometry_descs, inputs->mesh.geometry_desc_count, inputs->mesh.vertex_stride_bytes);
	
	u32 meshlet_target_face_count   = Clamp(inputs->meshlet_target_triangle_count, 1u, meshlet_max_face_count);
	u32 meshlet_target_vertex_count = Clamp(inputs->meshlet_target_vertex_count, 3u, meshlet_max_vertex_count);
	
	Array<FaceID> meshlet_group_faces;
	Array<u32> meshlet_group_face_prefix_sum;
	Array<VgtErrorMetric> meshlet_group_error_metrics;
	ArrayResize(meshlet_group_faces, allocator, mesh.face_count);
	ArrayReserve(meshlet_group_face_prefix_sum, allocator, mesh.face_count);
	ArrayReserve(meshlet_group_error_metrics, allocator, mesh.face_count);
	
	// Note that we're not adding an initial error metric to meshlet_group_error_metrics.
	// Meshlets built from the first group will use current_level_error_metric.bounds set to the geometric_sphere_bounds with zero error.
	for (u32 i = 0; i < meshlet_group_faces.count; i += 1) {
		meshlet_group_faces[i].index = i;
	}
	ArrayAppend(meshlet_group_face_prefix_sum, meshlet_group_faces.count);
	
	u32 meshlet_group_base_index = 0;
	u32 last_level_meshlet_count = u32_max;
	
	
	Array<u32> attributes_id_to_vertex_index;
	ArrayResizeMemset(attributes_id_to_vertex_index, allocator, mesh.attribute_count, 0xFF);
	
	Array<u8> changed_vertex_mask;
	ArrayResizeMemset(changed_vertex_mask, allocator, mesh.attribute_count, 0xFF);
	
	Array<float> vertices;
	AppendChangedVertices(mesh, allocator, heap_allocator, changed_vertex_mask, attributes_id_to_vertex_index, vertices);
	
	Array<VgtVirtualGeometryLevel> levels;
	ArrayReserve(levels, heap_allocator, virtual_geometry_max_levels_of_details);
	
	Array<VgtMeshletGroup> meshlet_groups;
	Array<VgtMeshlet>      meshlets;
	
	Array<u32> meshlet_vertex_indices;
	Array<VgtMeshletTriangle> meshlet_triangles;
	
	for (u32 level_index = 0; level_index < virtual_geometry_max_levels_of_details; level_index += 1) {
		u32 allocator_high_water = allocator.memory_block_count;
		
		auto meshlet_build_result = BuildMeshletsForFaceGroups(mesh, allocator, meshlet_group_faces, meshlet_group_face_prefix_sum, meshlet_group_error_metrics, meshlet_target_face_count, meshlet_target_vertex_count, meshlet_group_base_index);
		BuildMeshletVertexAndIndexBuffers(mesh, heap_allocator, meshlet_build_result, attributes_id_to_vertex_index, meshlet_vertex_indices, meshlet_triangles);
		
		auto meshlet_group_build_result = BuildMeshletGroups(mesh, allocator, meshlet_build_result.meshlets, meshlet_build_result.meshlet_adjacency, meshlet_build_result.meshlet_face_prefix_sum);
		ConvertMeshletGroupsToFaceGroups(mesh, allocator, meshlet_build_result, meshlet_group_build_result, meshlet_group_faces, meshlet_group_face_prefix_sum, meshlet_group_error_metrics);
		
		bool is_last_level = (level_index + 1 == virtual_geometry_max_levels_of_details) || (mesh.face_count <= meshlet_target_face_count) || (last_level_meshlet_count == meshlet_build_result.meshlets.count);
		last_level_meshlet_count = meshlet_build_result.meshlets.count;
		
		if (is_last_level == false) {
			DecimateMeshFaceGroups(
				mesh,
				allocator,
				heap_allocator,
				inputs->mesh.normalize_vertex_attributes,
				inputs->mesh.attribute_weights,
				meshlet_group_faces,
				meshlet_group_face_prefix_sum,
				meshlet_group_error_metrics,
				changed_vertex_mask
			);
			
			AppendChangedVertices(mesh, allocator, heap_allocator, changed_vertex_mask, attributes_id_to_vertex_index, vertices);
			
			// Compact the mesh after decimation to remove unused faces and edges.
			auto remap = CompactMesh(mesh, allocator);
			CompactMeshletGroupFaces(remap, meshlet_group_faces, meshlet_group_face_prefix_sum);
		} else {
			// There is no coarser version of the mesh. Set meshlet group errors to FLT_MAX to make sure LOD
			// culling test always succeeds for last level meshlets (i.e. coarser level is always too coarse).
			for (auto& error_metric : meshlet_group_error_metrics) error_metric.error = FLT_MAX;
		}
		
		VgtVirtualGeometryLevel level;
		level.begin_meshlet_groups_index = meshlet_groups.count;
		level.begin_meshlets_index       = meshlets.count;
		
		meshlet_group_base_index = meshlet_groups.count;
		BuildMeshletsAndMeshletGroups(heap_allocator, meshlet_build_result, meshlet_group_build_result, meshlet_group_error_metrics, meshlet_groups, meshlets, level_index);
		
		level.end_meshlet_groups_index = meshlet_groups.count;
		level.end_meshlets_index  = meshlets.count;
		ArrayAppend(levels, level);
		
		AllocatorFreeMemoryBlocks(allocator, allocator_high_water);
		
		if (is_last_level) break;
	}
	
	result->meshlet_groups         = meshlet_groups.data;
	result->meshlets               = meshlets.data;
	result->meshlet_vertex_indices = meshlet_vertex_indices.data;
	result->meshlet_triangles      = meshlet_triangles.data;
	result->vertices               = vertices.data;
	result->levels                 = levels.data;
	result->meshlet_group_count    = meshlet_groups.count;
	result->meshlet_count          = meshlets.count;
	result->meshlet_vertex_index_count = meshlet_vertex_indices.count;
	result->meshlet_triangle_count = meshlet_triangles.count;
	result->vertex_count           = vertices.count / (inputs->mesh.vertex_stride_bytes / sizeof(u32));
	result->level_count            = levels.count;
	VGT_ASSERT(heap_allocator.memory_block_count == 6);
	
	AllocatorFreeMemoryBlocks(allocator);
}

void VgtFreeVirtualGeometryBuildResult(const VgtVirtualGeometryBuildResult* result, const VgtSystemCallbacks* callbacks) {
	using namespace VirtualGeometryTools;
	
	Allocator heap_allocator;
	InitializeAllocator(heap_allocator, callbacks ? &callbacks->heap_allocator : nullptr);
	
	heap_allocator.memory_blocks[heap_allocator.memory_block_count++] = result->meshlet_groups;
	heap_allocator.memory_blocks[heap_allocator.memory_block_count++] = result->meshlets;
	heap_allocator.memory_blocks[heap_allocator.memory_block_count++] = result->meshlet_vertex_indices;
	heap_allocator.memory_blocks[heap_allocator.memory_block_count++] = result->meshlet_triangles;
	heap_allocator.memory_blocks[heap_allocator.memory_block_count++] = result->vertices;
	heap_allocator.memory_blocks[heap_allocator.memory_block_count++] = result->levels;
	AllocatorFreeMemoryBlocks(heap_allocator);
}


void VgtDecimateMesh(const VgtMeshDecimationInputs* inputs, VgtMeshDecimationResult* result, const VgtSystemCallbacks* callbacks) {
	using namespace VirtualGeometryTools;
	
	VGT_ASSERT(inputs);
	VGT_ASSERT(result);
	
	Allocator allocator;
	InitializeAllocator(allocator, callbacks ? &callbacks->temp_allocator : nullptr);

	Allocator heap_allocator;
	InitializeAllocator(heap_allocator, callbacks ? &callbacks->heap_allocator : nullptr);
	
	auto mesh = BuildEditableMesh(allocator, inputs->mesh.geometry_descs, inputs->mesh.geometry_desc_count, inputs->mesh.vertex_stride_bytes);
	
	Array<VgtDecimatedTriangleGeometryDesc> geometry_descs;
	ArrayResize(geometry_descs, heap_allocator, inputs->mesh.geometry_desc_count * inputs->level_of_detail_count);
	
	
	Array<u32> attributes_id_to_vertex_index;
	ArrayResizeMemset(attributes_id_to_vertex_index, allocator, mesh.attribute_count, 0xFF);
	
	Array<u8> changed_vertex_mask;
	ArrayResizeMemset(changed_vertex_mask, allocator, mesh.attribute_count, 0xFF);
	
	Array<u32>   indices;
	Array<float> vertices;
	
	for (u32 level_index = 0; level_index < inputs->level_of_detail_count; level_index += 1) {
		auto& level_of_detail_target = inputs->level_of_detail_descs[level_index];
		
		u32 allocator_high_water = allocator.memory_block_count;
		
		float max_error = 0.f;
		// Level of detail 0 commonly has the same number of faces as the source mesh.
		if (level_of_detail_target.target_face_count < mesh.face_count) {
			u32 heap_allocator_high_water = heap_allocator.memory_block_count;
			
			MeshDecimationState state;
			InitializeMeshDecimationState(mesh, inputs->mesh.attribute_weights, allocator, heap_allocator, state);
			
			
			EdgeCollapseHeap edge_collapse_heap;
			ArrayResize(edge_collapse_heap.edge_collapse_errors,  allocator, mesh.edge_count);
			ArrayResize(edge_collapse_heap.edge_id_to_heap_index, allocator, mesh.edge_count);
			ArrayResize(edge_collapse_heap.heap_index_to_edge_id, allocator, mesh.edge_count);
			
			for (EdgeID edge_id = { 0 }; edge_id.index < mesh.edge_count; edge_id.index += 1) {
				auto collapse_error = ComputeEdgeCollapseError(mesh, heap_allocator, state, edge_id);
				
				edge_collapse_heap.edge_collapse_errors[edge_id.index]  = collapse_error.min_error;
				edge_collapse_heap.edge_id_to_heap_index[edge_id.index] = edge_id.index;
				edge_collapse_heap.heap_index_to_edge_id[edge_id.index] = edge_id;
			}
			
			EdgeCollapseHeapInitialize(edge_collapse_heap);
			
			
			max_error = DecimateMeshFaceGroup(
				mesh,
				heap_allocator,
				state,
				edge_collapse_heap,
				inputs->mesh.normalize_vertex_attributes,
				changed_vertex_mask.data,
				level_of_detail_target.target_face_count,
				mesh.face_count,
				level_of_detail_target.target_error_limit
			);
			
			AllocatorFreeMemoryBlocks(heap_allocator, heap_allocator_high_water);
		}
		
		AppendChangedVertices(mesh, allocator, heap_allocator, changed_vertex_mask, attributes_id_to_vertex_index, vertices);
		
		CompactMesh(mesh, allocator);
		
		AllocatorFreeMemoryBlocks(allocator, allocator_high_water);
		
		
		if (indices.count + mesh.face_count * 3 > indices.capacity) {
			ArrayGrow(indices, heap_allocator, ArrayComputeNewCapacity(indices.capacity, indices.count + mesh.face_count * 3));
		}
		
		auto level_geometry_descs = CreateArrayView(geometry_descs, level_index * inputs->mesh.geometry_desc_count, (level_index + 1) * inputs->mesh.geometry_desc_count);
		
		u32 geometry_index = u32_max;
		for (FaceID face_id = { 0 }; face_id.index < mesh.face_count; face_id.index += 1) {
			auto& face = mesh[face_id];
			
			if (geometry_index != face.geometry_index) {
				VGT_ASSERT(geometry_index == u32_max || geometry_index < face.geometry_index);
				
				if (geometry_index != u32_max) {
					auto& geometry_desc = level_geometry_descs[geometry_index];
					geometry_desc.end_indices_index = indices.count;
				}
				geometry_index = face.geometry_index;
				
				auto& geometry_desc = level_geometry_descs[geometry_index];
				geometry_desc.level_of_detail_index = level_index;
				geometry_desc.max_error             = max_error;
				geometry_desc.begin_indices_index   = indices.count;
				geometry_desc.end_indices_index     = indices.count;
			}
			
			IterateCornerList<ElementType::Face>(mesh, face.corner_list_base, [&](CornerID corner_id) {
				auto attributes_id = mesh[corner_id].attributes_id;
				
				u32 vertex_index = attributes_id_to_vertex_index[attributes_id.index];
				ArrayAppend(indices, vertex_index);
			});
		}
		
		if (geometry_index != u32_max) {
			auto& geometry_desc = level_geometry_descs[geometry_index];
			geometry_desc.end_indices_index = indices.count;
		}
	}

	VGT_ASSERT(heap_allocator.memory_block_count == 3);
	
	u32 vertex_stride_dwords    = (mesh.attribute_stride_dwords + 3);
	result->geometry_descs      = geometry_descs.data;
	result->indices             = indices.data;
	result->vertices            = vertices.data;
	result->index_count         = indices.count;
	result->vertex_count        = vertices.count / vertex_stride_dwords;
	result->geometry_desc_count = geometry_descs.count;
	
	AllocatorFreeMemoryBlocks(allocator);
}

void VgtFreeMeshDecimationResult(const VgtMeshDecimationResult* result, const VgtSystemCallbacks* callbacks) {
	using namespace VirtualGeometryTools;
	
	Allocator heap_allocator;
	InitializeAllocator(heap_allocator, callbacks ? &callbacks->heap_allocator : nullptr);
	
	heap_allocator.memory_blocks[heap_allocator.memory_block_count++] = result->geometry_descs;
	heap_allocator.memory_blocks[heap_allocator.memory_block_count++] = result->indices;
	heap_allocator.memory_blocks[heap_allocator.memory_block_count++] = result->vertices;
	AllocatorFreeMemoryBlocks(heap_allocator);
}


//
// Based on [Kapoulkine 2025].
//
VgtSphereBounds VgtComputeSphereBoundsUnion(const VgtSphereBounds* source_sphere_bounds, uint32_t source_sphere_bounds_count) {
	using namespace VirtualGeometryTools;
	
	u32 aabb_min_index[3] = { 0, 0, 0 };
	u32 aabb_max_index[3] = { 0, 0, 0 };
	
	float aabb_min[3] = { +FLT_MAX, +FLT_MAX, +FLT_MAX };
	float aabb_max[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
	
	for (u32 i = 0; i < source_sphere_bounds_count; i += 1) {
		auto source_bounds = source_sphere_bounds[i];
		
		auto min = source_bounds.center - source_bounds.radius;
		auto max = source_bounds.center + source_bounds.radius;
		
		for (u32 axis = 0; axis < 3; axis += 1) {
			if (aabb_min[axis] > min[axis]) {
				aabb_min[axis] = min[axis];
				aabb_min_index[axis] = i;
			}
			
			if (aabb_max[axis] < max[axis]) {
				aabb_max[axis] = max[axis];
				aabb_max_index[axis] = i;
			}
		}
	}
	
	float max_axis_length = -FLT_MAX;
	u32 max_axis_length_index = 0;
	
	for (u32 axis = 0; axis < 3; axis += 1) {
		auto min_bounds = source_sphere_bounds[aabb_min_index[axis]];
		auto max_bounds = source_sphere_bounds[aabb_max_index[axis]];
		
		float axis_length = Length(max_bounds.center - min_bounds.center) + min_bounds.radius + max_bounds.radius;
		if (max_axis_length < axis_length) {
			max_axis_length = axis_length;
			max_axis_length_index = axis;
		}
	}
	
	VgtSphereBounds result_bounds;
	result_bounds.center = (source_sphere_bounds[aabb_min_index[max_axis_length_index]].center + source_sphere_bounds[aabb_max_index[max_axis_length_index]].center) * 0.5f;
	result_bounds.radius = max_axis_length * 0.5f;
	
	for (u32 i = 0; i < source_sphere_bounds_count; i += 1) {
		auto source_bounds = source_sphere_bounds[i];
		
		float distance = Length(source_bounds.center - result_bounds.center);
		if (distance + source_bounds.radius > result_bounds.radius) {
			float shift_t = distance > 0.f ? ((source_bounds.radius - result_bounds.radius) / distance) * 0.5f + 0.5f : 0.f;
			
			result_bounds.center = (result_bounds.center * (1.f - shift_t) + source_bounds.center * shift_t);
			result_bounds.radius = (result_bounds.radius + source_bounds.radius + distance) * 0.5f;
		}
	}
	
	return result_bounds;
}

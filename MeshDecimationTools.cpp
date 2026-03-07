#include "MeshDecimationTools.h"

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


#if !defined(MDT_CACHE_LINE_SIZE)
#define MDT_CACHE_LINE_SIZE 64
#endif // !defined(MDT_CACHE_LINE_SIZE)


#if defined(_MSC_VER)
#define never_inline_function __declspec(noinline)
#define always_inline_function __forceinline
#else // !defined(_MSC_VER)
#define never_inline_function
#define always_inline_function
#endif // !defined(_MSC_VER)

#if !defined(MDT_UNUSED_VARIABLE)
#define MDT_UNUSED_VARIABLE(name) (void)(name)
#endif // !defined(MDT_UNUSED_VARIABLE)

#if defined(__AVX__) && !defined(MDT_ENABLE_AVX)
#define MDT_ENABLE_AVX 1
#endif // defined(__AVX__) && !defined(MDT_ENABLE_AVX)

// Using AVX + FMA gives a different result from scalar or just AVX. AVX and scalar match each other.
#if !defined(MDT_ENABLE_FMA)
#define MDT_ENABLE_FMA 0
#endif // defined(MDT_ENABLE_FMA)

#if defined(MDT_ENABLE_AVX)
#include <immintrin.h>
#endif // defined(MDT_ENABLE_AVX)

#define compile_const constexpr static const


namespace MeshDecimationTools
{

using u8  = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;

static_assert(sizeof(u8)  == 1, "Invalid u8 size.");
static_assert(sizeof(u32) == 4, "Invalid u32 size.");
static_assert(sizeof(u64) == 8, "Invalid u64 size.");

compile_const u32 u32_max = (u32)0xFFFF'FFFF;
compile_const u64 u64_max = (u64)0xFFFF'FFFF'FFFF'FFFF;

using Vector3 = MdtVector3;

compile_const u32 meshlet_max_vertex_count = MDT_MAX_MESHLET_VERTEX_COUNT;
compile_const u32 meshlet_max_face_degree  = 3;

compile_const u32 meshlet_max_face_count = MDT_MAX_MESHLET_FACE_COUNT;
compile_const u32 meshlet_min_face_count = meshlet_max_face_count / 4;
compile_const float discontinuous_meshlet_max_expansion = 2.f; // Sqrt of the maximum AABB expansion when adding a discontinuous face to a meshlet.

compile_const u32 meshlet_group_max_meshlet_count = MDT_MESHLET_GROUP_SIZE;
compile_const u32 meshlet_group_min_meshlet_count = meshlet_group_max_meshlet_count / 2;
compile_const float discontinuous_meshlet_group_max_expansion = 4.f; // Sqrt of the maximum AABB expansion when adding a discontinuous meshlet to a group.

compile_const u32 continuous_lod_max_levels_of_details = MDT_MAX_CLOD_LEVEL_COUNT;

compile_const float default_geometric_error_weight = 0.5f;
compile_const float default_attribute_error_weight = 1.f;
compile_const u32 edge_collapse_candidate_position_count = 3;
compile_const u32 max_edge_collapse_validation_face_count = 32;


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
	VertexID vertex_0;
	VertexID vertex_1;
	
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


struct alignas(MDT_CACHE_LINE_SIZE) EditableMeshView {
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
	
	Face&   operator[] (FaceID face_id) const             { return faces[face_id.index]; }
	Edge&   operator[] (EdgeID edge_id) const             { return edges[edge_id.index]; }
	Vertex& operator[] (VertexID vertex_id) const         { return vertices[vertex_id.index]; }
	Corner& operator[] (CornerID corner_id) const         { return corners[corner_id.index]; }
	float*  operator[] (AttributesID attributes_id) const { return attributes + attributes_id.index * attribute_stride_dwords; }
};
static_assert(sizeof(EditableMeshView) == 64, "Invalid EditableMeshView size.");

struct alignas(MDT_CACHE_LINE_SIZE) IndexedMeshView {
	VertexID*     face_vertex_ids    = nullptr;
	AttributesID* face_attribute_ids = nullptr;
	Vector3*      vertices           = nullptr;
	float*        attributes         = nullptr;
	u32*          face_geometry_indices = nullptr;
	
	u32 face_count      = 0;
	u32 vertex_count    = 0;
	u32 attribute_count = 0;
	u32 attribute_stride_dwords = 0;
	
	u64 padding = 0;
	
	Vector3& operator[] (VertexID vertex_id) const         { return vertices[vertex_id.index]; }
	float*   operator[] (AttributesID attributes_id) const { return attributes + attributes_id.index * attribute_stride_dwords; }
};
static_assert(sizeof(IndexedMeshView) == 64, "Invalid IndexedMeshView size.");

static u64 PackEdgeKey(VertexID vertex_id_0, VertexID vertex_id_1) {
	// Always pack VertexIDs in ascending order to ensure that PackEdgeKey(A, B) == PackEdgeKey(B, A) and they hash to the same value.
	return vertex_id_1.index > vertex_id_0.index ?
		((u64)vertex_id_1.index << 32) | (u64)vertex_id_0.index :
		((u64)vertex_id_0.index << 32) | (u64)vertex_id_1.index;
}

static u64 LoadEdgeKey(const Edge& edge) {
	return (u64)edge.vertex_0.index | ((u64)edge.vertex_1.index << 32);
}

template<typename ElementID> ElementID GetElementID(const Corner& corner);
template<> always_inline_function VertexID GetElementID<VertexID>(const Corner& corner) { return corner.vertex_id; }
template<> always_inline_function EdgeID GetElementID<EdgeID>(const Corner& corner) { return corner.edge_id; }
template<> always_inline_function FaceID GetElementID<FaceID>(const Corner& corner) { return corner.face_id; }

template<typename ElementID>
static void CornerListInsert(const EditableMeshView& mesh, ElementID element_id, CornerID new_corner_id) {
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

template<typename ElementID>
static bool CornerListRemove(const EditableMeshView& mesh, CornerID corner_id) {
	compile_const u32 element_type = (u32)ElementID::element_type;
	
	auto& corner = mesh[corner_id];
	auto prev_corner_id = corner.corner_list_around[element_type].prev;
	auto next_corner_id = corner.corner_list_around[element_type].next;
	
	mesh[next_corner_id].corner_list_around[element_type].prev = prev_corner_id;
	mesh[prev_corner_id].corner_list_around[element_type].next = next_corner_id;
	
	// Remove the element if it lost it's last corner.
	bool is_last_reference = (prev_corner_id.index == corner_id.index);
	auto new_corner_list_base = is_last_reference ? CornerID{ u32_max } : prev_corner_id;
	
	mesh[GetElementID<ElementID>(corner)].corner_list_base = new_corner_list_base;
	
	corner.corner_list_around[element_type].prev.index = u32_max;
	corner.corner_list_around[element_type].next.index = u32_max;
	
	return is_last_reference;
}

// Iterate linked list around a given element type starting with the base corner id. 
// Removal while iterating is allowed.
template<typename ElementID, typename Lambda>
static void IterateCornerList(const EditableMeshView& mesh, CornerID corner_list_base, Lambda&& lambda) {
	auto& element = mesh[GetElementID<ElementID>(mesh[corner_list_base])];
	
	auto current_corner_id = corner_list_base;
	do {
		auto next_corner_id = mesh[current_corner_id].corner_list_around[(u32)ElementID::element_type].next;
		
		lambda(current_corner_id);
		
		current_corner_id = next_corner_id;
	} while (current_corner_id.index != corner_list_base.index && element.corner_list_base.index != u32_max);
}

template<typename Lambda>
static void IterateIncomingAndOutgoingFaceCornerEdges(const EditableMeshView& mesh, CornerID corner_list_base, Lambda&& lambda) {
	auto corner_id_0 = mesh[corner_list_base].corner_list_around[(u32)FaceID::element_type].prev;
	auto corner_id_1 = corner_list_base;
	
	lambda(corner_id_0); // Incoming
	lambda(corner_id_1); // Outgoing
}

always_inline_function static void PatchReferencesToElement(const EditableMeshView& mesh, VertexID element_0, VertexID element_1, CornerID corner_id) {
	mesh[corner_id].vertex_id = element_0;
	
	IterateIncomingAndOutgoingFaceCornerEdges(mesh, corner_id, [&](CornerID corner_id) {
		auto& edge = mesh[mesh[corner_id].edge_id];
		if (edge.vertex_0.index == element_1.index) edge.vertex_0 = element_0;
		if (edge.vertex_1.index == element_1.index) edge.vertex_1 = element_0;
		MDT_ASSERT(edge.vertex_0.index != edge.vertex_1.index);
	});
}

always_inline_function static void PatchReferencesToElement(const EditableMeshView& mesh, EdgeID element_0, EdgeID /*element_1*/, CornerID corner_id) {
	mesh[corner_id].edge_id = element_0;
}

// Merge linked lists around element_0 and element_1 and remove element_1.
// Patch up references to element_1 with a reference to element_0.
template<typename ElementID>
static ElementID CornerListMerge(const EditableMeshView& mesh, ElementID element_0, ElementID element_1) {
	auto base_id_0 = mesh[element_0].corner_list_base;
	auto base_id_1 = mesh[element_1].corner_list_base;
	
	compile_const ElementType element_type_t = ElementID::element_type;
	compile_const u32 element_type = (u32)element_type_t;
	
	auto remaining_element_id = ElementID{ u32_max };
	if (base_id_0.index != u32_max && base_id_1.index != u32_max) {
		IterateCornerList<ElementID>(mesh, base_id_1, [&](CornerID corner_id) {
			PatchReferencesToElement(mesh, element_0, element_1, corner_id);
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
	compile_const u32 max_memory_block_count = 64;
	
	MdtAllocatorCallbacks callbacks;
	
	u32 memory_block_count = 0;
	void* memory_blocks[max_memory_block_count] = {};
};

static void* AllocateMemoryBlock(Allocator& allocator, void* old_memory_block, u64 size_bytes) {
	MDT_ASSERT(old_memory_block != nullptr || allocator.memory_block_count < Allocator::max_memory_block_count);
	MDT_ASSERT(old_memory_block == nullptr || allocator.memory_block_count > 0);
	MDT_ASSERT(old_memory_block == nullptr || allocator.memory_blocks[allocator.memory_block_count - 1] == old_memory_block);
	
	void* memory_block = allocator.callbacks.reallocate(old_memory_block, size_bytes, allocator.callbacks.user_data);
	u32 memory_block_index = old_memory_block ? allocator.memory_block_count - 1 : allocator.memory_block_count++;
	
	allocator.memory_blocks[memory_block_index] = memory_block;
	
	return memory_block;
}

static void InitializeAllocator(Allocator& allocator, const MdtAllocatorCallbacks* callbacks) {
	if (callbacks && callbacks->reallocate) {
		allocator.callbacks = *callbacks;
	} else {
		allocator.callbacks.reallocate = [](void* old_memory_block, u64 size_bytes, void*) {
			void* result = nullptr;
			if (size_bytes == 0) {
				free(old_memory_block);
			} else {
				result = realloc(old_memory_block, size_bytes);
			}
			return result;
		};
	}
}

static void AllocatorFreeMemoryBlocks(Allocator& allocator, u32 last_memory_block_index = 0) {
	for (u32 i = allocator.memory_block_count; i > last_memory_block_index; i -= 1) {
		allocator.callbacks.reallocate(allocator.memory_blocks[i - 1], 0, allocator.callbacks.user_data);
	}
	allocator.memory_block_count = last_memory_block_index;
}

static u32 AllocatorFindMemoryBlock(Allocator& allocator, void* old_memory_block) {
	for (u32 i = allocator.memory_block_count; i > 0; i -= 1) {
		if (allocator.memory_blocks[i - 1] == old_memory_block) return i - 1;
	}
	return u32_max;	
}

static u32 ParallelForThreadCount(const MdtParallelForCallbacks* parallel_for) {
	return parallel_for && parallel_for->callback && parallel_for->thread_count ? parallel_for->thread_count : 1;
}

static u32 DivideAndRoundUp(u32 numerator, u32 denominator) {
	return (numerator + denominator - 1) / denominator;
}

template<typename Lambda>
static void ParallelFor(const MdtParallelForCallbacks* parallel_for, u32 work_item_count, Lambda&& lambda) {
	if (ParallelForThreadCount(parallel_for) >= 2) {
		parallel_for->callback(parallel_for->user_data, &lambda, work_item_count, [](void* user_data, u32 work_item_index) {
			auto& lambda = *(Lambda*)user_data;
			lambda(work_item_index);
		});
	} else {
		for (u32 i = 0; i < work_item_count; i += 1) {
			lambda(i);
		}
	}
}

template<typename Lambda>
static void ParallelForBatched(const MdtParallelForCallbacks* parallel_for, u32 work_item_count, u32 batch_size, Lambda&& lambda) {
	if (ParallelForThreadCount(parallel_for) >= 2) {
		u32 loop_count = DivideAndRoundUp(work_item_count, batch_size);
		ParallelFor(parallel_for, loop_count, [&](u32 work_item_index) {
			u32 end_index = (work_item_index + 1) * batch_size;
			if (end_index > work_item_count) end_index = work_item_count;
			
			for (u32 i = work_item_index * batch_size; i < end_index; i += 1) {
				lambda(i);
			}
		});
	} else {
		for (u32 i = 0; i < work_item_count; i += 1) {
			lambda(i);
		}
	}
}


#define DECLARE_ARRAY_OPERATORS() \
	T& operator[] (u32 index) { MDT_ASSERT(index < count); return data[index]; } \
	const T& operator[] (u32 index) const { MDT_ASSERT(index < count); return data[index]; } \
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
static_assert(sizeof(Array<u32>) == 16, "Invalid Array<T> size.");

template<typename T, u32 compile_time_capacity>
struct FixedSizeArray {
	using ValueType = T;
	compile_const u32 capacity = compile_time_capacity;
	
	T data[capacity] = {};
	u32 count = 0;

	DECLARE_ARRAY_OPERATORS()
};
static_assert(sizeof(FixedSizeArray<u32, 1>) == 8, "Invalid FixedSizeArray<T, c> size.");

template<typename T>
struct ArrayView {
	using ValueType = T;
	
	T* data = nullptr;
	u32 count = 0;
	
	DECLARE_ARRAY_OPERATORS()
};
static_assert(sizeof(ArrayView<u32>) == 16, "Invalid ArrayView<T> size.");

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
	MDT_ASSERT(array.count < array.capacity);
	array.data[array.count++] = value;
}

template<typename T>
never_inline_function static void ArrayGrow(Array<T>& array, Allocator& allocator, u32 new_capacity) {
	u32 old_memory_block_index = AllocatorFindMemoryBlock(allocator, array.data);
	
	void* memory_block = allocator.callbacks.reallocate(array.data, new_capacity * sizeof(T), allocator.callbacks.user_data);
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
	MDT_ASSERT(index < array.count);
	
	array.data[index] = array.data[array.count - 1];
	array.count -= 1;
}

template<typename ArrayT>
static typename ArrayT::ValueType& ArrayLastElement(ArrayT& array) {
	MDT_ASSERT(array.count != 0);
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

static VertexID HashTableAddOrFind(VertexHashTable& table, Array<Vector3>& vertices, const Vector3& position) {
	u32 table_size = table.vertex_ids.count;
	u32 mod_mask   = table_size - 1u;
	
	u32 hash  = ComputePositionHash(position);
	u32 index = (hash & mod_mask);
	
	for (u32 i = 0; i <= mod_mask; i += 1) {
		auto vertex_id = table.vertex_ids[index];
		
		if (vertex_id.index == u32_max) {
			auto new_vertex_id = VertexID{ vertices.count };
			table.vertex_ids[index] = new_vertex_id;
			
			ArrayAppend(vertices, position);
			
			return new_vertex_id;
		}
		
		auto existing_position = vertices[vertex_id.index];
		if (existing_position.x == position.x && existing_position.y == position.y && existing_position.z == position.z) {
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
			edge.vertex_0.index = (u32)(edge_key >> 0);
			edge.vertex_1.index = (u32)(edge_key >> 32);
			edge.corner_list_base.index = u32_max;
			ArrayAppend(edges, edge);
			
			return new_edge_id;
		}
		
		if (LoadEdgeKey(edges[edge_id.index]) == edge_key) {
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
	void* memory_block = heap_allocator.callbacks.reallocate(nullptr, new_capacity * sizeof(EdgeDuplicateMap::KeyValue), heap_allocator.callbacks.user_data);
	u32 memory_block_index = old_memory_block_index != u32_max ? old_memory_block_index : heap_allocator.memory_block_count++;
	
	heap_allocator.memory_blocks[memory_block_index] = memory_block;
	
	table.keys_values = (EdgeDuplicateMap::KeyValue*)memory_block;
	table.capacity    = new_capacity;
	
	HashTableClear(table);
	
	for (auto key_value: ArrayView<EdgeDuplicateMap::KeyValue>{ old_keys_values, old_capacity }) {
		if (key_value.edge_key != u64_max) HashTableAddOrFind(table, heap_allocator, key_value.edge_key, key_value.edge_id);
	}
	
	heap_allocator.callbacks.reallocate(old_keys_values, 0, heap_allocator.callbacks.user_data);
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

static IndexedMeshView BuildIndexedMesh(Allocator& allocator, const MdtTriangleGeometryDesc* geometry_descs, u32 geometry_desc_count, u32 vertex_stride_bytes) {
	MDT_PROFILER_SCOPE("BuildIndexedMesh");
	
	u32 vertex_stride_dwords    = vertex_stride_bytes / sizeof(u32);
	u32 attribute_stride_dwords = vertex_stride_dwords - 3;
	
	if (vertex_stride_dwords < 3 || attribute_stride_dwords > MDT_MAX_ATTRIBUTE_STRIDE_DWORDS) return {};
	
	
	u32 vertices_count = 0;
	u32 indices_count  = 0;
	
	for (u32 geometry_index = 0; geometry_index < geometry_desc_count; geometry_index += 1) {
		auto& desc = geometry_descs[geometry_index];
		vertices_count += desc.vertex_count;
		indices_count  += desc.index_count;
	}
	u32 triangle_count = (indices_count / 3);
	
	
	Array<VertexID>     face_vertex_ids;
	Array<AttributesID> face_attribute_ids;
	Array<u32>          face_geometry_indices;
	Array<Vector3>      vertices;
	Array<float>        attributes;
	ArrayResize(face_vertex_ids, allocator, triangle_count * 3);
	ArrayResize(face_attribute_ids, allocator, triangle_count * 3);
	ArrayReserve(face_geometry_indices, allocator, triangle_count);
	ArrayReserve(vertices, allocator, vertices_count);
	ArrayResize(attributes, allocator, vertices_count * attribute_stride_dwords);
	
	IndexedMeshView mesh;
	mesh.face_vertex_ids       = face_vertex_ids.data;
	mesh.face_attribute_ids    = face_attribute_ids.data;
	mesh.face_geometry_indices = face_geometry_indices.data;
	mesh.vertices              = vertices.data;
	mesh.attributes            = attributes.data;
	mesh.attribute_count       = vertices_count;
	mesh.attribute_stride_dwords = attribute_stride_dwords;
	
	u32 allocator_high_water = allocator.memory_block_count;
	
	Array<VertexID> src_vertex_index_to_vertex_id;
	ArrayResize(src_vertex_index_to_vertex_id, allocator, vertices_count);
	
	VertexHashTable vertex_table;
	ArrayResizeMemset(vertex_table.vertex_ids, allocator, ComputeHashTableSize(vertices_count), 0xFF);
	
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
			
			bool has_duplicate_vertices = 
				vertex_ids[0].index == vertex_ids[1].index ||
				vertex_ids[0].index == vertex_ids[2].index ||
				vertex_ids[1].index == vertex_ids[2].index;
			
			// Skip primitives that reduce to a single line or a point. PerformEdgeCollapse(...) can't reliably handle them.
			if (has_duplicate_vertices) continue;
			
			
			u32 face_index = face_geometry_indices.count;
			ArrayAppend(face_geometry_indices, geometry_index);
			
			for (u32 corner_index = 0; corner_index < 3; corner_index += 1) {
				face_vertex_ids[face_index * 3 + corner_index] = vertex_ids[corner_index];
				face_attribute_ids[face_index * 3 + corner_index] = AttributesID{ indices[corner_index] };
			}
		}
		base_vertex_index += desc.vertex_count;
	}
	mesh.face_count = face_geometry_indices.count;
	
	AllocatorFreeMemoryBlocks(allocator, allocator_high_water);
	
	return mesh;
}


struct EdgeCollapseResult {
	VertexID remaining_vertex_id;
	u32 removed_face_count = 0;
};

static EdgeCollapseResult PerformEdgeCollapse(const EditableMeshView& mesh, EdgeID edge_id, Allocator& heap_allocator, EdgeDuplicateMap& edge_duplicate_map, Array<EdgeID>& removed_edge_array) {
	auto& edge = mesh[edge_id];
	
	MDT_ASSERT(edge.vertex_0.index != edge.vertex_1.index);
	MDT_ASSERT(edge.corner_list_base.index != u32_max);
	MDT_ASSERT(mesh[edge.vertex_0].corner_list_base.index != u32_max);
	MDT_ASSERT(mesh[edge.vertex_1].corner_list_base.index != u32_max);
	MDT_ASSERT(mesh[edge.vertex_0].corner_list_base.index != mesh[edge.vertex_1].corner_list_base.index);
	
	removed_edge_array.count = 0;
	u32 removed_face_count = 0;
	IterateCornerList<EdgeID>(mesh, edge.corner_list_base, [&](CornerID corner_id) {
		auto& corner = mesh[corner_id];
		auto& face   = mesh[corner.face_id];
		
		IterateCornerList<FaceID>(mesh, face.corner_list_base, [&](CornerID corner_id) {
			CornerListRemove<VertexID>(mesh, corner_id);
			bool edge_removed = CornerListRemove<EdgeID>(mesh, corner_id);
			bool face_removed = CornerListRemove<FaceID>(mesh, corner_id);
			
			if (edge_removed) ArrayAppendMaybeGrow(removed_edge_array, heap_allocator, mesh[corner_id].edge_id);
			removed_face_count += (u32)face_removed;
		});
	});
	
	auto remaining_vertex_id = CornerListMerge<VertexID>(mesh, edge.vertex_0, edge.vertex_1);

	if (remaining_vertex_id.index != u32_max) {
		auto remaining_base_id = mesh[remaining_vertex_id].corner_list_base;
		
		IterateCornerList<VertexID>(mesh, remaining_base_id, [&](CornerID corner_id) {
			IterateIncomingAndOutgoingFaceCornerEdges(mesh, corner_id, [&](CornerID corner_id) {
				auto edge_id_1 = mesh[corner_id].edge_id;
				auto& edge_1 = mesh[edge_id_1];
				
				auto edge_id_0 = HashTableAddOrFind(edge_duplicate_map, heap_allocator, PackEdgeKey(edge_1.vertex_0, edge_1.vertex_1), edge_id_1);
				if (edge_id_0.index != edge_id_1.index) {
					CornerListMerge<EdgeID>(mesh, edge_id_0, edge_id_1);
					ArrayAppendMaybeGrow(removed_edge_array, heap_allocator, edge_id_1);
				}
			});
		});
		
		IterateCornerList<VertexID>(mesh, remaining_base_id, [&](CornerID corner_id_0) {
			IterateCornerList<FaceID>(mesh, corner_id_0, [&](CornerID corner_id_1) {
				if (remaining_base_id.index == corner_id_1.index) return;
				
				IterateCornerList<VertexID>(mesh, corner_id_1, [&](CornerID corner_id_2) {
					if (corner_id_1.index == corner_id_2.index) return;
					
					IterateIncomingAndOutgoingFaceCornerEdges(mesh, corner_id_2, [&](CornerID corner_id) {
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
static_assert(sizeof(Quadric) == sizeof(float) * 11, "Invalid Quadric size.");

struct QuadricAttributeGradient {
	Vector3 g = { 0.f, 0.f, 0.f };
	float   d = 0.f;
};

struct QuadricWithAttributes : Quadric {
#if MDT_ENABLE_ATTRIBUTE_SUPPORT
	QuadricAttributeGradient attributes[MDT_MAX_ATTRIBUTE_STRIDE_DWORDS]; // Note that this array is used as a 'flexible array'.
	
	// Quadric with attributes cannot be copied by value as it's variable size (i.e. it might be missing trailing attributes).
	QuadricWithAttributes() { memset(this, 0, sizeof(QuadricWithAttributes)); }
	QuadricWithAttributes(const QuadricWithAttributes&) = delete;
	QuadricWithAttributes& operator= (const QuadricWithAttributes&) = delete;
	QuadricWithAttributes& operator= (QuadricWithAttributes&&) = delete;
#endif // MDT_ENABLE_ATTRIBUTE_SUPPORT
};
static_assert(sizeof(QuadricWithAttributes) == sizeof(Quadric) + sizeof(QuadricAttributeGradient) * MDT_MAX_ATTRIBUTE_STRIDE_DWORDS, "Invalid QuadricWithAttributes size.");


//
// Matt Pharr's blog. 2019. Accurate Differences of Products with Kahan's Algorithm.
//
always_inline_function static float DifferenceOfProducts(float a, float b, float c, float d) {
	float cd = c * d;
	float err = fmaf(c, -d,  cd);
	float dop = fmaf(a,  b, -cd);
	return dop + err;
}

always_inline_function static Vector3 operator* (const Vector3& lh, const Vector3& rh) { return Vector3{ lh.x * rh.x, lh.y * rh.y, lh.z * rh.z }; }
always_inline_function static Vector3 operator+ (const Vector3& lh, const Vector3& rh) { return Vector3{ lh.x + rh.x, lh.y + rh.y, lh.z + rh.z }; }
always_inline_function static Vector3 operator- (const Vector3& lh, const Vector3& rh) { return Vector3{ lh.x - rh.x, lh.y - rh.y, lh.z - rh.z }; }
always_inline_function static Vector3 operator* (const Vector3& lh, float rh) { return Vector3{ lh.x * rh, lh.y * rh, lh.z * rh }; }
always_inline_function static Vector3 operator+ (const Vector3& lh, float rh) { return Vector3{ lh.x + rh, lh.y + rh, lh.z + rh }; }
always_inline_function static Vector3 operator- (const Vector3& lh, float rh) { return Vector3{ lh.x - rh, lh.y - rh, lh.z - rh }; }
always_inline_function static float DotProduct(const Vector3& lh, const Vector3& rh) { return lh.x * rh.x + lh.y * rh.y + lh.z * rh.z; }
always_inline_function static float Length(const Vector3& v) { return sqrtf(DotProduct(v, v)); }
always_inline_function static Vector3 CrossProduct(const Vector3& lh, const Vector3& rh) { return Vector3{ lh.y * rh.z - lh.z * rh.y, lh.z * rh.x - lh.x * rh.z, lh.x * rh.y - lh.y * rh.x }; }
always_inline_function static float LoadElementByIndex(const Vector3& vector, u32 index) { return (&vector.x)[index]; }
always_inline_function static void StoreElementByIndex(Vector3& vector, u32 index, float value) { (&vector.x)[index] = value; }


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

template<typename T>
always_inline_function T Min(T value, T min) {
	return value < min ? value : min;
}


#if MDT_ENABLE_AVX
#define VectorLoadSimd(result, address, stride)\
	auto result##x = _mm256_loadu_ps((address) + (stride) * 0);\
	auto result##y = _mm256_loadu_ps((address) + (stride) * 1);\
	auto result##z = _mm256_loadu_ps((address) + (stride) * 2)

#define VectorLoadBroadcastSimd(result, address)\
	auto result##x = _mm256_broadcast_ss(address + 0);\
	auto result##y = _mm256_broadcast_ss(address + 1);\
	auto result##z = _mm256_broadcast_ss(address + 2)

#define VectorLoadConstantSimd(result, constant)\
	auto result = _mm256_set1_ps(constant);

#define VectorLoadZeroSimd(result)\
	auto result = _mm256_setzero_ps();

#define VectorSubSimd(result, lh, rh)\
	auto result##x = _mm256_sub_ps(lh##x, rh##x);\
	auto result##y = _mm256_sub_ps(lh##y, rh##y);\
	auto result##z = _mm256_sub_ps(lh##z, rh##z)

#if MDT_ENABLE_FMA
#define CrossProductSimd(result, lh, rh)\
	auto result##x = _mm256_fmsub_ps(lh##y, rh##z, _mm256_mul_ps(lh##z, rh##y));\
	auto result##y = _mm256_fmsub_ps(lh##z, rh##x, _mm256_mul_ps(lh##x, rh##z));\
	auto result##z = _mm256_fmsub_ps(lh##x, rh##y, _mm256_mul_ps(lh##y, rh##x))
#else // !MDT_ENABLE_FMA
#define CrossProductSimd(result, lh, rh)\
	auto result##x = _mm256_sub_ps(_mm256_mul_ps(lh##y, rh##z), _mm256_mul_ps(lh##z, rh##y));\
	auto result##y = _mm256_sub_ps(_mm256_mul_ps(lh##z, rh##x), _mm256_mul_ps(lh##x, rh##z));\
	auto result##z = _mm256_sub_ps(_mm256_mul_ps(lh##x, rh##y), _mm256_mul_ps(lh##y, rh##x))
#endif // !MDT_ENABLE_FMA

#if MDT_ENABLE_FMA
#define DotProductSimd(result, lh, rh)\
	auto result = _mm256_fmadd_ps(lh##x, rh##x, _mm256_fmadd_ps(lh##y, rh##y, _mm256_mul_ps(lh##z, rh##z)))
#else // !MDT_ENABLE_FMA
#define DotProductSimd(result, lh, rh)\
	auto result = _mm256_add_ps(_mm256_mul_ps(lh##x, rh##x), _mm256_add_ps(_mm256_mul_ps(lh##y, rh##y), _mm256_mul_ps(lh##z, rh##z)))
#endif // !MDT_ENABLE_FMA

#define CompareLessThanSimd(result, lh, rh)\
	auto result = _mm256_movemask_ps(_mm256_cmp_ps(lh, rh, _CMP_LT_OS));

#define CompareGreaterThanSimd(result, lh, rh)\
	auto result = _mm256_movemask_ps(_mm256_cmp_ps(lh, rh, _CMP_GT_OS));

#define GetSimdWidth() 8u
#else // !MDT_ENABLE_AVX
#define GetSimdWidth() 1u
#endif // !MDT_ENABLE_AVX

#define VectorStoreSOA(address, vector, stride)\
	(address)[stride * 0] = vector.x;\
	(address)[stride * 1] = vector.y;\
	(address)[stride * 2] = vector.z;


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
	
#if MDT_ENABLE_ATTRIBUTE_SUPPORT
	for (u32 i = 0; i < attribute_stride_dwords; i += 1) {
		auto& attribute_accumulator = accumulator.attributes[i];
		auto& attribute_quadric     = quadric.attributes[i];

		attribute_accumulator.g.x += attribute_quadric.g.x;
		attribute_accumulator.g.y += attribute_quadric.g.y;
		attribute_accumulator.g.z += attribute_quadric.g.z;
		attribute_accumulator.d   += attribute_quadric.d;
	}
#endif // MDT_ENABLE_ATTRIBUTE_SUPPORT
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
	
	
#if MDT_ENABLE_ATTRIBUTE_SUPPORT
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
#endif // MDT_ENABLE_ATTRIBUTE_SUPPORT
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
	
#if MDT_ENABLE_ATTRIBUTE_SUPPORT
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
#endif // MDT_ENABLE_ATTRIBUTE_SUPPORT
	
	return fabsf(weighted_error);
}

#if MDT_ENABLE_ATTRIBUTE_SUPPORT
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
#endif // MDT_ENABLE_ATTRIBUTE_SUPPORT

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
	
#if MDT_ENABLE_ATTRIBUTE_SUPPORT
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
#endif // MDT_ENABLE_ATTRIBUTE_SUPPORT
	
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

// Check if any face normal around the collapsed edge is flipped or becomes zero area, excluding collapsed faces.
static u32 ValidateEdgeCollapsePositions(Vector3 candidate_positions[edge_collapse_candidate_position_count], float face_vertex_positions[max_edge_collapse_validation_face_count * 9], u32 face_count) {
	u32 valid_position_mask = (1u << edge_collapse_candidate_position_count) - 1u;
	
#if GetSimdWidth() != 1
	VectorLoadConstantSimd(eps, FLT_EPSILON);
	VectorLoadZeroSimd(zero);
	
	compile_const u32 stride = max_edge_collapse_validation_face_count;
	
	u32 iteration_count = (face_count + GetSimdWidth() - 1) / GetSimdWidth();
	for (u32 i = 0; i < iteration_count; i += 1) {
		VectorLoadSimd(p0, face_vertex_positions + i * GetSimdWidth() + stride * 0, stride);
		VectorLoadSimd(p1, face_vertex_positions + i * GetSimdWidth() + stride * 3, stride);
		VectorLoadSimd(p2, face_vertex_positions + i * GetSimdWidth() + stride * 6, stride);
		
		VectorSubSimd(p20, p2, p0);
		VectorSubSimd(p21, p2, p1);
		CrossProductSimd(n0, p21, p20);
		
		DotProductSimd(n0l, n0, n0);
		CompareGreaterThanSimd(is_non_zero_area_old, n0l, eps);
		
		u32 lane_count = (face_count - i * GetSimdWidth());
		u32 active_lane_mask = lane_count > GetSimdWidth() ? ((1u << GetSimdWidth()) - 1) : ((1u << lane_count) - 1);
		
		for (u32 candidate_index = 0; candidate_index < edge_collapse_candidate_position_count; candidate_index += 1) {
			VectorLoadBroadcastSimd(c, &candidate_positions[candidate_index].x);
			
			VectorSubSimd(p2c, p2, c);
			CrossProductSimd(n1, p2c, p20); // p1 is replaced with the candidate vertex.
			
			DotProductSimd(n0n1, n0, n1);
			CompareLessThanSimd(is_flipped, n0n1, zero);
			
			DotProductSimd(n1l, n1, n1);
			CompareLessThanSimd(is_zero_area_new, n1l, eps);
			
			// Prevent flipped face normals and zero area faces.
			bool reject_edge_collapse = ((is_flipped | (is_non_zero_area_old & is_zero_area_new)) & active_lane_mask) != 0;
			
			if (reject_edge_collapse) {
				valid_position_mask &= ~(1u << candidate_index);
			}
		}
	}
#else // GetSimdWidth() == 1
	for (u32 i = 0; i < face_count; i += 1) {
		auto p0 = Vector3{ face_vertex_positions[i * 9 + 0], face_vertex_positions[i * 9 + 1], face_vertex_positions[i * 9 + 2] };
		auto p1 = Vector3{ face_vertex_positions[i * 9 + 3], face_vertex_positions[i * 9 + 4], face_vertex_positions[i * 9 + 5] };
		auto p2 = Vector3{ face_vertex_positions[i * 9 + 6], face_vertex_positions[i * 9 + 7], face_vertex_positions[i * 9 + 8] };
		
		auto p20 = p2 - p0;
		auto n0 = CrossProduct(p2 - p1, p20);
		
		bool is_non_zero_area = DotProduct(n0, n0) > FLT_EPSILON;
		for (u32 i = 0; i < edge_collapse_candidate_position_count; i += 1) {
			auto n1 = CrossProduct(p2 - candidate_positions[i], p20); // p1 is replaced with the candidate vertex.
			
			// Prevent flipped face normals and zero area faces.
			bool reject_edge_collapse = (DotProduct(n0, n1) < 0.f) || (is_non_zero_area && DotProduct(n1, n1) < FLT_EPSILON);
			
			if (reject_edge_collapse) {
				valid_position_mask &= ~(1u << i);
			}
		}
	}
#endif // GetSimdWidth() == 1
	
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
	
	QuadricWithAttributes& operator[] (u32 index) { MDT_ASSERT(index < count); return *(QuadricWithAttributes*)((u8*)data + index * data_stride_bytes); }
};

// TODO: Reuse existing array functions.
never_inline_function static void ArrayGrow(QuadricWithAttributesArray& array, Allocator& allocator, u32 new_capacity) {
	u32 old_memory_block_index = AllocatorFindMemoryBlock(allocator, array.data);
	
	void* memory_block = allocator.callbacks.reallocate(array.data, new_capacity * array.data_stride_bytes + sizeof(QuadricWithAttributes), allocator.callbacks.user_data);
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


struct alignas(MDT_CACHE_LINE_SIZE) MeshDecimationState {
	// Edge quadrics accumulated on vertices.
	Array<Quadric> vertex_edge_quadrics;
	
	// Face quadrics accumulated on attributes.
	QuadricWithAttributesArray attribute_face_quadrics;
	
	EdgeDuplicateMap edge_duplicate_map;
	Array<EdgeID>    removed_edge_array;
	
	QuadricWithAttributesArray wedge_quadrics;
	Array<AttributesID> wedge_attributes_ids;
	AttributeWedgeMap wedge_attribute_set;
	
	float attribute_weights[MDT_MAX_ATTRIBUTE_STRIDE_DWORDS];
	float rcp_attribute_weights[MDT_MAX_ATTRIBUTE_STRIDE_DWORDS];
	
	float position_weight;
	float rcp_position_weight;
};

struct EdgeCollapseError {
	float min_error;
	Vector3 new_position;
};

static EdgeCollapseError ComputeEdgeCollapseError(const EditableMeshView& mesh, Allocator& heap_allocator, MeshDecimationState& state, EdgeID edge_id) {
	state.wedge_quadrics.count       = 0;
	state.wedge_attributes_ids.count = 0;
	state.wedge_attribute_set.count  = 0;
	
	auto& edge = mesh[edge_id];
	u32 attribute_stride_dwords = mesh.attribute_stride_dwords;
	
	// Wedges spanning collapsed edge must be unified. Manually set their wedge index to the same value and accumulate quadrics.
	// For reference see [Hugues Hoppe 1999] Section 5 Attribute Discontinuities, Figure 5.
	IterateCornerList<EdgeID>(mesh, edge.corner_list_base, [&](CornerID corner_id) {
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
	
	alignas(MDT_CACHE_LINE_SIZE) float face_vertex_positions[max_edge_collapse_validation_face_count * 9];
	u32 face_index = 0;
	
	auto accumulate_quadrics = [&](CornerID corner_id) {
		auto& corner = mesh[corner_id];
		auto attribute_id = corner.attributes_id;
		
		if (AttributeWedgeMapFind(state.wedge_attribute_set, attribute_id) == u32_max) {
			AttributeWedgeMapAdd(state.wedge_attribute_set,  attribute_id,   state.wedge_quadrics.count);
			ArrayAppendMaybeGrow(state.wedge_quadrics,       heap_allocator, state.attribute_face_quadrics[attribute_id.index]);
			ArrayAppendMaybeGrow(state.wedge_attributes_ids, heap_allocator, attribute_id);
		}
		
		auto v0 = mesh[corner.corner_list_around[(u32)ElementType::Face].prev].vertex_id;
		auto v1 = corner.vertex_id; // v1 is always the vertex being replaced with a candidate.
		auto v2 = mesh[corner.corner_list_around[(u32)ElementType::Face].next].vertex_id;
		
		bool is_collapsed_face =
			(v0.index == edge.vertex_0.index || v0.index == edge.vertex_1.index) ||
			(v2.index == edge.vertex_0.index || v2.index == edge.vertex_1.index);
		
		if (is_collapsed_face == false && face_index < 32) {
			auto p0 = mesh[v0].position;
			auto p1 = mesh[v1].position;
			auto p2 = mesh[v2].position;
			
#if GetSimdWidth() != 1
			compile_const u32 stride = max_edge_collapse_validation_face_count;
			auto* positions = face_vertex_positions + face_index;
			VectorStoreSOA(positions + stride * 0, p0, stride);
			VectorStoreSOA(positions + stride * 3, p1, stride);
			VectorStoreSOA(positions + stride * 6, p2, stride);
#else // GetSimdWidth() == 1
			auto* positions = face_vertex_positions + face_index * 9;
			VectorStoreSOA(positions + 0, p0, 1);
			VectorStoreSOA(positions + 3, p1, 1);
			VectorStoreSOA(positions + 6, p2, 1);
#endif // GetSimdWidth() == 1
			
			face_index += 1;
		}
	};
	
	auto& v0 = mesh[edge.vertex_0];
	auto& v1 = mesh[edge.vertex_1];
	
	IterateCornerList<VertexID>(mesh, v0.corner_list_base, accumulate_quadrics);
	IterateCornerList<VertexID>(mesh, v1.corner_list_base, accumulate_quadrics);
	
	EdgeCollapseError collapse_error;
	collapse_error.min_error = FLT_MAX;
	{
		auto edge_quadrics = state.vertex_edge_quadrics[edge.vertex_0.index];
		AccumulateQuadric(edge_quadrics, state.vertex_edge_quadrics[edge.vertex_1.index]);
		
		// Try a few different positions for the new vertex.
		Vector3 candidate_positions[edge_collapse_candidate_position_count];
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
		
		// ~30% of the execution time.
		u32 valid_position_mask = ValidateEdgeCollapsePositions(candidate_positions, face_vertex_positions, face_index);
		
		for (u32 i = 0; i < edge_collapse_candidate_position_count; i += 1) {
			float error = valid_position_mask & (1u << i) ? 0.f : total_quadric.weight;
			if (error > collapse_error.min_error) continue;
			
			auto p = candidate_positions[i] * state.position_weight;
			error += ComputeQuadricError(edge_quadrics, p);
			if (error > collapse_error.min_error) continue;
			
			for (u32 wedge_index = 0; wedge_index < wedge_count; wedge_index += 1) {
				error += ComputeQuadricErrorWithAttributes(state.wedge_quadrics[wedge_index], p, attribute_stride_dwords);
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
	MDT_ASSERT(heap.edge_collapse_errors.count != 0);
	
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
	MDT_PROFILER_SCOPE("EdgeCollapseHeapInitialize");
	
	if (heap.edge_collapse_errors.count <= 1) return;
	
	u32 node_index = HeapParentIndex(heap.edge_collapse_errors.count - 1);
	
	for (u32 i = node_index; i > 0; i -= 1) {
		EdgeCollapseHeapSiftDown(heap, i);
	}
	EdgeCollapseHeapSiftDown(heap, 0);
}

static void AllocateMeshDecimationState(u32 vertex_count, u32 attribute_count, u32 attribute_stride_dwords, Allocator& allocator, Allocator& heap_allocator, MeshDecimationState& state) {
	ArrayReserve(state.vertex_edge_quadrics, allocator, vertex_count);
	ArrayReserve(state.attribute_face_quadrics, allocator, attribute_count, attribute_stride_dwords);
	
	ArrayReserve(state.wedge_quadrics,       heap_allocator, 64, attribute_stride_dwords);
	ArrayReserve(state.wedge_attributes_ids, heap_allocator, 64);
	ArrayReserve(state.removed_edge_array,   heap_allocator, 64);
	HashTableGrow(state.edge_duplicate_map,  heap_allocator, ComputeHashTableSize(128u));
}

static void InitializeMeshDecimationState(const EditableMeshView& mesh, const MdtTriangleMeshDesc& mesh_desc, MeshDecimationState& state) {
	MDT_PROFILER_SCOPE("InitializeMeshDecimationState");
	
	state.vertex_edge_quadrics.count = mesh.vertex_count;
	state.attribute_face_quadrics.count = mesh.attribute_count;
	
	memset(state.vertex_edge_quadrics.data, 0, mesh.vertex_count * sizeof(Quadric));
	memset(state.attribute_face_quadrics.data, 0, mesh.attribute_count * state.attribute_face_quadrics.data_stride_bytes);
	
	
	auto* attribute_weights = mesh_desc.attribute_weights;
	for (u32 i = 0; i < MDT_MAX_ATTRIBUTE_STRIDE_DWORDS; i += 1) {
		float attribute_weight = attribute_weights && i < mesh.attribute_stride_dwords ? attribute_weights[i] : default_attribute_error_weight;
		
		state.attribute_weights[i]     = attribute_weight;
		state.rcp_attribute_weights[i] = 1.f / attribute_weight;
	}
	
	
	{
		MDT_PROFILER_SCOPE("ComputePositionWeight");
		
		float twice_mesh_surface_area = 0.f;
		for (FaceID face_id = { 0 }; face_id.index < mesh.face_count; face_id.index += 1) {
			auto& face = mesh[face_id];
			MDT_ASSERT(face.corner_list_base.index != u32_max);
			
			auto& c1 = mesh[face.corner_list_base];
			auto& c0 = mesh[c1.corner_list_around[(u32)ElementType::Face].prev];
			auto& c2 = mesh[c1.corner_list_around[(u32)ElementType::Face].next];
			
			auto p0 = mesh[c0.vertex_id].position;
			auto p1 = mesh[c1.vertex_id].position;
			auto p2 = mesh[c2.vertex_id].position;
			
			twice_mesh_surface_area += Length(CrossProduct(p1 - p0, p2 - p0));
		}
		
		//
		// Scale the mesh such that average face surface area is equal to mesh_desc.geometric_weight^2.
		// See [Karis 2021] for reference.
		//
		float face_surface_area       = mesh.face_count ? twice_mesh_surface_area * 0.5f / (float)mesh.face_count : 1.f;
		float rcp_target_face_size    = 1.f / (mesh_desc.geometric_weight > FLT_EPSILON ? mesh_desc.geometric_weight : default_geometric_error_weight);
		float rcp_mesh_position_scale = sqrtf(face_surface_area) * rcp_target_face_size;
		float mesh_position_scale     = 1.f / rcp_mesh_position_scale;
		
		bool is_valid_position_scale = (rcp_mesh_position_scale > FLT_EPSILON);
		state.position_weight     = is_valid_position_scale ? mesh_position_scale     : 1.f;
		state.rcp_position_weight = is_valid_position_scale ? rcp_mesh_position_scale : 1.f;
	}
	
	{
		MDT_PROFILER_SCOPE("BuildFaceQuadrics");
		
		float position_weight = state.position_weight;
		u32 attribute_stride_dwords = mesh.attribute_stride_dwords;
		for (FaceID face_id = { 0 }; face_id.index < mesh.face_count; face_id.index += 1) {
			auto& face = mesh[face_id];
			MDT_ASSERT(face.corner_list_base.index != u32_max);
			
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
		MDT_PROFILER_SCOPE("BuildEdgeQuadrics");
		
		float position_weight = state.position_weight;
		for (EdgeID edge_id = { 0 }; edge_id.index < mesh.edge_count; edge_id.index += 1) {
			auto& edge = mesh[edge_id];
			MDT_ASSERT(edge.corner_list_base.index != u32_max);
			
			auto& c0 = mesh[edge.corner_list_base];
			auto& c1 = mesh[c0.corner_list_around[(u32)ElementType::Face].next];
			auto& c2 = mesh[c0.corner_list_around[(u32)ElementType::Face].prev];
			
			auto attributes_id_0 = c0.attributes_id;
			auto attributes_id_1 = c1.attributes_id;
			
			u32 geometry_index_0 = mesh[c0.face_id].geometry_index;
			
			u32 edge_degree = 0;
			bool attribute_edge = false;
			IterateCornerList<EdgeID>(mesh, edge.corner_list_base, [&](CornerID corner_id) {
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
			ComputeEdgeQuadric(quadric, p0, p1, p2, 1.f / (float)edge_degree);
			
			AccumulateQuadric(state.vertex_edge_quadrics[v0.index], quadric);
			AccumulateQuadric(state.vertex_edge_quadrics[v1.index], quadric);
		}
	}
}

static float DecimateMeshFaceGroup(
	const EditableMeshView& mesh,
	Allocator& heap_allocator,
	MeshDecimationState& state,
	EdgeCollapseHeap& edge_collapse_heap,
	MdtNormalizeVertexAttributes normalize_vertex_attributes,
	u8* changed_vertex_mask,
	u32 target_face_count,
	u32 active_face_count,
	float target_error_limit = FLT_MAX) {
	MDT_PROFILER_SCOPE("DecimateMeshFaceGroup");
	
	target_error_limit = target_error_limit * state.position_weight;
	target_error_limit = target_error_limit * target_error_limit;
	
	float max_error = 0.f;
	while (active_face_count > target_face_count && edge_collapse_heap.edge_collapse_errors.count) {
		// ~80% of the execution time.
		for (auto& key_value : state.edge_duplicate_map) {
			if (key_value.edge_key == u64_max) continue;
			
			u32 heap_index = edge_collapse_heap.edge_id_to_heap_index[key_value.edge_id.index];
			if (heap_index == u32_max) continue;
			
			auto collapse_error = ComputeEdgeCollapseError(mesh, heap_allocator, state, key_value.edge_id);
			
			EdgeCollapseHeapUpdate(edge_collapse_heap, heap_index, collapse_error.min_error);
		}
		HashTableClear(state.edge_duplicate_map);
		
		
		auto edge_id = EdgeCollapseHeapPop(edge_collapse_heap);
		
		// 2% of the execution time
		auto collapse_error = ComputeEdgeCollapseError(mesh, heap_allocator, state, edge_id);
		if (collapse_error.min_error > target_error_limit) break;
		
		max_error = max_error < collapse_error.min_error ? collapse_error.min_error : max_error;
		
		// 15% of the execution time
		auto collapse_result = PerformEdgeCollapse(mesh, edge_id, heap_allocator, state.edge_duplicate_map, state.removed_edge_array);
		active_face_count -= collapse_result.removed_face_count;
		
		for (auto removed_edge_id : state.removed_edge_array) {
			u32 heap_index = edge_collapse_heap.edge_id_to_heap_index[removed_edge_id.index];
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
		
#if MDT_ENABLE_ATTRIBUTE_SUPPORT
		// Update attributes.
		if (collapse_result.remaining_vertex_id.index != u32_max) {
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
				changed_vertex_mask[collapse_result.remaining_vertex_id.index] = 0xFF;
			}
			
			IterateCornerList<VertexID>(mesh, mesh[collapse_result.remaining_vertex_id].corner_list_base, [&](CornerID corner_id) {
				u32 index = AttributeWedgeMapFind(state.wedge_attribute_set, mesh[corner_id].attributes_id);
				if (index != u32_max) mesh[corner_id].attributes_id = state.wedge_attributes_ids[index];
			});
		}
#endif // MDT_ENABLE_ATTRIBUTE_SUPPORT
	}
	
	HashTableClear(state.edge_duplicate_map);
	state.removed_edge_array.count = 0;
	
	return sqrtf(max_error) * state.rcp_position_weight;
}

struct DecimationThreadContext {
	EditableMeshView sub_mesh;
	MeshDecimationState state;
	
	Allocator heap_allocator;
	Allocator allocator;
	
	Array<Face>   sub_mesh_faces;
	Array<Edge>   sub_mesh_edges;
	Array<Vertex> sub_mesh_vertices;
	Array<Corner> sub_mesh_corners;
	Array<float>  sub_mesh_attributes;
	
	EdgeCollapseHeap edge_collapse_heap;
	
	EdgeHashTable edge_table;
	
	Array<VertexID> vertex_id_to_sub_mesh_vertex_id;
	Array<VertexID> sub_mesh_vertex_id_to_vertex_id;
	
	Array<AttributesID> attributes_id_to_sub_mesh_attributes_id;
	Array<AttributesID> sub_mesh_attributes_id_to_attributes_id;
	
	Array<u8> sub_mesh_vertex_is_locked;
	Array<u8> sub_mesh_changed_vertex_mask;
};

static void AllocateDecimationThreadContext(DecimationThreadContext& context, const IndexedMeshView& mesh, Allocator& allocator, Allocator& heap_allocator) {
	compile_const u32 max_vertex_count = meshlet_group_max_meshlet_count * meshlet_max_vertex_count;
	compile_const u32 max_face_count = meshlet_group_max_meshlet_count * meshlet_max_face_count;
	compile_const u32 max_corner_count = max_face_count * meshlet_max_face_degree;
	compile_const u32 max_edge_count = max_face_count * meshlet_max_face_degree;
	
	context.heap_allocator.callbacks = heap_allocator.callbacks;
	context.allocator.callbacks = allocator.callbacks;
	
	ArrayReserve(context.sub_mesh_vertices, context.allocator, max_vertex_count);
	ArrayReserve(context.sub_mesh_corners, context.allocator, max_corner_count);
	ArrayReserve(context.sub_mesh_faces, context.allocator, max_face_count);
	ArrayReserve(context.sub_mesh_edges, context.allocator, max_edge_count);
	ArrayReserve(context.sub_mesh_attributes, context.allocator, max_corner_count * mesh.attribute_stride_dwords);
	
	context.sub_mesh.faces      = context.sub_mesh_faces.data;
	context.sub_mesh.edges      = context.sub_mesh_edges.data;
	context.sub_mesh.vertices   = context.sub_mesh_vertices.data;
	context.sub_mesh.corners    = context.sub_mesh_corners.data;
	context.sub_mesh.attributes = context.sub_mesh_attributes.data;
	context.sub_mesh.attribute_stride_dwords = mesh.attribute_stride_dwords;
	
	ArrayResize(context.edge_collapse_heap.edge_collapse_errors,  context.allocator, max_edge_count);
	ArrayResize(context.edge_collapse_heap.edge_id_to_heap_index, context.allocator, max_edge_count);
	ArrayResize(context.edge_collapse_heap.heap_index_to_edge_id, context.allocator, max_edge_count);
	
	ArrayResize(context.edge_table.edge_ids, context.allocator, ComputeHashTableSize(max_edge_count));
	
	ArrayResizeMemset(context.vertex_id_to_sub_mesh_vertex_id, context.allocator, mesh.vertex_count, 0xFF);
	ArrayResize(context.sub_mesh_vertex_id_to_vertex_id, context.allocator, max_vertex_count);
	
	ArrayResizeMemset(context.attributes_id_to_sub_mesh_attributes_id, context.allocator, mesh.attribute_count, 0xFF);
	ArrayResize(context.sub_mesh_attributes_id_to_attributes_id, context.allocator, max_corner_count);
	
	ArrayResize(context.sub_mesh_vertex_is_locked, context.allocator, max_vertex_count);
	ArrayResizeMemset(context.sub_mesh_changed_vertex_mask, context.allocator, max_vertex_count, 0);
	
	AllocateMeshDecimationState(max_vertex_count, max_corner_count, mesh.attribute_stride_dwords, context.allocator, context.heap_allocator, context.state);
}

static void DeallocateDecimationThreadContext(DecimationThreadContext& context) {
	AllocatorFreeMemoryBlocks(context.allocator, 0);
	AllocatorFreeMemoryBlocks(context.heap_allocator, 0);
}

static void DecimateMeshFaceGroups(
	const IndexedMeshView& mesh,
	Allocator& allocator,
	Allocator& heap_allocator,
	const MdtTriangleMeshDesc& mesh_desc,
	const MdtParallelForCallbacks* parallel_for,
	Array<u32> meshlet_group_face_prefix_sum,
	Array<MdtErrorMetric> meshlet_group_error_metrics,
	Array<u8> changed_vertex_mask) {
	
	MDT_PROFILER_SCOPE("DecimateMeshFaceGroups");
	
	u32 allocator_high_water = allocator.memory_block_count;
	u32 heap_allocator_high_water = heap_allocator.memory_block_count;
	
	compile_const u32 vertex_group_index_locked = u32_max - 1;
	
	Array<u32> vertex_group_indices;
	ArrayResizeMemset(vertex_group_indices, allocator, mesh.vertex_count, 0xFF);
	
	{
		MDT_PROFILER_SCOPE("FindSharedVertices");
		
		for (u32 group_index = 0, begin_face_index = 0; group_index < meshlet_group_face_prefix_sum.count; group_index += 1) {
			u32 end_face_index = meshlet_group_face_prefix_sum[group_index];
			
			for (u32 face_index = begin_face_index; face_index < end_face_index; face_index += 1) {
				auto face_id = FaceID{ face_index };
				
				for (u32 i = 0; i < 3; i += 1) {
					auto vertex_id = mesh.face_vertex_ids[face_id.index * 3 + i];
					u32& index = vertex_group_indices[vertex_id.index];
					
					if (index == u32_max) {
						index = group_index;
					} else if (index != group_index) {
						index = vertex_group_index_locked; // Lock the vertex.
					}
				}
			}
			
			begin_face_index = end_face_index;
		}
	}
	
	u32 face_groups_per_thread = DivideAndRoundUp(meshlet_group_face_prefix_sum.count, ParallelForThreadCount(parallel_for));
	u32 work_item_count        = DivideAndRoundUp(meshlet_group_face_prefix_sum.count, face_groups_per_thread);
	
	ParallelFor(parallel_for, work_item_count, [&](u32 work_item_index) {
		DecimationThreadContext context;
		AllocateDecimationThreadContext(context, mesh, allocator, heap_allocator);
		
		u32 begin_group_index = work_item_index * face_groups_per_thread;
		u32 end_group_index   = Min(begin_group_index + face_groups_per_thread, meshlet_group_face_prefix_sum.count);
		
		u32 begin_face_index = begin_group_index ? meshlet_group_face_prefix_sum[begin_group_index - 1] : 0;
		for (u32 group_index = begin_group_index; group_index < end_group_index; group_index += 1) {
			MDT_PROFILER_SCOPE("ProcessFaceGroup");
			
			u32 end_face_index = meshlet_group_face_prefix_sum[group_index];
			
			u32 face_count = end_face_index - begin_face_index;
			
			memset(context.edge_table.edge_ids.data, 0xFF, context.edge_table.edge_ids.count * sizeof(EdgeID));
			context.sub_mesh_faces.count = 0;
			context.sub_mesh_edges.count = 0;
			context.sub_mesh_vertices.count = 0;
			context.sub_mesh_corners.count = face_count * 3;
			context.sub_mesh_attributes.count = 0;
			context.sub_mesh_vertex_is_locked.count = 0;
			context.sub_mesh_vertex_id_to_vertex_id.count = 0;
			context.sub_mesh_attributes_id_to_attributes_id.count = 0;
			
			for (u32 face_index = begin_face_index; face_index < end_face_index; face_index += 1) {
				auto source_face_id = FaceID{ face_index };
				
				for (u32 i = 0; i < 3; i += 1) {
					auto vertex_id = mesh.face_vertex_ids[source_face_id.index * 3 + i];
					auto attributes_id = mesh.face_attribute_ids[source_face_id.index * 3 + i];
					
					auto sub_mesh_corner_id = CornerID{ (face_index - begin_face_index) * 3 + i };
					auto& sub_mesh_vertex_id = context.vertex_id_to_sub_mesh_vertex_id[vertex_id.index];
					auto& sub_mesh_attributes_id = context.attributes_id_to_sub_mesh_attributes_id[attributes_id.index];
					
					if (sub_mesh_vertex_id.index == u32_max) {
						sub_mesh_vertex_id.index = context.sub_mesh_vertices.count;
						
						Vertex vertex;
						vertex.position = mesh[vertex_id];
						vertex.corner_list_base.index = u32_max;
						
						ArrayAppend(context.sub_mesh_vertices, vertex);
						ArrayAppend(context.sub_mesh_vertex_id_to_vertex_id, vertex_id);
						ArrayAppend(context.sub_mesh_vertex_is_locked, vertex_group_indices[vertex_id.index] == vertex_group_index_locked ? 1 : 0);
					}
					
					if (sub_mesh_attributes_id.index == u32_max) {
						sub_mesh_attributes_id.index = context.sub_mesh_attributes.count / context.sub_mesh.attribute_stride_dwords;
						
						auto* attributes = mesh[attributes_id];
						memcpy(context.sub_mesh_attributes.end(), attributes, context.sub_mesh.attribute_stride_dwords * sizeof(u32));
						context.sub_mesh_attributes.count += context.sub_mesh.attribute_stride_dwords;
						
						ArrayAppend(context.sub_mesh_attributes_id_to_attributes_id, attributes_id);
					}
				}
			}
			
			for (u32 face_index = begin_face_index; face_index < end_face_index; face_index += 1) {
				auto source_face_id = FaceID{ face_index };
				
				VertexID vertex_ids[3] = {
					context.vertex_id_to_sub_mesh_vertex_id[mesh.face_vertex_ids[source_face_id.index * 3 + 0].index],
					context.vertex_id_to_sub_mesh_vertex_id[mesh.face_vertex_ids[source_face_id.index * 3 + 1].index],
					context.vertex_id_to_sub_mesh_vertex_id[mesh.face_vertex_ids[source_face_id.index * 3 + 2].index],
				};
				
				AttributesID attributes_ids[3] = {
					context.attributes_id_to_sub_mesh_attributes_id[mesh.face_attribute_ids[source_face_id.index * 3 + 0].index],
					context.attributes_id_to_sub_mesh_attributes_id[mesh.face_attribute_ids[source_face_id.index * 3 + 1].index],
					context.attributes_id_to_sub_mesh_attributes_id[mesh.face_attribute_ids[source_face_id.index * 3 + 2].index],
				};
				
				u64 edge_keys[3] = {
					PackEdgeKey(vertex_ids[0], vertex_ids[1]),
					PackEdgeKey(vertex_ids[1], vertex_ids[2]),
					PackEdgeKey(vertex_ids[2], vertex_ids[0]),
				};
				
				auto face_id = FaceID{ context.sub_mesh_faces.count };
				
				Face face;
				face.corner_list_base.index = u32_max;
				face.geometry_index = mesh.face_geometry_indices[source_face_id.index];
				ArrayAppend(context.sub_mesh_faces, face);
				
				for (u32 corner_index = 0; corner_index < 3; corner_index += 1) {
					auto corner_id = CornerID{ face_id.index * 3 + corner_index };
					
					auto edge_id = HashTableAddOrFind(context.edge_table, context.sub_mesh_edges, edge_keys[corner_index]);
					
					auto& corner = context.sub_mesh_corners[corner_id.index];
					corner.face_id       = face_id;
					corner.edge_id       = edge_id;
					corner.vertex_id     = vertex_ids[corner_index];
					corner.attributes_id = attributes_ids[corner_index];
					
					CornerListInsert<VertexID>(context.sub_mesh, corner.vertex_id, corner_id);
					CornerListInsert<EdgeID>(context.sub_mesh, corner.edge_id, corner_id);
					CornerListInsert<FaceID>(context.sub_mesh, corner.face_id, corner_id);
				}
			}
			
			context.sub_mesh.face_count = context.sub_mesh_faces.count;
			context.sub_mesh.edge_count = context.sub_mesh_edges.count;
			context.sub_mesh.vertex_count = context.sub_mesh_vertices.count;
			context.sub_mesh.corner_count = context.sub_mesh_corners.count;
			context.sub_mesh.attribute_count = context.sub_mesh_attributes.count / context.sub_mesh.attribute_stride_dwords;
			
			u32 edge_count = context.sub_mesh_edges.count;
			context.edge_collapse_heap.edge_collapse_errors.count  = edge_count;
			context.edge_collapse_heap.edge_id_to_heap_index.count = edge_count;
			context.edge_collapse_heap.heap_index_to_edge_id.count = edge_count;
			
			InitializeMeshDecimationState(context.sub_mesh, mesh_desc, context.state);
			
			{
				MDT_PROFILER_SCOPE("BuildEdgeHeap");
				
				u32 local_edge_index = 0;
				for (EdgeID edge_id = { 0 }; edge_id.index < context.sub_mesh_edges.count; edge_id.index += 1) {
					auto& edge = context.sub_mesh[edge_id];
					bool edge_is_locked = context.sub_mesh_vertex_is_locked[edge.vertex_0.index] || context.sub_mesh_vertex_is_locked[edge.vertex_1.index];
					
					if (edge_is_locked == false) {
						auto collapse_error = ComputeEdgeCollapseError(context.sub_mesh, context.heap_allocator, context.state, edge_id);
						
						context.edge_collapse_heap.edge_collapse_errors[local_edge_index]  = collapse_error.min_error;
						context.edge_collapse_heap.edge_id_to_heap_index[edge_id.index]    = local_edge_index;
						context.edge_collapse_heap.heap_index_to_edge_id[local_edge_index] = edge_id;
						local_edge_index += 1;
					} else {
						context.edge_collapse_heap.edge_id_to_heap_index[edge_id.index] = u32_max;
					}
				}
				context.edge_collapse_heap.edge_collapse_errors.count  = local_edge_index;
				context.edge_collapse_heap.heap_index_to_edge_id.count = local_edge_index;
				
				EdgeCollapseHeapInitialize(context.edge_collapse_heap);
			}
			
			context.sub_mesh_changed_vertex_mask.count = context.sub_mesh.vertex_count;
			
			
			u32 target_face_count = face_count / 2;
			u32 active_face_count = face_count;
			float decimation_error = DecimateMeshFaceGroup(
				context.sub_mesh,
				context.heap_allocator,
				context.state,
				context.edge_collapse_heap,
				mesh_desc.normalize_vertex_attributes,
				context.sub_mesh_changed_vertex_mask.data,
				target_face_count,
				active_face_count
			);
			
			auto& error_metric = meshlet_group_error_metrics[group_index];
			error_metric.error = error_metric.error > decimation_error ? error_metric.error : decimation_error;
			
			for (auto& vertex_id : context.sub_mesh_vertex_id_to_vertex_id) {
				context.vertex_id_to_sub_mesh_vertex_id[vertex_id.index].index = u32_max;
			}
			
			for (auto& attributes_id : context.sub_mesh_attributes_id_to_attributes_id) {
				context.attributes_id_to_sub_mesh_attributes_id[attributes_id.index].index = u32_max;
			}
			
			for (VertexID vertex_id = { 0 }; vertex_id.index < context.sub_mesh.vertex_count; vertex_id.index += 1) {
				if (context.sub_mesh_changed_vertex_mask[vertex_id.index] == 0) continue;
				context.sub_mesh_changed_vertex_mask[vertex_id.index] = 0;
				
				auto& vertex = context.sub_mesh[vertex_id];
				
				auto source_vertex_id = context.sub_mesh_vertex_id_to_vertex_id[vertex_id.index];
				mesh[source_vertex_id] = vertex.position;
				
				if (vertex.corner_list_base.index == u32_max) {
					context.sub_mesh_vertex_id_to_vertex_id[vertex_id.index].index = u32_max;
				}
				
				changed_vertex_mask[source_vertex_id.index] = 0xFF;
			}
			
			for (AttributesID attributes_id = { 0 }; attributes_id.index < context.sub_mesh.attribute_count; attributes_id.index += 1) {
				auto source_attributes_id = context.sub_mesh_attributes_id_to_attributes_id[attributes_id.index];
				auto* attributes = context.sub_mesh[attributes_id];
				
				memcpy(mesh[source_attributes_id], attributes, context.sub_mesh.attribute_stride_dwords * sizeof(float));
			}
			
			for (FaceID face_id = { 0 }; face_id.index < context.sub_mesh.face_count; face_id.index += 1) {
				auto source_face_id = FaceID{ face_id.index + begin_face_index };
				auto& face = context.sub_mesh[face_id];
				
				if (face.corner_list_base.index != u32_max) {
					u32 corner_index = 0;
					IterateCornerList<FaceID>(context.sub_mesh, face.corner_list_base, [&](CornerID corner_id) {
						auto& corner = context.sub_mesh[corner_id];
						mesh.face_vertex_ids[source_face_id.index * 3 + corner_index] = context.sub_mesh_vertex_id_to_vertex_id[corner.vertex_id.index];
						mesh.face_attribute_ids[source_face_id.index * 3 + corner_index] = context.sub_mesh_attributes_id_to_attributes_id[corner.attributes_id.index];
						corner_index += 1;
					});
				} else {
					for (u32 corner_index = 0; corner_index < 3; corner_index += 1) {
						mesh.face_vertex_ids[source_face_id.index * 3 + corner_index].index = u32_max;
						mesh.face_attribute_ids[source_face_id.index * 3 + corner_index].index = u32_max;
					}
				}
			}
			
			begin_face_index = end_face_index;
		}
		
		DeallocateDecimationThreadContext(context);
	});
	
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
static_assert(sizeof(KdTreeElement) == 16, "Invalid KdTreeElement size.");

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
	float min[3] = { +FLT_MAX, +FLT_MAX, +FLT_MAX };
	float max[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
	float indices_count = 0.f;
	
	for (u32 i = 0; i < indices.count; i += 1) {
		auto& element = elements[indices[i]];
		
		for (u32 j = 0; j < 3; j += 1) {
			float position_j = LoadElementByIndex(element.position, j);
			min[j] = min[j] < position_j ? min[j] : position_j;
			max[j] = max[j] > position_j ? max[j] : position_j;
		}
		sum = sum + element.position;
		indices_count += 1.f;
	}
	
	auto mean = (indices_count > 0.f) ? sum * (1.f / indices_count) : sum;
	float extent[3] = { max[0] - min[0], max[1] - min[1], max[2] - min[2] };
	
	u32 split_axis = 0;
	for (u32 axis = 1; axis < 3; axis += 1) {
		if (extent[axis] > extent[split_axis]) split_axis = axis;
	}
	float split_position = LoadElementByIndex(mean, split_axis);
	
	
	u32 split_index = 0;
	for (u32 i = 0; i < indices.count; i += 1) {
		u32 index = indices[i];
		float position = LoadElementByIndex(elements[index].position, split_axis);

		// Swap(indices[i], indices[split_index]);
		indices[i] = indices[split_index];
		indices[split_index] = index;

		if (position < split_position) split_index += 1;
	}
	
	node.split = split_position;
	node.axis  = split_axis;
	
	return split_index;
}

static u32 KdTreeBuildLeafNode(Array<KdTreeNode>& nodes, const ArrayView<u32>& indices) {
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
		return KdTreeBuildLeafNode(nodes, indices);
	}
	
	KdTreeNode node;
	u32 split_index = KdTreeSplit(elements, indices, node);
	
	// Faild to split the subtree. Create a single leaf to prevent infinite recursion.
	if (split_index == 0 || split_index >= indices.count) {
		return KdTreeBuildLeafNode(nodes, indices);
	}
	
	u32 node_index = nodes.count;
	ArrayAppend(nodes, node);
	
	u32 node_index_0 = KdTreeBuildNode(nodes, elements, CreateArrayView(indices, 0, split_index));
	u32 node_index_1 = KdTreeBuildNode(nodes, elements, CreateArrayView(indices, split_index, indices.count));
	
	MDT_ASSERT(node_index_0 == node_index + 1); // Left node is always the next node after the local root.
	MDT_ASSERT(node_index_1 > node_index);      // Right node offset is non zero. Zero means branch is pruned.
	MDT_UNUSED_VARIABLE(node_index_0);
	
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
		float delta  = LoadElementByIndex(point, node.axis) - node.split;
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

static void KdTreeBuildElementsForFaces(const IndexedMeshView& mesh, KdTree& kd_tree, u32 begin_element_index, u32 end_element_index) {
	MDT_PROFILER_SCOPE("KdTreeBuildElementsForFaces");
	
	u32 element_count = end_element_index - begin_element_index;
	for (u32 i = 0; i < element_count; i += 1) {
		auto face_id = FaceID{ i + begin_element_index };
		
		auto position_sum =
			mesh[mesh.face_vertex_ids[face_id.index * 3 + 0]] +
			mesh[mesh.face_vertex_ids[face_id.index * 3 + 1]] +
			mesh[mesh.face_vertex_ids[face_id.index * 3 + 2]];
		
		auto& element = kd_tree.elements[i];
		element.position          = position_sum * (1.f / 3.f);
		element.is_active_element = 1;
		element.partition_index   = mesh.face_geometry_indices[face_id.index];
		kd_tree.element_indices[i] = i;
	}
}

static void KdTreeBuildElementsForMeshlets(ArrayView<MdtMeshlet> meshlets, Allocator& allocator, Array<KdTreeElement>& elements) {
	MDT_PROFILER_SCOPE("KdTreeBuildElementsForMeshlets");
	
	ArrayResize(elements, allocator, meshlets.count);
	
	for (u32 meshlet_index = 0; meshlet_index < meshlets.count; meshlet_index += 1) {
		auto& element = elements[meshlet_index];
		auto& meshlet = meshlets[meshlet_index];
		
		element.position = (meshlet.aabb_min + meshlet.aabb_max) * 0.5f;
		element.is_active_element = 1;
		element.partition_index   = 0;
	}
}

static MdtSphereBounds ComputeSphereBoundsUnion(ArrayView<MdtSphereBounds> source_sphere_bounds) {
	return MdtComputeSphereBoundsUnion(source_sphere_bounds.data, source_sphere_bounds.count);
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
	ArrayView<MdtMeshlet> meshlets;
	
	ArrayView<u8>       meshlet_triangles;
	ArrayView<CornerID> meshlet_corners;
	ArrayView<u32>      meshlet_corner_prefix_sum;
	
	MeshletAdjacency meshlet_adjacency;
};

static MeshletAdjacency BuildMeshletAdjacency(
	const IndexedMeshView& mesh,
	Allocator& allocator,
	ArrayView<u32> corner_list_around_vertex,
	ArrayView<u32> corner_list_around_vertex_prefix_sum,
	ArrayView<CornerID> meshlet_corners,
	ArrayView<u32> meshlet_corner_prefix_sum,
	ArrayView<KdTreeElement> kd_tree_elements);

//
// Based on [Kapoulkine 2025].
//
static void BuildMeshletsForFaceGroup(
	const IndexedMeshView& mesh,
	KdTree kd_tree,
	u32 begin_face_index,
	u32 end_face_index,
	u32 meshlet_target_face_count,
	u32 meshlet_target_vertex_count,
	ArrayView<u32>   corner_list_around_vertex,
	ArrayView<u32>   corner_list_around_vertex_prefix_sum,
	ArrayView<u8>    vertex_usage_map,
	Array<FaceID>&   meshlet_faces,
	Array<u8>&       meshlet_triangles,
	Array<u32>&      meshlet_face_prefix_sum,
	Array<CornerID>& meshlet_corners,
	Array<u32>&      meshlet_corner_prefix_sum) {
	
	MDT_PROFILER_SCOPE("BuildMeshletsForFaceGroup");
	
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
			
			auto& element = kd_tree.elements[face_id.index - begin_face_index];
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
			for (u32 corner_index = 0; corner_index < 3; corner_index += 1) {
				auto corner_id = CornerID{ face_id.index * 3 + corner_index };
				auto attributes_id = mesh.face_attribute_ids[corner_id.index];
				
				u8 vertex_index = vertex_usage_map[attributes_id.index];
				if (vertex_index == 0xFF) new_vertex_count += 1;
			}
			
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
			u32 best_element_index = u32_max;
			kd_tree_is_empty = KdTreeFindClosestActiveElement(kd_tree, center, meshlet_geometry_index, best_element_index, min_distance);
			
			best_face_id.index = best_element_index == u32_max ? u32_max : (best_element_index + begin_face_index);
			
			if (best_face_id.index != u32_max && meshlet_face_count >= meshlet_min_face_count) {
				auto& element = kd_tree.elements[best_face_id.index - begin_face_index];
				
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
			for (u32 i = 0; i < 3; i += 1) {
				auto attributes_id = mesh.face_attribute_ids[best_face_id.index * 3 + i];
				u8 vertex_index = vertex_usage_map[attributes_id.index];
				if (vertex_index == 0xFF) new_vertex_count += 1;
			}
		}
		
		if (restart_meshlet || best_face_id.index == u32_max || (meshlet_vertex_count + new_vertex_count > meshlet_target_vertex_count) || (meshlet_face_count + 1 > meshlet_target_face_count)) {
			MDT_ASSERT(meshlet_face_count   <= meshlet_target_face_count);
			MDT_ASSERT(meshlet_vertex_count <= meshlet_target_vertex_count);
			
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
		
		u32 best_face_geometry_index = mesh.face_geometry_indices[best_face_id.index];
		MDT_ASSERT(meshlet_face_count == 0 || meshlet_geometry_index == best_face_geometry_index);
		
		new_vertex_count = 0;
		
		for (u32 i = 0; i < 3; i += 1) {
			auto vertex_id     = mesh.face_vertex_ids[best_face_id.index * 3 + i];
			auto attributes_id = mesh.face_attribute_ids[best_face_id.index * 3 + i];
			
			u8 vertex_index = vertex_usage_map[attributes_id.index];
			if (vertex_index == 0xFF) {
				vertex_index = (u8)(meshlet_vertex_count + new_vertex_count);
				vertex_usage_map[attributes_id.index] = vertex_index;
				
				new_vertex_count += 1;
				
				ArrayAppend(meshlet_vertices, attributes_id);
				ArrayAppend(meshlet_corners, CornerID{ best_face_id.index * 3 + i });
			}
			
			ArrayAppend(meshlet_triangles, vertex_index);
			
			u32 corner_list_begin_index = corner_list_around_vertex_prefix_sum[vertex_id.index];
			u32 corner_list_end_index   = corner_list_around_vertex_prefix_sum[vertex_id.index + 1];
			
			for (u32 corner_list_index = corner_list_begin_index; corner_list_index < corner_list_end_index; corner_list_index += 1) {
				u32 corner_id = corner_list_around_vertex[corner_list_index];
				auto  face_id = FaceID{ corner_id / 3 };
				
				if ((face_id.index != best_face_id.index) &&
					(face_id.index >= begin_face_index) &&
					(face_id.index <  end_face_index) &&
					(kd_tree.elements[face_id.index - begin_face_index].is_active_element != 0) &&
					(kd_tree.elements[face_id.index - begin_face_index].partition_index   == best_face_geometry_index) && 
					(meshlet_candidate_elements.count < meshlet_candidate_elements.capacity)) {
					ArrayAppend(meshlet_candidate_elements, face_id);
				}
			}
		}
		
		ArrayAppend(meshlet_faces, best_face_id);
		meshlet_vertex_count += new_vertex_count;
		meshlet_face_count   += 1;
		MDT_ASSERT(meshlet_face_count   <= meshlet_target_face_count);
		MDT_ASSERT(meshlet_vertex_count <= meshlet_target_vertex_count);
		
		auto& element = kd_tree.elements[best_face_id.index - begin_face_index];
		element.partition_index   = meshlet_face_prefix_sum.count;
		element.is_active_element = 0;
		
		meshlet_geometry_index = best_face_geometry_index;
		
		meshlet_aabb_min = VectorMin(meshlet_aabb_min, element.position);
		meshlet_aabb_max = VectorMax(meshlet_aabb_max, element.position);
	}
	
	if (meshlet_face_count) {
		MDT_ASSERT(meshlet_face_count   <= meshlet_target_face_count);
		MDT_ASSERT(meshlet_vertex_count <= meshlet_target_vertex_count);
		
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
	const IndexedMeshView& mesh,
	const MdtParallelForCallbacks* parallel_for,
	Allocator& allocator,
	Array<u32> meshlet_group_face_prefix_sum,
	Array<MdtErrorMetric> meshlet_group_error_metrics,
	u32 meshlet_target_face_count,
	u32 meshlet_target_vertex_count,
	u32 meshlet_group_base_index) {
	
	MDT_PROFILER_SCOPE("BuildMeshletsForFaceGroups");
	
	KdTree kd_tree_allocations;
	ArrayResize(kd_tree_allocations.elements, allocator, mesh.face_count);
	ArrayReserve(kd_tree_allocations.nodes, allocator, kd_tree_allocations.elements.count * 2);
	ArrayResize(kd_tree_allocations.element_indices, allocator, kd_tree_allocations.elements.count);
	
	Array<u32> corner_list_around_vertex_prefix_sum;
	Array<u32> corner_list_around_vertex;
	
	{
		MDT_PROFILER_SCOPE("BuildVertexCornerLists");
		
		ArrayResizeMemset(corner_list_around_vertex_prefix_sum, allocator, mesh.vertex_count + 1, 0);
		ArrayResize(corner_list_around_vertex, allocator, mesh.face_count * 3);
		
		for (u32 corner_index = 0; corner_index < mesh.face_count * 3; corner_index += 1) {
			auto vertex_id = mesh.face_vertex_ids[corner_index];
			corner_list_around_vertex_prefix_sum[vertex_id.index + 1] += 1;
		}
		
		for (u32 vertex_index = 0, corner_prefix_sum = 0; vertex_index < mesh.vertex_count; vertex_index += 1) {
			u32 corner_count = corner_list_around_vertex_prefix_sum[vertex_index + 1];
			corner_list_around_vertex_prefix_sum[vertex_index + 1] = corner_prefix_sum;
			corner_prefix_sum += corner_count;
		}
		
		for (u32 corner_index = 0; corner_index < mesh.face_count * 3; corner_index += 1) {
			auto vertex_id = mesh.face_vertex_ids[corner_index];
			u32 corner_list_index = corner_list_around_vertex_prefix_sum[vertex_id.index + 1]++;
			corner_list_around_vertex[corner_list_index] = corner_index;
		}
	}
	
	
	Array<FaceID>   meshlet_faces;
	Array<CornerID> meshlet_corners;
	Array<u8>       meshlet_triangles;
	ArrayResize(meshlet_faces,     allocator, mesh.face_count);
	ArrayResize(meshlet_corners,   allocator, mesh.face_count * meshlet_max_face_degree);
	ArrayResize(meshlet_triangles, allocator, mesh.face_count * meshlet_max_face_degree);
	
	Array<u32> meshlet_face_prefix_sum;
	Array<u32> meshlet_corner_prefix_sum;
	ArrayResize(meshlet_face_prefix_sum,   allocator, mesh.face_count);
	ArrayResize(meshlet_corner_prefix_sum, allocator, mesh.face_count);
	
	Array<u32> meshlet_group_meshlet_prefix_sum;
	ArrayResize(meshlet_group_meshlet_prefix_sum, allocator, meshlet_group_face_prefix_sum.count);
	
	u32 face_groups_per_thread  = DivideAndRoundUp(meshlet_group_face_prefix_sum.count, ParallelForThreadCount(parallel_for));
	u32 meshlet_work_item_count = DivideAndRoundUp(meshlet_group_face_prefix_sum.count, face_groups_per_thread);
	
	ParallelFor(parallel_for, meshlet_work_item_count, [&](u32 work_item_index) {
		MDT_PROFILER_SCOPE("BuildMeshletsForFaceGroup");
		
		Allocator thread_allocator;
		thread_allocator.callbacks = allocator.callbacks;
		
		Array<u8> vertex_usage_map;
		ArrayResizeMemset(vertex_usage_map, thread_allocator, mesh.attribute_count, 0xFF);
		
		u32 begin_group_index = work_item_index * face_groups_per_thread;
		u32 end_group_index   = Min(begin_group_index + face_groups_per_thread, meshlet_group_face_prefix_sum.count);
		
		u32 begin_element_index = begin_group_index ? meshlet_group_face_prefix_sum[begin_group_index - 1] : 0;
		for (u32 group_index = begin_group_index; group_index < end_group_index; group_index += 1) {
			u32 end_element_index = meshlet_group_face_prefix_sum[group_index];
			u32 element_count     = end_element_index - begin_element_index;
			
			KdTree kd_tree;
			kd_tree.elements        = { kd_tree_allocations.elements.data        + begin_element_index, element_count, element_count };
			kd_tree.element_indices = { kd_tree_allocations.element_indices.data + begin_element_index, element_count, element_count };
			kd_tree.nodes           = { kd_tree_allocations.nodes.data           + begin_element_index * 2, 0, element_count * 2 };
			
			KdTreeBuildElementsForFaces(mesh, kd_tree, begin_element_index, end_element_index);
			KdTreeBuildNode(kd_tree.nodes, CreateArrayView(kd_tree.elements), CreateArrayView(kd_tree.element_indices));
			
			Array<FaceID>    group_meshlet_faces             = { meshlet_faces.data             + begin_element_index,     0, element_count };
			Array<u8>        group_meshlet_triangles         = { meshlet_triangles.data         + begin_element_index * 3, 0, element_count * 3 };
			Array<CornerID>  group_meshlet_corners           = { meshlet_corners.data           + begin_element_index * 3, 0, element_count * 3 };
			Array<u32>       group_meshlet_face_prefix_sum   = { meshlet_face_prefix_sum.data   + begin_element_index,     0, element_count };
			Array<u32>       group_meshlet_corner_prefix_sum = { meshlet_corner_prefix_sum.data + begin_element_index,     0, element_count };
			
			BuildMeshletsForFaceGroup(
				mesh,
				kd_tree,
				begin_element_index,
				end_element_index,
				meshlet_target_face_count,
				meshlet_target_vertex_count,
				CreateArrayView(corner_list_around_vertex),
				CreateArrayView(corner_list_around_vertex_prefix_sum),
				CreateArrayView(vertex_usage_map),
				group_meshlet_faces,
				group_meshlet_triangles,
				group_meshlet_face_prefix_sum,
				group_meshlet_corners,
				group_meshlet_corner_prefix_sum
			);
			MDT_ASSERT(group_meshlet_faces.count == element_count);
			
			meshlet_group_meshlet_prefix_sum[group_index] = group_meshlet_face_prefix_sum.count;
			begin_element_index = end_element_index;
		}
		
		AllocatorFreeMemoryBlocks(thread_allocator, 0);
	});
	
	MDT_ASSERT(meshlet_faces.count == mesh.face_count);
	
	u32 meshlet_prefix_sum = 0;
	u32 face_prefix_sum    = 0;
	u32 corner_prefix_sum  = 0;
	for (u32 group_index = 0; group_index < meshlet_group_face_prefix_sum.count; group_index += 1) {
		u32 meshlet_count = meshlet_group_meshlet_prefix_sum[group_index];
		
		u32 begin_element_index = group_index ? meshlet_group_face_prefix_sum[group_index - 1] : 0;
		u32 end_element_index   = meshlet_group_face_prefix_sum[group_index];
		u32 element_count       = end_element_index - begin_element_index;
		
		auto group_meshlet_faces             = CreateArrayView(meshlet_faces,             begin_element_index,     begin_element_index     + element_count);
		auto group_meshlet_triangles         = CreateArrayView(meshlet_triangles,         begin_element_index * 3, begin_element_index * 3 + element_count * 3);
		auto group_meshlet_corners           = CreateArrayView(meshlet_corners,           begin_element_index * 3, begin_element_index * 3 + element_count * 3);
		auto group_meshlet_face_prefix_sum   = CreateArrayView(meshlet_face_prefix_sum,   begin_element_index,     begin_element_index     + meshlet_count);
		auto group_meshlet_corner_prefix_sum = CreateArrayView(meshlet_corner_prefix_sum, begin_element_index,     begin_element_index     + meshlet_count);
		
		u32 face_count = group_meshlet_face_prefix_sum[meshlet_count - 1];
		MDT_ASSERT(face_count == element_count);
		
		u32 corner_count = group_meshlet_corner_prefix_sum[meshlet_count - 1];
		
		for (u32 i = begin_element_index; i < end_element_index; i += 1) {
			// Translate meshlet index within the group to a global meshlet index.
			kd_tree_allocations.elements[i].partition_index += meshlet_prefix_sum;
		}
		
		for (u32 i = 0; i < meshlet_count; i += 1, meshlet_prefix_sum += 1) {
			meshlet_face_prefix_sum[meshlet_prefix_sum]   = group_meshlet_face_prefix_sum[i]   + face_prefix_sum;
			meshlet_corner_prefix_sum[meshlet_prefix_sum] = group_meshlet_corner_prefix_sum[i] + corner_prefix_sum;
		}
		meshlet_group_meshlet_prefix_sum[group_index] = meshlet_prefix_sum;
		
		memmove(meshlet_corners.data + corner_prefix_sum, group_meshlet_corners.data, corner_count * sizeof(CornerID));
		
		face_prefix_sum   += face_count;
		corner_prefix_sum += corner_count;
	}
	meshlet_face_prefix_sum.count   = meshlet_prefix_sum;
	meshlet_corner_prefix_sum.count = meshlet_prefix_sum;
	meshlet_corners.count           = corner_prefix_sum;
	MDT_ASSERT(meshlet_faces.count     == face_prefix_sum);
	MDT_ASSERT(meshlet_triangles.count == face_prefix_sum * 3);
	
	
	Array<MdtMeshlet> meshlets;
	ArrayResize(meshlets, allocator, meshlet_corner_prefix_sum.count);
	
	ParallelForBatched(parallel_for, meshlet_corner_prefix_sum.count, 64, [&](u32 meshlet_index) {
		u32 begin_corner_index = meshlet_index ? meshlet_corner_prefix_sum[meshlet_index - 1] : 0;
		u32 end_corner_index   = meshlet_corner_prefix_sum[meshlet_index];
		
		auto meshlet_aabb_min = Vector3{ +FLT_MAX, +FLT_MAX, +FLT_MAX };
		auto meshlet_aabb_max = Vector3{ -FLT_MAX, -FLT_MAX, -FLT_MAX };
		
		FixedSizeArray<MdtSphereBounds, meshlet_max_vertex_count> vertex_sphere_bounds;
		MDT_ASSERT(end_corner_index - begin_corner_index <= meshlet_max_vertex_count);
		
		for (u32 corner_index = begin_corner_index; corner_index < end_corner_index; corner_index += 1) {
			auto corner_id = meshlet_corners[corner_index];
			auto vertex_id = mesh.face_vertex_ids[corner_id.index];
			
			auto position = mesh[vertex_id];
			meshlet_aabb_min = VectorMin(meshlet_aabb_min, position);
			meshlet_aabb_max = VectorMax(meshlet_aabb_max, position);
			
			MdtSphereBounds bounds;
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
		meshlet.geometry_index = mesh.face_geometry_indices[meshlet_corners[begin_corner_index].index / 3];
	});
	
	for (u32 group_index = 0, meshlet_begin_index = 0; group_index < meshlet_group_error_metrics.count; group_index += 1) {
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
	
	result.meshlet_adjacency = BuildMeshletAdjacency(
		mesh,
		allocator,
		CreateArrayView(corner_list_around_vertex),
		CreateArrayView(corner_list_around_vertex_prefix_sum),
		result.meshlet_corners,
		result.meshlet_corner_prefix_sum,
		CreateArrayView(kd_tree_allocations.elements)
	);
	
	return result;
}

static MeshletAdjacency BuildMeshletAdjacency(
	const IndexedMeshView& mesh,
	Allocator& allocator,
	ArrayView<u32> corner_list_around_vertex,
	ArrayView<u32> corner_list_around_vertex_prefix_sum,
	ArrayView<CornerID> meshlet_corners,
	ArrayView<u32> meshlet_corner_prefix_sum,
	ArrayView<KdTreeElement> kd_tree_elements) {
	
	MDT_PROFILER_SCOPE("BuildMeshletAdjacency");
	
	FixedSizeArray<VertexID, meshlet_max_vertex_count> counted_vertex_ids;
	
	Array<u8> is_vertex_counted;
	ArrayResizeMemset(is_vertex_counted, allocator, mesh.vertex_count, 0);
	
	Array<u32> meshlet_adjacency_info_indices;
	ArrayResizeMemset(meshlet_adjacency_info_indices, allocator, meshlet_corner_prefix_sum.count, 0xFF);
	
	Array<u32> meshlet_adjacency_prefix_sum;
	ArrayReserve(meshlet_adjacency_prefix_sum, allocator, meshlet_corner_prefix_sum.count);
	
	Array<MeshletAdjacencyInfo> meshlet_adjacency_infos;
	ArrayReserve(meshlet_adjacency_infos, allocator, meshlet_corner_prefix_sum.count * 8);
	
	u32 begin_corner_index = 0;
	for (u32 meshlet_index = 0; meshlet_index < meshlet_corner_prefix_sum.count; meshlet_index += 1) {
		u32 end_corner_index = meshlet_corner_prefix_sum[meshlet_index];
		
		// At least reserve one meshlet per face edge. Do this upfront instead of adding code in the inner loop to improve performance.
		compile_const u32 reserve_size = meshlet_max_face_count * meshlet_max_face_degree;
		if (meshlet_adjacency_infos.count + reserve_size >= meshlet_adjacency_infos.capacity) {
			ArrayReserve(meshlet_adjacency_infos, allocator, ArrayComputeNewCapacity(meshlet_adjacency_infos.capacity, meshlet_adjacency_infos.capacity + reserve_size));
		}
		
		u32 begin_adjacency_info_index = meshlet_adjacency_infos.count;
		for (u32 corner_index = begin_corner_index; corner_index < end_corner_index; corner_index += 1) {
			auto corner_id = meshlet_corners[corner_index];
			auto vertex_id = mesh.face_vertex_ids[corner_id.index];
			
			// Count each vertex at most once per meshlet.
			if (is_vertex_counted[vertex_id.index] != 0) continue;
			
			is_vertex_counted[vertex_id.index] = 0xFF;
			ArrayAppend(counted_vertex_ids, vertex_id);
			
			u32 corner_list_begin_index = corner_list_around_vertex_prefix_sum[vertex_id.index];
			u32 corner_list_end_index   = corner_list_around_vertex_prefix_sum[vertex_id.index + 1];
			
			// At least approximately deduplicate meshlets within the corner list of the given vertex.
			FixedSizeArray<u32, 16> other_meshlet_indices;
			
			for (u32 corner_list_index = corner_list_begin_index; corner_list_index < corner_list_end_index; corner_list_index += 1) {
				u32 other_corner_id = corner_list_around_vertex[corner_list_index];
				auto other_face_id = FaceID{ other_corner_id / 3 };
				
				u32 other_meshlet_index = kd_tree_elements[other_face_id.index].partition_index;
				if (other_meshlet_index == meshlet_index) continue;
				
				bool meshlet_is_already_counted = false;
				for (u32 i = 0; i < other_meshlet_indices.count && meshlet_is_already_counted == false; i += 1) {
					meshlet_is_already_counted |= other_meshlet_indices[i] == other_meshlet_index;
				}
				if (meshlet_is_already_counted) continue;
				
				if (other_meshlet_indices.count < other_meshlet_indices.capacity) {
					ArrayAppend(other_meshlet_indices, other_meshlet_index);
				}
				
				MDT_ASSERT(kd_tree_elements[other_face_id.index].is_active_element == 0); // Face isn't a part of any meshlet.
				
				u32 adjacency_info_index = meshlet_adjacency_info_indices[other_meshlet_index];
				if (adjacency_info_index == u32_max) {
					adjacency_info_index = meshlet_adjacency_infos.count;
					
					// Enough memory should be reserved upfront. Reserving directly in this loop
					// slows down adjacency search by 40% even if we never need to grow the array.
					if (meshlet_adjacency_infos.count >= meshlet_adjacency_infos.capacity) continue;
					
					meshlet_adjacency_info_indices[other_meshlet_index] = meshlet_adjacency_infos.count;
					
					MeshletAdjacencyInfo info;
					info.meshlet_index     = other_meshlet_index;
					info.shared_edge_count = 1;
					ArrayAppend(meshlet_adjacency_infos, info);
				} else {
					meshlet_adjacency_infos[adjacency_info_index].shared_edge_count += 1;
				}
			}
		}
		u32 end_adjacency_info_index = meshlet_adjacency_infos.count;
		
		ArrayAppend(meshlet_adjacency_prefix_sum, end_adjacency_info_index);
		
		for (u32 adjacency_info_index = begin_adjacency_info_index; adjacency_info_index < end_adjacency_info_index; adjacency_info_index += 1) {
			meshlet_adjacency_info_indices[meshlet_adjacency_infos[adjacency_info_index].meshlet_index] = u32_max;
		}
		
		for (auto vertex_id : counted_vertex_ids) {
			is_vertex_counted[vertex_id.index] = 0;
		}
		counted_vertex_ids.count = 0;
		
		begin_corner_index = end_corner_index;
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

static MeshletGroupBuildResult BuildMeshletGroups(Allocator& allocator, ArrayView<MdtMeshlet> meshlets, MeshletAdjacency meshlet_adjacency) {
	MDT_PROFILER_SCOPE("BuildMeshletGroups");
	
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
	
	MDT_ASSERT(meshlet_group_meshlet_indices.count == meshlets.count);
	
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

static void BuildInitialFaceGroupsRecursive(Array<u32>& meshlet_group_face_prefix_sum, ArrayView<u32> face_morton_codes, u32 current_bit, u32 begin_index, u32 end_index) {
	MDT_PROFILER_SCOPE("BuildInitialFaceGroupsRecursive");
	
	// Larger groups generally give higher quality results, but since there are less
	// of them they don't get distrubuted across threads as well as smaller groups.
	compile_const u32 max_initial_group_size = meshlet_group_max_meshlet_count * meshlet_max_face_count;
	
	if (end_index - begin_index <= max_initial_group_size || current_bit == 0) {
		ArrayAppend(meshlet_group_face_prefix_sum, end_index);
	} else if ((face_morton_codes[begin_index] & current_bit) == (face_morton_codes[end_index - 1] & current_bit)) {
		BuildInitialFaceGroupsRecursive(meshlet_group_face_prefix_sum, face_morton_codes, current_bit >> 1, begin_index, end_index);
	} else {
		u32 l0 = begin_index;
		u32 l1 = end_index - 1;
		
		while (l0 <= l1) {
			u32 center_index = l0 + (l1 - l0) / 2;
			u32 center_bit   = (face_morton_codes[center_index] & current_bit);
			
			if (center_bit == 0) {
				l0 = center_index + 1;
			} else {
				l1 = center_index - 1;
			}
		}
		
		BuildInitialFaceGroupsRecursive(meshlet_group_face_prefix_sum, face_morton_codes, current_bit >> 1, begin_index, l0);
		BuildInitialFaceGroupsRecursive(meshlet_group_face_prefix_sum, face_morton_codes, current_bit >> 1, l0, end_index);
	}
}

static void BuildInitialFaceGroups(const IndexedMeshView& mesh, Array<u32>& meshlet_group_face_prefix_sum, Allocator& allocator) {
	MDT_PROFILER_SCOPE("BuildInitialFaceGroups");
	u32 allocator_high_water = allocator.memory_block_count;
	
	Array<u32> face_morton_codes;
	Array<u32> face_morton_codes_swap;
	Array<u32> face_indices;
	Array<u32> face_indices_swap;
	
	ArrayResize(face_indices, allocator, mesh.face_count);
	ArrayResize(face_indices_swap, allocator, mesh.face_count);
	ArrayResize(face_morton_codes, allocator, mesh.face_count);
	ArrayResize(face_morton_codes_swap, allocator, mesh.face_count);
	
	auto aabb_min = Vector3{ +FLT_MAX, +FLT_MAX, +FLT_MAX };
	auto aabb_max = Vector3{ -FLT_MAX, -FLT_MAX, -FLT_MAX };
	
	for (u32 i = 0; i < mesh.vertex_count; i += 1) {
		aabb_min = VectorMin(aabb_min, mesh.vertices[i]);
		aabb_max = VectorMax(aabb_max, mesh.vertices[i]);
	}
	
	auto aabb_extent = aabb_max - aabb_min;
	auto aabb_scale  = Vector3{ aabb_extent.x == 0.f ? 0.f : 1.f / aabb_extent.x, aabb_extent.y == 0.f ? 0.f : 1.f / aabb_extent.y, aabb_extent.z == 0.f ? 0.f : 1.f / aabb_extent.z };
	auto aabb_offset = Vector3{ -aabb_min.x * aabb_scale.x, -aabb_min.y * aabb_scale.y, -aabb_min.z * aabb_scale.z };
	
	u32  deposit_masks[3] = { 0, 0, 0 };
	auto axis_float_to_u32 = Vector3{ 1.f, 1.f, 1.f };
	
	compile_const u32 morton_code_bit_count = 30;
	for (u32 i = 0; i < morton_code_bit_count; i += 1) {
		u32   axis_index  = 0;
		float axis_length = LoadElementByIndex(aabb_extent, 0);
		
		for (u32 axis = 1; axis < 3; axis += 1) {
			if (axis_length < LoadElementByIndex(aabb_extent, axis)) {
				axis_length = LoadElementByIndex(aabb_extent, axis);
				axis_index  = axis;
			}
		}
		
		// Split along the longest axis.
		u32 bit_index = morton_code_bit_count - 1 - i;
		deposit_masks[axis_index] |= (1u << bit_index);
		
		StoreElementByIndex(aabb_extent, axis_index, axis_length * 0.5f);
		StoreElementByIndex(axis_float_to_u32, axis_index, LoadElementByIndex(axis_float_to_u32, axis_index) * 2.f);
	}
	
	for (u32 i = 0; i < mesh.face_count; i += 1) {
		auto& p0 = mesh[mesh.face_vertex_ids[i * 3 + 0]];
		auto& p1 = mesh[mesh.face_vertex_ids[i * 3 + 1]];
		auto& p2 = mesh[mesh.face_vertex_ids[i * 3 + 2]];
		
		auto normalized_position = (p0 + p1 + p2) * (1.f / 3.f) * aabb_scale + aabb_offset;
		u32 x = (u32)(normalized_position.x * axis_float_to_u32.x);
		u32 y = (u32)(normalized_position.y * axis_float_to_u32.y);
		u32 z = (u32)(normalized_position.z * axis_float_to_u32.z);
		
		u32 encoded = 0;
		encoded |= _pdep_u32(x, deposit_masks[0]);
		encoded |= _pdep_u32(y, deposit_masks[1]);
		encoded |= _pdep_u32(z, deposit_masks[2]);
		
		face_morton_codes[i] = encoded;
		face_indices[i] = i;
	}
	
	for (u32 offset = 0; offset < 32; offset += 8) {
		MDT_PROFILER_SCOPE("RadixSortPass");
		
		u32 prefix_sum[256] = {};
		
		for (u32 vertex_index = 0; vertex_index < face_indices.count; vertex_index += 1) {
			u32 bin = (face_morton_codes[vertex_index] >> offset) & 0xFF;
			prefix_sum[bin] += 1;
		}
		
		u32 prefix_count = 0;
		for (u32 bin = 0; bin < 256; bin += 1) {
			u32 count = prefix_sum[bin];
			prefix_sum[bin] = prefix_count;
			prefix_count += count;
		}
		
		for (u32 vertex_index = 0; vertex_index < face_indices.count; vertex_index += 1) {
			u32 bin = (face_morton_codes[vertex_index] >> offset) & 0xFF;
			u32 new_index = prefix_sum[bin]++;
			
			face_indices_swap[new_index] = face_indices[vertex_index];
			face_morton_codes_swap[new_index] = face_morton_codes[vertex_index];
		}
		
		u32* face_indices_swap_data = face_indices_swap.data;
		u32* face_morton_codes_swap_data = face_morton_codes_swap.data;
		face_indices_swap.data = face_indices.data;
		face_morton_codes_swap.data = face_morton_codes.data;
		face_indices.data = face_indices_swap_data;
		face_morton_codes.data = face_morton_codes_swap_data;
	}
	
	BuildInitialFaceGroupsRecursive(meshlet_group_face_prefix_sum, CreateArrayView(face_morton_codes), 1u << (morton_code_bit_count - 1), 0, face_morton_codes.count);
	
	{
		MDT_PROFILER_SCOPE("ReorderFaces");
		
		Array<VertexID> face_vertex_ids;
		Array<AttributesID> face_attribute_ids;
		Array<u32> face_geometry_indices;
		ArrayResize(face_vertex_ids, allocator, face_indices.count * 3);
		ArrayResize(face_attribute_ids, allocator, face_indices.count * 3);
		ArrayResize(face_geometry_indices, allocator, face_indices.count);
		
		for (u32 output_face_index = 0; output_face_index < face_indices.count; output_face_index += 1) {
			u32 src_face_index = face_indices[output_face_index];
			
			memcpy(&face_vertex_ids[output_face_index * 3], &mesh.face_vertex_ids[src_face_index * 3], 3 * sizeof(VertexID));
			memcpy(&face_attribute_ids[output_face_index * 3], &mesh.face_attribute_ids[src_face_index * 3], 3 * sizeof(AttributesID));
			face_geometry_indices[output_face_index] = mesh.face_geometry_indices[src_face_index];
		}
		
		memcpy(mesh.face_vertex_ids, face_vertex_ids.data, face_vertex_ids.count * sizeof(VertexID));
		memcpy(mesh.face_attribute_ids, face_attribute_ids.data, face_attribute_ids.count * sizeof(AttributesID));
		memcpy(mesh.face_geometry_indices, face_geometry_indices.data, face_geometry_indices.count * sizeof(u32));
	}
	
	AllocatorFreeMemoryBlocks(allocator, allocator_high_water);
}

static void ConvertMeshletGroupsToFaceGroups(
	const IndexedMeshView& mesh,
	MeshletBuildResult meshlet_build_result,
	MeshletGroupBuildResult meshlet_group_build_result,
	Allocator& allocator,
	Array<u32>& meshlet_group_face_prefix_sum,
	Array<MdtErrorMetric>& meshlet_group_error_metrics) {
	
	MDT_PROFILER_SCOPE("ConvertMeshletGroupsToFaceGroups");
	
	MDT_ASSERT(meshlet_group_face_prefix_sum.capacity >= meshlet_build_result.meshlet_faces.count);
	MDT_ASSERT(meshlet_group_error_metrics.capacity   >= meshlet_build_result.meshlet_faces.count);
	
	meshlet_group_face_prefix_sum.count = 0;
	meshlet_group_error_metrics.count   = 0;
	
	u32 allocator_high_water = allocator.memory_block_count;
	
	{
		MDT_PROFILER_SCOPE("ReorderFaces");
		
		Array<VertexID> face_vertex_ids;
		Array<AttributesID> face_attribute_ids;
		Array<u32> face_geometry_indices;
		ArrayResize(face_vertex_ids, allocator, meshlet_build_result.meshlet_faces.count * 3);
		ArrayResize(face_attribute_ids, allocator, meshlet_build_result.meshlet_faces.count * 3);
		ArrayResize(face_geometry_indices, allocator, meshlet_build_result.meshlet_faces.count);
		
		u32 group_meshlet_begin_index = 0;
		for (u32 group_index = 0, output_face_index = 0; group_index < meshlet_group_build_result.prefix_sum.count; group_index += 1) {
			u32 group_meshlet_end_index = meshlet_group_build_result.prefix_sum[group_index];
			
			for (u32 group_meshlet_index = group_meshlet_begin_index; group_meshlet_index < group_meshlet_end_index; group_meshlet_index += 1) {
				u32 meshlet_index = meshlet_group_build_result.meshlet_indices[group_meshlet_index];
				
				u32 begin_face_index = meshlet_index > 0 ? meshlet_build_result.meshlet_face_prefix_sum[meshlet_index - 1] : 0;
				u32 end_face_index   = meshlet_build_result.meshlet_face_prefix_sum[meshlet_index];
				for (u32 face_index = begin_face_index; face_index < end_face_index; face_index += 1, output_face_index += 1) {
					auto src_face_id = meshlet_build_result.meshlet_faces[face_index];
					meshlet_build_result.meshlet_faces[face_index].index = output_face_index;
					
					memcpy(&face_vertex_ids[output_face_index * 3], &mesh.face_vertex_ids[src_face_id.index * 3], 3 * sizeof(VertexID));
					memcpy(&face_attribute_ids[output_face_index * 3], &mesh.face_attribute_ids[src_face_id.index * 3], 3 * sizeof(AttributesID));
					face_geometry_indices[output_face_index] = mesh.face_geometry_indices[src_face_id.index];
				}
			}
			
			group_meshlet_begin_index = group_meshlet_end_index;
		}
		
		memcpy(mesh.face_vertex_ids, face_vertex_ids.data, face_vertex_ids.count * sizeof(VertexID));
		memcpy(mesh.face_attribute_ids, face_attribute_ids.data, face_attribute_ids.count * sizeof(AttributesID));
		memcpy(mesh.face_geometry_indices, face_geometry_indices.data, face_geometry_indices.count * sizeof(u32));
	}
	
	AllocatorFreeMemoryBlocks(allocator, allocator_high_water);
	
	
	u32 group_meshlet_begin_index = 0;
	for (u32 group_index = 0, meshlet_group_face_count = 0; group_index < meshlet_group_build_result.prefix_sum.count; group_index += 1) {
		u32 group_meshlet_end_index = meshlet_group_build_result.prefix_sum[group_index];
		
		FixedSizeArray<MdtSphereBounds, meshlet_group_max_meshlet_count> meshlet_error_sphere_bounds;
		float max_error = 0.f;
		
		for (u32 group_meshlet_index = group_meshlet_begin_index; group_meshlet_index < group_meshlet_end_index; group_meshlet_index += 1) {
			u32 meshlet_index = meshlet_group_build_result.meshlet_indices[group_meshlet_index];
			
			u32 begin_face_index = meshlet_index > 0 ? meshlet_build_result.meshlet_face_prefix_sum[meshlet_index - 1] : 0;
			u32 end_face_index   = meshlet_build_result.meshlet_face_prefix_sum[meshlet_index];
			
			u32 meshlet_face_count = end_face_index - begin_face_index;
			meshlet_group_face_count += meshlet_face_count;
			
			auto& meshlet = meshlet_build_result.meshlets[meshlet_index];
			ArrayAppend(meshlet_error_sphere_bounds, meshlet.current_level_error_metric.bounds);
			max_error = meshlet.current_level_error_metric.error > max_error ? meshlet.current_level_error_metric.error : max_error;
		}
		ArrayAppend(meshlet_group_face_prefix_sum, meshlet_group_face_count);
		
		//
		// Coarser level meshlets should have at least the same error as their children (finer representation).
		// This error metric might get increased during meshlet group decimation.
		//
		// See [Karis 2021] for reference on monotonic error metric.
		//
		MdtErrorMetric meshlet_group_minimum_error_metric;
		meshlet_group_minimum_error_metric.bounds = ComputeSphereBoundsUnion(CreateArrayView(meshlet_error_sphere_bounds));
		meshlet_group_minimum_error_metric.error  = max_error;
		ArrayAppend(meshlet_group_error_metrics, meshlet_group_minimum_error_metric);
		
		group_meshlet_begin_index = group_meshlet_end_index;
	}
}

static void AppendNewMeshletsAndMeshletGroups(Allocator& heap_allocator, MeshletBuildResult meshlet_build_result, MeshletGroupBuildResult meshlet_group_build_result, Array<MdtErrorMetric> meshlet_group_error_metrics, Array<MdtMeshletGroup>& meshlet_groups, Array<MdtMeshlet>& meshlets, u32 level_index) {
	MDT_PROFILER_SCOPE("AppendNewMeshletsAndMeshletGroups");
	
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
		
		FixedSizeArray<MdtSphereBounds, meshlet_group_max_meshlet_count> meshlet_sphere_bounds;
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
		
		MdtMeshletGroup meshlet_group;
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


static ArrayView<FaceID> CreateMeshFaceRemap(IndexedMeshView& mesh, Allocator& allocator) {
	MDT_PROFILER_SCOPE("CreateMeshFaceRemap");
	
	Array<FaceID> old_face_id_to_new_face_id;
	ArrayResize(old_face_id_to_new_face_id, allocator, mesh.face_count);
	
	u32 old_face_count = old_face_id_to_new_face_id.count;
	u32 new_face_count = 0;
	
	for (FaceID old_face_id = { 0 }; old_face_id.index < old_face_count; old_face_id.index += 1) {
		auto vertex_id = mesh.face_vertex_ids[old_face_id.index * 3];
		
		FaceID new_face_id = { u32_max };
		if (vertex_id.index != u32_max) {
			new_face_id = { new_face_count };
			new_face_count += 1;
			
			memcpy(&mesh.face_vertex_ids[new_face_id.index * 3], &mesh.face_vertex_ids[old_face_id.index * 3], 3 * sizeof(VertexID));
			memcpy(&mesh.face_attribute_ids[new_face_id.index * 3], &mesh.face_attribute_ids[old_face_id.index * 3], 3 * sizeof(AttributesID));
			mesh.face_geometry_indices[new_face_id.index] = mesh.face_geometry_indices[old_face_id.index];
		}
		old_face_id_to_new_face_id[old_face_id.index] = new_face_id;
	}
	mesh.face_count = new_face_count;
	
	return CreateArrayView(old_face_id_to_new_face_id);
}

static void CompactMeshletGroupFaces(ArrayView<FaceID> old_face_id_to_new_face_id, Array<u32>& meshlet_group_face_prefix_sum) {
	MDT_PROFILER_SCOPE("CompactMeshletGroupFaces");
	
	u32 new_prefix_sum = 0;
	
	u32 begin_face_index = 0;
	for (u32 group_index = 0; group_index < meshlet_group_face_prefix_sum.count; group_index += 1) {
		u32 end_face_index = meshlet_group_face_prefix_sum[group_index];
		
		for (u32 face_index = begin_face_index; face_index < end_face_index; face_index += 1) {
			auto new_face_id = old_face_id_to_new_face_id[face_index];
			new_prefix_sum += new_face_id.index != u32_max ? 1 : 0;
		}
		meshlet_group_face_prefix_sum[group_index] = new_prefix_sum;
		
		begin_face_index = end_face_index;
	}
}

// TODO: We could write this data directly into the output buffers from BuildMeshletsForFaceGroups.
static void BuildMeshletVertexAndIndexBuffers(
	const IndexedMeshView& mesh,
	Allocator& heap_allocator,
	MeshletBuildResult meshlet_build_result,
	Array<u32> attributes_id_to_vertex_index,
	Array<u32>& meshlet_vertex_indices,
	Array<MdtMeshletTriangle>& meshlet_triangles) {
	MDT_PROFILER_SCOPE("BuildMeshletVertexAndIndexBuffers");
	
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
			auto attributes_id = mesh.face_attribute_ids[corner_id.index];
			
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
			MdtMeshletTriangle triangle;
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
	const IndexedMeshView& mesh,
	Allocator& allocator,
	Allocator& heap_allocator,
	Array<u8> changed_vertex_mask,
	Array<u32> attributes_id_to_vertex_index,
	Array<float>& vertices) {
	MDT_PROFILER_SCOPE("AppendChangedVertices");
	
	Array<VertexID> attributes_id_to_vertex_id;
	ArrayResizeMemset(attributes_id_to_vertex_id, allocator, mesh.attribute_count, 0xFF);

	for (u32 corner_index = 0; corner_index < mesh.face_count * 3; corner_index += 1) {
		auto vertex_id = mesh.face_vertex_ids[corner_index];
		if (vertex_id.index == u32_max || changed_vertex_mask[vertex_id.index] == 0) continue;

		auto attributes_id = mesh.face_attribute_ids[corner_index];
		attributes_id_to_vertex_id[attributes_id.index] = vertex_id;
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
		memcpy(vertex + 0, &mesh[vertex_id], sizeof(Vector3));
		memcpy(vertex + 3, mesh[attributes_id], attribute_stride_dwords * sizeof(u32));
		output_vertex_index += 1;
	}

	MDT_ASSERT(vertices.count == output_vertex_index * vertex_stride_dwords);
}

} // namespace MeshDecimationTools


void MdtBuildContinuousLod(const MdtContinuousLodBuildInputs* inputs, MdtContinuousLodBuildResult* result, const MdtSystemCallbacks* callbacks) {
	using namespace MeshDecimationTools;
	MDT_PROFILER_SCOPE("MdtBuildContinuousLod");
	
	MDT_ASSERT(inputs);
	MDT_ASSERT(result);
	
	Allocator allocator;
	InitializeAllocator(allocator, callbacks ? &callbacks->temp_allocator : nullptr);

	Allocator heap_allocator;
	InitializeAllocator(heap_allocator, callbacks ? &callbacks->heap_allocator : nullptr);
	
	auto mesh = BuildIndexedMesh(allocator, inputs->mesh.geometry_descs, inputs->mesh.geometry_desc_count, inputs->mesh.vertex_stride_bytes);
	
	u32 meshlet_target_face_count   = Clamp(inputs->meshlet_target_triangle_count, 1u, meshlet_max_face_count);
	u32 meshlet_target_vertex_count = Clamp(inputs->meshlet_target_vertex_count, 3u, meshlet_max_vertex_count);
	
	Array<u32> meshlet_group_face_prefix_sum;
	Array<MdtErrorMetric> meshlet_group_error_metrics;
	ArrayReserve(meshlet_group_face_prefix_sum, allocator, mesh.face_count);
	ArrayReserve(meshlet_group_error_metrics, allocator, mesh.face_count);
	
	// Note that we're not adding an initial error metric to meshlet_group_error_metrics.
	// Meshlets built from the initial groups will use current_level_error_metric.bounds set to the geometric_sphere_bounds with zero error.
	bool split_into_initial_groups_for_threading = ParallelForThreadCount(callbacks ? &callbacks->parallel_for : nullptr) >= 2;
	if (split_into_initial_groups_for_threading) {
		// Build initial groups so we can parallelize the initial meshlet generation across them.
		BuildInitialFaceGroups(mesh, meshlet_group_face_prefix_sum, allocator);
	} else {
		ArrayAppend(meshlet_group_face_prefix_sum, mesh.face_count);
	}
	
	u32 meshlet_group_base_index = 0;
	u32 last_level_meshlet_count = 0;
	
	
	Array<u32> attributes_id_to_vertex_index;
	ArrayResizeMemset(attributes_id_to_vertex_index, allocator, mesh.attribute_count, 0xFF);
	
	Array<u8> changed_vertex_mask;
	ArrayResizeMemset(changed_vertex_mask, allocator, mesh.attribute_count, 0xFF);
	
	Array<float> vertices;
	AppendChangedVertices(mesh, allocator, heap_allocator, changed_vertex_mask, attributes_id_to_vertex_index, vertices);
	
	Array<MdtContinuousLodLevel> levels;
	ArrayReserve(levels, heap_allocator, continuous_lod_max_levels_of_details);
	
	Array<MdtMeshletGroup> meshlet_groups;
	Array<MdtMeshlet>      meshlets;
	
	Array<u32> meshlet_vertex_indices;
	Array<MdtMeshletTriangle> meshlet_triangles;
	
	for (u32 level_index = 0; level_index < continuous_lod_max_levels_of_details; level_index += 1) {
		u32 allocator_high_water = allocator.memory_block_count;
		
		auto meshlet_build_result = BuildMeshletsForFaceGroups(
			mesh,
			callbacks ? &callbacks->parallel_for : nullptr,
			allocator,
			meshlet_group_face_prefix_sum,
			meshlet_group_error_metrics,
			meshlet_target_face_count,
			meshlet_target_vertex_count,
			meshlet_group_base_index
		);
		
		BuildMeshletVertexAndIndexBuffers(
			mesh,
			heap_allocator,
			meshlet_build_result,
			attributes_id_to_vertex_index,
			meshlet_vertex_indices,
			meshlet_triangles
		);
		
		
		auto meshlet_group_build_result = BuildMeshletGroups(
			allocator,
			meshlet_build_result.meshlets,
			meshlet_build_result.meshlet_adjacency
		);
		
		ConvertMeshletGroupsToFaceGroups(
			mesh,
			meshlet_build_result,
			meshlet_group_build_result,
			allocator,
			meshlet_group_face_prefix_sum,
			meshlet_group_error_metrics
		);
		
		
		bool is_last_level = (level_index + 1 == continuous_lod_max_levels_of_details) || (mesh.face_count <= meshlet_target_face_count) || (last_level_meshlet_count == meshlet_build_result.meshlets.count);
		last_level_meshlet_count = meshlet_build_result.meshlets.count;
		
		if (is_last_level == false) {
			DecimateMeshFaceGroups(
				mesh,
				allocator,
				heap_allocator,
				inputs->mesh,
				callbacks ? &callbacks->parallel_for : nullptr,
				meshlet_group_face_prefix_sum,
				meshlet_group_error_metrics,
				changed_vertex_mask
			);
			
			AppendChangedVertices(mesh, allocator, heap_allocator, changed_vertex_mask, attributes_id_to_vertex_index, vertices);
			
			// Compact the mesh after decimation to remove unused faces and edges.
			auto remap = CreateMeshFaceRemap(mesh, allocator);
			CompactMeshletGroupFaces(remap, meshlet_group_face_prefix_sum);
		} else {
			// There is no coarser version of the mesh. Set meshlet group errors to FLT_MAX to make sure LOD
			// culling test always succeeds for last level meshlets (i.e. coarser level is always too coarse).
			for (auto& error_metric : meshlet_group_error_metrics) error_metric.error = FLT_MAX;
		}
		
		MdtContinuousLodLevel level;
		level.begin_meshlet_groups_index = meshlet_groups.count;
		level.begin_meshlets_index       = meshlets.count;
		
		meshlet_group_base_index = meshlet_groups.count;
		AppendNewMeshletsAndMeshletGroups(
			heap_allocator,
			meshlet_build_result,
			meshlet_group_build_result,
			meshlet_group_error_metrics,
			meshlet_groups,
			meshlets,
			level_index
		);
		
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
	MDT_ASSERT(heap_allocator.memory_block_count == 6);
	
	AllocatorFreeMemoryBlocks(allocator);
}

void MdtFreeContinuousLodBuildResult(const MdtContinuousLodBuildResult* result, const MdtSystemCallbacks* callbacks) {
	using namespace MeshDecimationTools;
	
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


void MdtBuildDiscreteLod(const MdtDiscreteLodBuildInputs* /*inputs*/, MdtDiscreteLodBuildResult* /*result*/, const MdtSystemCallbacks* /*callbacks*/) {
	using namespace MeshDecimationTools;
	
#if 0
	MDT_ASSERT(inputs);
	MDT_ASSERT(result);
	
	Allocator allocator;
	InitializeAllocator(allocator, callbacks ? &callbacks->temp_allocator : nullptr);

	Allocator heap_allocator;
	InitializeAllocator(heap_allocator, callbacks ? &callbacks->heap_allocator : nullptr);
	
	auto mesh = BuildEditableMesh(allocator, inputs->mesh.geometry_descs, inputs->mesh.geometry_desc_count, inputs->mesh.vertex_stride_bytes);
	
	Array<MdtDecimatedGeometryDesc> geometry_descs;
	ArrayResize(geometry_descs, heap_allocator, inputs->mesh.geometry_desc_count * inputs->level_of_detail_count);
	
	Array<MdtDiscreteLodLevel> level_of_detail_result_descs;
	ArrayResize(level_of_detail_result_descs, heap_allocator, inputs->level_of_detail_count);
	
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
			InitializeMeshDecimationState(mesh, inputs->mesh, allocator, heap_allocator, state);
			
			
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
		
		auto& level_of_detail_result = level_of_detail_result_descs[level_index];
		level_of_detail_result.max_error = max_error;
		level_of_detail_result.begin_geometry_index = level_index * inputs->mesh.geometry_desc_count;
		level_of_detail_result.end_geometry_index   = level_of_detail_result.begin_geometry_index + inputs->mesh.geometry_desc_count;
		
		auto level_geometry_descs = CreateArrayView(geometry_descs, level_of_detail_result.begin_geometry_index, level_of_detail_result.end_geometry_index);
		
		u32 geometry_index = u32_max;
		for (FaceID face_id = { 0 }; face_id.index < mesh.face_count; face_id.index += 1) {
			auto& face = mesh[face_id];
			
			if (geometry_index != face.geometry_index) {
				MDT_ASSERT(geometry_index == u32_max || geometry_index < face.geometry_index);
				
				if (geometry_index != u32_max) {
					auto& geometry_desc = level_geometry_descs[geometry_index];
					geometry_desc.end_indices_index = indices.count;
				}
				geometry_index = face.geometry_index;
				
				auto& geometry_desc = level_geometry_descs[geometry_index];
				geometry_desc.begin_indices_index   = indices.count;
				geometry_desc.end_indices_index     = indices.count;
			}
			
			IterateCornerList<FaceID>(mesh, face.corner_list_base, [&](CornerID corner_id) {
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

	MDT_ASSERT(heap_allocator.memory_block_count == 4);
	
	u32 vertex_stride_dwords = (mesh.attribute_stride_dwords + 3);
	
	result->level_of_detail_descs = level_of_detail_result_descs.data;
	result->geometry_descs        = geometry_descs.data;
	result->indices               = indices.data;
	result->vertices              = vertices.data;
	result->level_of_detail_count = level_of_detail_result_descs.count;
	result->geometry_desc_count   = geometry_descs.count;
	result->index_count           = indices.count;
	result->vertex_count          = vertices.count / vertex_stride_dwords;
	
	AllocatorFreeMemoryBlocks(allocator);
#endif
}

void MdtFreeDiscreteLodBuildResult(const MdtDiscreteLodBuildResult* result, const MdtSystemCallbacks* callbacks) {
	using namespace MeshDecimationTools;
	
	Allocator heap_allocator;
	InitializeAllocator(heap_allocator, callbacks ? &callbacks->heap_allocator : nullptr);
	
	heap_allocator.memory_blocks[heap_allocator.memory_block_count++] = result->level_of_detail_descs;
	heap_allocator.memory_blocks[heap_allocator.memory_block_count++] = result->geometry_descs;
	heap_allocator.memory_blocks[heap_allocator.memory_block_count++] = result->indices;
	heap_allocator.memory_blocks[heap_allocator.memory_block_count++] = result->vertices;
	AllocatorFreeMemoryBlocks(heap_allocator);
}


//
// Based on [Kapoulkine 2025].
//
MdtSphereBounds MdtComputeSphereBoundsUnion(const MdtSphereBounds* source_sphere_bounds, uint32_t source_sphere_bounds_count) {
	using namespace MeshDecimationTools;
	
	u32 aabb_min_index[3] = { 0, 0, 0 };
	u32 aabb_max_index[3] = { 0, 0, 0 };
	
	float aabb_min[3] = { +FLT_MAX, +FLT_MAX, +FLT_MAX };
	float aabb_max[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
	
	for (u32 i = 0; i < source_sphere_bounds_count; i += 1) {
		auto source_bounds = source_sphere_bounds[i];
		
		auto min = source_bounds.center - source_bounds.radius;
		auto max = source_bounds.center + source_bounds.radius;
		
		for (u32 axis = 0; axis < 3; axis += 1) {
			if (aabb_min[axis] > LoadElementByIndex(min, axis)) {
				aabb_min[axis] = LoadElementByIndex(min, axis);
				aabb_min_index[axis] = i;
			}
			
			if (aabb_max[axis] < LoadElementByIndex(max, axis)) {
				aabb_max[axis] = LoadElementByIndex(max, axis);
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
	
	MdtSphereBounds result_bounds;
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

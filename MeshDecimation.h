#ifndef MESHDECIMATION_H
#define MESHDECIMATION_H

#include <assert.h>

#define compile_const constexpr static const

using u8 = unsigned char;
using s8 = signed char;
static_assert(sizeof(u8) == 1);
static_assert(sizeof(s8) == 1);
compile_const u8 u8_max = (u8)0xFF;
compile_const u8 u8_min = (u8)0x00;
compile_const s8 s8_max = (s8)0x7F;
compile_const s8 s8_min = (s8)0x80;


using u16 = unsigned short;
using s16 = signed short;
static_assert(sizeof(u16) == 2);
static_assert(sizeof(s16) == 2);
compile_const u16 u16_max = (u16)0xFFFF;
compile_const u16 u16_min = (u16)0x0000;
compile_const s16 s16_max = (s16)0x7FFF;
compile_const s16 s16_min = (s16)0x8000;


using u32 = unsigned int;
using s32 = signed int;
static_assert(sizeof(u32) == 4);
static_assert(sizeof(s32) == 4);
compile_const u32 u32_max = (u32)0xFFFF'FFFF;
compile_const u32 u32_min = (u32)0x0000'0000;
compile_const s32 s32_max = (s32)0x7FFF'FFFF;
compile_const s32 s32_min = (s32)0x8000'0000;


using u64 = unsigned long long;
using s64 = signed long long;
static_assert(sizeof(u64) == 8);
static_assert(sizeof(s64) == 8);
compile_const u64 u64_max = (u64)0xFFFF'FFFF'FFFF'FFFF;
compile_const u64 u64_min = (u64)0x0000'0000'0000'0000;
compile_const s64 s64_max = (s64)0x7FFF'FFFF'FFFF'FFFF;
compile_const s64 s64_min = (s64)0x8000'0000'0000'0000;

compile_const u32 u8_bit_count = 8;
compile_const u32 s8_bit_count = 8;
compile_const u32 u16_bit_count = 16;
compile_const u32 s16_bit_count = 16;
compile_const u32 u32_bit_count = 32;
compile_const u32 s32_bit_count = 32;
compile_const u32 u64_bit_count = 64;
compile_const u32 s64_bit_count = 64;
compile_const u32 f32_bit_count = 32;
compile_const u32 f64_bit_count = 64;


struct Vector3 {
	float x = 0.f;
	float y = 0.f;
	float z = 0.f;
	
	friend bool operator== (const Vector3& lh, const Vector3& rh) { return lh.x == rh.x && lh.y == rh.y && lh.z == rh.z; }
	float& operator[] (u32 index) { return (&x)[index]; }
	float operator[] (u32 index) const { return (&x)[index]; }
};

struct SphereBounds {
	Vector3 center = { 0.f, 0.f, 0.f };
	float   radius = 0.f;
};

struct ErrorMetric {
	SphereBounds bounds = { 0.f, 0.f, 0.f };
	float        error  = 0.f;
};

template<typename T>
struct ArrayView {
	using ValueType = T;
	
	T* data = nullptr;
	u32 count = 0;
	
	T& operator[] (u32 index) { assert(index < count); return data[index]; }
	const T& operator[] (u32 index) const { assert(index < count); return data[index]; }

	T* begin() { return data; }
	T* end() { return data + count; }
	const T* begin() const { return data; }
	const T* end() const { return data + count; }
};

using NormalizeVertexAttributes = void(*)(float*);

struct TriangleGeometryDesc {
	const u32* indices = nullptr;
	u32 index_count = 0;
	
	const float* vertices = nullptr;
	u32 vertex_count = 0;
};

struct TriangleMeshDesc {
	const TriangleGeometryDesc* geometry_descs = nullptr;
	u32 geometry_desc_count = 0;
	u32 vertex_stride_bytes = 0;
	
	float* attribute_weights = nullptr;
	NormalizeVertexAttributes normalize_vertex_attributes;
};

struct VirtualGeometryBuildInputs {
	TriangleMeshDesc mesh;
	
	// TODO: Custom meshlet size.
	// u32 meshlet_max_vertex_count   = 128;
	// u32 meshlet_max_triangle_count = 128;
	
	// TODO: Custom BVH format.
	// u32 mehslet_group_bvh_max_branching_factor = 4;
};

struct MeshDecimationInputs {
	TriangleMeshDesc mesh;
	
	u32 target_face_count = 0;
	// TODO: Add support for error limit.
	// float error_limit = 0.f;
	
	// TODO: Generate multiple levels of detail at once.
	// u32 level_of_detail_count = 0;
};

struct alignas(16) Meshlet {
	// Bounding box over meshlet vertex positions.
	alignas(16) Vector3 aabb_min;
	alignas(16) Vector3 aabb_max;
	
	// Bounding sphere over meshlet vertex positions.
	SphereBounds geometric_sphere_bounds;
	
	// Error metric of this meshlet. Transferred from the group this meshlet was built from.
	ErrorMetric current_level_error_metric;
	// current_level_error_metric is extracted from this BVH node.
	u32 current_level_bvh_node_index = 0;
	
	// Error metric of one level coarser representation of this meshlet. Transferred from the group that was built using this meshlet.
	ErrorMetric coarser_level_error_metric;
	// coarser_level_error_metric is extracted from this BVH node.
	u32 coarser_level_bvh_node_index = 0;
	
	u32 begin_vertex_indices_index = 0;
	u32 end_vertex_indices_index   = 0;
	
	u32 begin_meshlet_triangles_index = 0;
	u32 end_meshlet_triangles_index   = 0;
	
	u32 geometry_index = 0;
};

struct MeshletGroupInternalBvhNodeData {
	u32 child_indices[4];
};

struct MeshletGroupLeafBvhNodeData {
	u32 begin_child_index;
	u32 end_child_index;
};

using ReallocCallback = void* (*)(void* old_memory_block, u64 size_bytes, void* user_data);

struct AllocatorCallbacks {
	ReallocCallback realloc = nullptr;
	void* user_data = nullptr;
};

struct SystemCallbacks {
	// Optional memory allocation callbacks. If they're not provided the system falls back to realloc().
	AllocatorCallbacks temp_allocator; // Memory blocks are allocated and freed in stack order.
	AllocatorCallbacks heap_allocator; // Memory blocks are allocated and freed in any order.
};

struct alignas(16) MeshletGroupBvhNode {
	// Bounding box over child bounding boxes.
	alignas(16) Vector3 aabb_min;
	alignas(16) Vector3 aabb_max;
	
	// Bounding sphere over child bounding spheres.
	SphereBounds geometric_sphere_bounds;
	
	// Internal nodes store union of child node error metrics.
	// Leaf nodes store coarser_level_error_metric from child meshlets (it is the same by construction).
	ErrorMetric error_metric;
	
	union {
		// Internal nodes store indices of child meshlet group BVH nodes.
		MeshletGroupInternalBvhNodeData internal;
		
		// Leaf nodes store a range of meshlet indices.
		MeshletGroupLeafBvhNodeData leaf;
	};
	
	bool is_leaf_node = false;
};

struct VirtualGeometryLevel {
	u32 begin_bvh_nodes_index = 0;
	u32 end_bvh_nodes_index   = 0;
	
	u32 begin_meshlets_index = 0;
	u32 end_meshlets_index   = 0;
	
	u32 meshlet_group_bvh_root_node_index = 0;
};

struct VirtualGeometryBuildResult {
	ArrayView<MeshletGroupBvhNode>  bvh_nodes;
	ArrayView<Meshlet>              meshlets;
	ArrayView<u32>                  meshlet_vertex_indices;
	ArrayView<u8>                   meshlet_triangles;
	ArrayView<float>                vertices;
	ArrayView<VirtualGeometryLevel> levels;
	
	u32 meshlet_group_bvh_root_node_index = 0;
};

struct MeshDecimationResult {
	// TODO: Output per geometry index and vertex ranges.
	ArrayView<u32>   indices;
	ArrayView<float> vertices;
	float max_error = 0.f;
};

void BuildVirtualGeometry(const VirtualGeometryBuildInputs& inputs, VirtualGeometryBuildResult& result, const SystemCallbacks& callbacks);
void DecimateMesh(const MeshDecimationInputs& inputs, MeshDecimationResult& result, const SystemCallbacks& callbacks);

void FreeResultBuffers(const VirtualGeometryBuildResult& result, const SystemCallbacks& callbacks);
void FreeResultBuffers(const MeshDecimationResult& result, const SystemCallbacks& callbacks);

#endif // MESHDECIMATION_H

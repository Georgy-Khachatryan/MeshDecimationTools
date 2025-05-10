#ifndef MESHDECIMATION_H
#define MESHDECIMATION_H

#include <assert.h>
#include <stdint.h>

struct Vector3 {
	float x = 0.f;
	float y = 0.f;
	float z = 0.f;
	
	friend bool operator== (const Vector3& lh, const Vector3& rh) { return lh.x == rh.x && lh.y == rh.y && lh.z == rh.z; }
	float& operator[] (uint32_t index) { return (&x)[index]; }
	float operator[] (uint32_t index) const { return (&x)[index]; }
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
	uint32_t count = 0;
	
	T& operator[] (uint32_t index) { assert(index < count); return data[index]; }
	const T& operator[] (uint32_t index) const { assert(index < count); return data[index]; }

	T* begin() { return data; }
	T* end() { return data + count; }
	const T* begin() const { return data; }
	const T* end() const { return data + count; }
};


using ReallocCallback = void* (*)(void* old_memory_block, uint64_t size_bytes, void* user_data);

struct AllocatorCallbacks {
	ReallocCallback realloc = nullptr;
	void* user_data = nullptr;
};

struct SystemCallbacks {
	// Optional memory allocation callbacks. If they're not provided the system falls back to realloc().
	AllocatorCallbacks temp_allocator; // Memory blocks are allocated and freed in stack order.
	AllocatorCallbacks heap_allocator; // Memory blocks are allocated and freed in any order.
};


using NormalizeVertexAttributes = void(*)(float*);

struct TriangleGeometryDesc {
	const uint32_t* indices = nullptr;
	uint32_t index_count = 0;
	
	const float* vertices = nullptr;
	uint32_t vertex_count = 0;
};

struct TriangleMeshDesc {
	const TriangleGeometryDesc* geometry_descs = nullptr;
	uint32_t geometry_desc_count = 0;
	uint32_t vertex_stride_bytes = 0;
	
	float* attribute_weights = nullptr;
	NormalizeVertexAttributes normalize_vertex_attributes;
};

struct VirtualGeometryBuildInputs {
	TriangleMeshDesc mesh;
	
	// TODO: Custom meshlet size.
	// uint32_t meshlet_max_vertex_count   = 128;
	// uint32_t meshlet_max_triangle_count = 128;
};

struct MeshDecimationInputs {
	TriangleMeshDesc mesh;
	
	uint32_t target_face_count = 0;
	// TODO: Add support for error limit.
	// float error_limit = 0.f;
	
	// TODO: Generate multiple levels of detail at once.
	// uint32_t level_of_detail_count = 0;
};

struct alignas(16) Meshlet {
	// Bounding box over meshlet vertex positions.
	alignas(16) Vector3 aabb_min;
	alignas(16) Vector3 aabb_max;
	
	// Bounding sphere over meshlet vertex positions.
	SphereBounds geometric_sphere_bounds;
	
	// Error metric of this meshlet. Transferred from the group this meshlet was built from.
	ErrorMetric current_level_error_metric;
	// current_level_error_metric is extracted from this meshlet group.
	uint32_t current_level_meshlet_group_index = 0;
	
	// Error metric of one level coarser representation of this meshlet. Transferred from the group that was built using this meshlet.
	ErrorMetric coarser_level_error_metric;
	// coarser_level_error_metric is extracted from this meshlet group.
	uint32_t coarser_level_meshlet_group_index = 0;
	
	uint32_t begin_vertex_indices_index = 0;
	uint32_t end_vertex_indices_index   = 0;
	
	uint32_t begin_meshlet_triangles_index = 0;
	uint32_t end_meshlet_triangles_index   = 0;
	
	uint32_t geometry_index = 0;
};

struct alignas(16) MeshletGroup {
	// Bounding box over child bounding boxes.
	alignas(16) Vector3 aabb_min;
	alignas(16) Vector3 aabb_max;
	
	// Bounding sphere over child bounding spheres.
	SphereBounds geometric_sphere_bounds;
	
	// Internal nodes store union of child node error metrics.
	// Leaf nodes store coarser_level_error_metric from child meshlets (it is the same by construction).
	ErrorMetric error_metric;
	
	uint32_t begin_meshlet_index = 0;
	uint32_t end_meshlet_index   = 0;
	
	uint32_t level_of_detail_index = 0;
};

struct VirtualGeometryLevel {
	uint32_t begin_meshlet_groups_index = 0;
	uint32_t end_meshlet_groups_index   = 0;
	
	uint32_t begin_meshlets_index = 0;
	uint32_t end_meshlets_index   = 0;
};

struct VirtualGeometryBuildResult {
	ArrayView<MeshletGroup>         meshlet_groups;
	ArrayView<Meshlet>              meshlets;
	ArrayView<uint32_t>             meshlet_vertex_indices;
	ArrayView<uint8_t>              meshlet_triangles;
	ArrayView<float>                vertices;
	ArrayView<VirtualGeometryLevel> levels;
};

struct MeshDecimationResult {
	// TODO: Output per geometry index and vertex ranges.
	ArrayView<uint32_t> indices;
	ArrayView<float>    vertices;
	float max_error = 0.f;
};

void BuildVirtualGeometry(const VirtualGeometryBuildInputs& inputs, VirtualGeometryBuildResult& result, const SystemCallbacks& callbacks);
void DecimateMesh(const MeshDecimationInputs& inputs, MeshDecimationResult& result, const SystemCallbacks& callbacks);

void FreeResultBuffers(const VirtualGeometryBuildResult& result, const SystemCallbacks& callbacks);
void FreeResultBuffers(const MeshDecimationResult& result, const SystemCallbacks& callbacks);

SphereBounds ComputeSphereBoundsUnion(ArrayView<SphereBounds> source_sphere_bounds);

#endif // MESHDECIMATION_H

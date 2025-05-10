#pragma once

#ifndef MESHDECIMATION_H
#define MESHDECIMATION_H

#include <assert.h>
#include <stdint.h>

struct Vector3 {
	float x;
	float y;
	float z;

#if defined(__cplusplus)	
	friend bool operator== (const Vector3& lh, const Vector3& rh) { return lh.x == rh.x && lh.y == rh.y && lh.z == rh.z; }
	float& operator[] (uint32_t index) { return (&x)[index]; }
	float operator[] (uint32_t index) const { return (&x)[index]; }
#endif // defined(__cplusplus)	
};

struct SphereBounds {
	struct Vector3 center;
	float          radius;
};

struct ErrorMetric {
	struct SphereBounds bounds;
	float               error;
};


typedef void* (*ReallocCallback)(void* old_memory_block, uint64_t size_bytes, void* user_data);

struct AllocatorCallbacks {
	ReallocCallback realloc;
	void* user_data;
};

struct SystemCallbacks {
	// Optional memory allocation callbacks. If they're not provided the system falls back to realloc().
	struct AllocatorCallbacks temp_allocator; // Memory blocks are allocated and freed in stack order.
	struct AllocatorCallbacks heap_allocator; // Memory blocks are allocated and freed in any order.
};


typedef void(*NormalizeVertexAttributes)(float* attributes);

struct TriangleGeometryDesc {
	const uint32_t* indices;
	uint32_t index_count;
	
	const float* vertices;
	uint32_t vertex_count;
};

struct TriangleMeshDesc {
	const struct TriangleGeometryDesc* geometry_descs;
	uint32_t geometry_desc_count;
	uint32_t vertex_stride_bytes;
	
	float* attribute_weights;
	NormalizeVertexAttributes normalize_vertex_attributes;
};

struct VirtualGeometryBuildInputs {
	struct TriangleMeshDesc mesh;
	
	// TODO: Custom meshlet size.
	// uint32_t meshlet_max_vertex_count   = 128;
	// uint32_t meshlet_max_triangle_count = 128;
};

struct MeshDecimationInputs {
	struct TriangleMeshDesc mesh;
	
	uint32_t target_face_count;
	// TODO: Add support for error limit.
	// float error_limit = 0.f;
	
	// TODO: Generate multiple levels of detail at once.
	// uint32_t level_of_detail_count = 0;
};

struct Meshlet {
	// Bounding box over meshlet vertex positions.
	struct Vector3 aabb_min;
	struct Vector3 aabb_max;
	
	// Bounding sphere over meshlet vertex positions.
	struct SphereBounds geometric_sphere_bounds;
	
	// Error metric of this meshlet. Transferred from the group this meshlet was built from.
	struct ErrorMetric current_level_error_metric;
	// current_level_error_metric is extracted from this meshlet group.
	uint32_t current_level_meshlet_group_index;
	
	// Error metric of one level coarser representation of this meshlet. Transferred from the group that was built using this meshlet.
	struct ErrorMetric coarser_level_error_metric;
	// coarser_level_error_metric is extracted from this meshlet group.
	uint32_t coarser_level_meshlet_group_index;
	
	uint32_t begin_vertex_indices_index;
	uint32_t end_vertex_indices_index;
	
	uint32_t begin_meshlet_triangles_index;
	uint32_t end_meshlet_triangles_index;
	
	uint32_t geometry_index;
};

struct MeshletGroup {
	// Bounding box over child bounding boxes.
	struct Vector3 aabb_min;
	struct Vector3 aabb_max;
	
	// Bounding sphere over child bounding spheres.
	struct SphereBounds geometric_sphere_bounds;
	
	// Internal nodes store union of child node error metrics.
	// Leaf nodes store coarser_level_error_metric from child meshlets (it is the same by construction).
	struct ErrorMetric error_metric;
	
	uint32_t begin_meshlet_index;
	uint32_t end_meshlet_index;
	
	uint32_t level_of_detail_index;
};

struct VirtualGeometryLevel {
	uint32_t begin_meshlet_groups_index;
	uint32_t end_meshlet_groups_index;
	
	uint32_t begin_meshlets_index;
	uint32_t end_meshlets_index;
};

struct VirtualGeometryBuildResult {
	struct MeshletGroup* meshlet_groups;
	struct Meshlet* meshlets;
	uint32_t* meshlet_vertex_indices;
	uint8_t* meshlet_triangles;
	float* vertices;
	struct VirtualGeometryLevel* levels;
	
	uint32_t meshlet_group_count;
	uint32_t meshlet_count;
	uint32_t meshlet_vertex_index_count;
	uint32_t meshlet_triangle_count;
	uint32_t vertex_count;
	uint32_t level_count;
};

struct MeshDecimationResult {
	// TODO: Output per geometry index and vertex ranges.
	uint32_t* indices;
	float* vertices;
	
	uint32_t index_count;
	uint32_t vertex_count;
	
	float max_error;
};

void BuildVirtualGeometry(const struct VirtualGeometryBuildInputs* inputs, struct VirtualGeometryBuildResult* result, const struct SystemCallbacks* callbacks);
void DecimateMesh(const struct MeshDecimationInputs* inputs, struct MeshDecimationResult* result, const struct SystemCallbacks* callbacks);

void FreeVirtualGeometryBuildResult(const struct VirtualGeometryBuildResult* result, const struct SystemCallbacks* callbacks);
void FreeMeshDecimationResult(const struct MeshDecimationResult* result, const struct SystemCallbacks* callbacks);

struct SphereBounds ComputeSphereBoundsUnion(const struct SphereBounds* source_sphere_bounds, uint32_t source_sphere_bounds_count);

#endif // MESHDECIMATION_H

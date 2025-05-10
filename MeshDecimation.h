#pragma once

#ifndef MESHDECIMATION_H
#define MESHDECIMATION_H

#include <assert.h>
#include <stdint.h>

struct VgtVector3 {
	float x;
	float y;
	float z;

#if defined(__cplusplus)	
	friend bool operator== (const VgtVector3& lh, const VgtVector3& rh) { return lh.x == rh.x && lh.y == rh.y && lh.z == rh.z; }
	float& operator[] (uint32_t index) { return (&x)[index]; }
	float operator[] (uint32_t index) const { return (&x)[index]; }
#endif // defined(__cplusplus)	
};

struct VgtSphereBounds {
	struct VgtVector3 center;
	float             radius;
};

struct VgtErrorMetric {
	struct VgtSphereBounds bounds;
	float                  error;
};


typedef void* (*VgtReallocCallback)(void* old_memory_block, uint64_t size_bytes, void* user_data);

struct VgtAllocatorCallbacks {
	VgtReallocCallback realloc;
	void*            user_data;
};

struct VgtSystemCallbacks {
	// Optional memory allocation callbacks. If they're not provided the system falls back to realloc().
	struct VgtAllocatorCallbacks temp_allocator; // Memory blocks are allocated and freed in stack order.
	struct VgtAllocatorCallbacks heap_allocator; // Memory blocks are allocated and freed in any order.
};


typedef void (*VgtNormalizeVertexAttributes)(float* attributes);

struct VgtTriangleGeometryDesc {
	const uint32_t* indices;
	uint32_t index_count;
	
	const float* vertices;
	uint32_t vertex_count;
};

struct VgtTriangleMeshDesc {
	const struct VgtTriangleGeometryDesc* geometry_descs;
	uint32_t geometry_desc_count;
	uint32_t vertex_stride_bytes;
	
	float* attribute_weights;
	VgtNormalizeVertexAttributes normalize_vertex_attributes;
};

struct VgtVirtualGeometryBuildInputs {
	struct VgtTriangleMeshDesc mesh;
	
	// TODO: Custom meshlet size.
	// uint32_t meshlet_max_vertex_count   = 128;
	// uint32_t meshlet_max_triangle_count = 128;
};

struct VgtMeshDecimationInputs {
	struct VgtTriangleMeshDesc mesh;
	
	uint32_t target_face_count;
	// TODO: Add support for error limit.
	// float error_limit = 0.f;
	
	// TODO: Generate multiple levels of detail at once.
	// uint32_t level_of_detail_count = 0;
};

struct VgtMeshletTriangle {
	uint32_t i0 : 10;
	uint32_t i1 : 10;
	uint32_t i2 : 10;
};

struct VgtMeshlet {
	// Bounding box over meshlet vertex positions.
	struct VgtVector3 aabb_min;
	struct VgtVector3 aabb_max;
	
	// Bounding sphere over meshlet vertex positions.
	struct VgtSphereBounds geometric_sphere_bounds;
	
	// Error metric of this meshlet. Transferred from the group this meshlet was built from.
	struct VgtErrorMetric current_level_error_metric;
	// current_level_error_metric is extracted from this meshlet group.
	uint32_t current_level_meshlet_group_index;
	
	// Error metric of one level coarser representation of this meshlet. Transferred from the group that was built using this meshlet.
	struct VgtErrorMetric coarser_level_error_metric;
	// coarser_level_error_metric is extracted from this meshlet group.
	uint32_t coarser_level_meshlet_group_index;
	
	uint32_t begin_vertex_indices_index;
	uint32_t end_vertex_indices_index;
	
	uint32_t begin_meshlet_triangles_index;
	uint32_t end_meshlet_triangles_index;
	
	uint32_t geometry_index;
};

struct VgtMeshletGroup {
	// Bounding box over child bounding boxes.
	struct VgtVector3 aabb_min;
	struct VgtVector3 aabb_max;
	
	// Bounding sphere over child bounding spheres.
	struct VgtSphereBounds geometric_sphere_bounds;
	
	// Internal nodes store union of child node error metrics.
	// Leaf nodes store coarser_level_error_metric from child meshlets (it is the same by construction).
	struct VgtErrorMetric error_metric;
	
	uint32_t begin_meshlet_index;
	uint32_t end_meshlet_index;
	
	uint32_t level_of_detail_index;
};

struct VgtVirtualGeometryLevel {
	uint32_t begin_meshlet_groups_index;
	uint32_t end_meshlet_groups_index;
	
	uint32_t begin_meshlets_index;
	uint32_t end_meshlets_index;
};

struct VgtVirtualGeometryBuildResult {
	struct VgtMeshletGroup* meshlet_groups;
	struct VgtMeshlet* meshlets;
	uint32_t* meshlet_vertex_indices;
	struct VgtMeshletTriangle* meshlet_triangles;
	float* vertices;
	struct VgtVirtualGeometryLevel* levels;
	
	uint32_t meshlet_group_count;
	uint32_t meshlet_count;
	uint32_t meshlet_vertex_index_count;
	uint32_t meshlet_triangle_count;
	uint32_t vertex_count;
	uint32_t level_count;
};

struct VgtMeshDecimationResult {
	// TODO: Output per geometry index and vertex ranges.
	uint32_t* indices;
	float* vertices;
	
	uint32_t index_count;
	uint32_t vertex_count;
	
	float max_error;
};

void VgtBuildVirtualGeometry(const struct VgtVirtualGeometryBuildInputs* inputs, struct VgtVirtualGeometryBuildResult* result, const struct VgtSystemCallbacks* callbacks);
void VgtDecimateMesh(const struct VgtMeshDecimationInputs* inputs, struct VgtMeshDecimationResult* result, const struct VgtSystemCallbacks* callbacks);

void VgtFreeVirtualGeometryBuildResult(const struct VgtVirtualGeometryBuildResult* result, const struct VgtSystemCallbacks* callbacks);
void VgtFreeMeshDecimationResult(const struct VgtMeshDecimationResult* result, const struct VgtSystemCallbacks* callbacks);

struct VgtSphereBounds VgtComputeSphereBoundsUnion(const struct VgtSphereBounds* source_sphere_bounds, uint32_t source_sphere_bounds_count);

#endif // MESHDECIMATION_H

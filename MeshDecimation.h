#pragma once

#ifndef MESHDECIMATION_H
#define MESHDECIMATION_H

#include <stdint.h>


#if defined(VGT_CONFIGURATION_FILE)
#include VGT_CONFIGURATION_FILE
#endif // defined(VGT_CONFIGURATION_FILE)

#if !defined(VGT_ASSERT)
#include <assert.h>
#define VGT_ASSERT(condition) assert(condition)
#endif // !defined(VGT_ASSERT) 

#if !defined(VGT_MAX_ATTRIBUTE_STRIDE_DWORDS)
#define VGT_MAX_ATTRIBUTE_STRIDE_DWORDS 16
#endif // !defined(VGT_MAX_ATTRIBUTE_STRIDE_DWORDS)

#if !defined(VGT_MESHLET_GROUP_SIZE)
#define VGT_MESHLET_GROUP_SIZE 32
#endif // !defined(VGT_MESHLET_GROUP_SIZE)

#if !defined(VGT_ENABLE_ATTRIBUTE_SUPPORT)
#define VGT_ENABLE_ATTRIBUTE_SUPPORT 1
#endif // !defined(VGT_ENABLE_ATTRIBUTE_SUPPORT)


#if defined(__cplusplus)
extern "C" {
#endif // defined(__cplusplus)

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


//
// Memory reallocation function similar to C realloc():
// - Allocate a new memory block if (old_memory_block == NULL && size_bytes != 0).
// - Free the old memory block if (old_memory_block != NULL && size_bytes == 0).
// - Extend the old memory block or allocate a new memory block and memcpy the old contents to it if (old_memory_block != NULL && size_bytes != 0).
//
typedef void* (*VgtReallocCallback)(void* old_memory_block, uint64_t size_bytes, void* user_data);

struct VgtAllocatorCallbacks {
	// Reallocation callback, see definition of VgtReallocCallback for reference.
	VgtReallocCallback realloc;
	
	// User defined allocator state, passed to VgtReallocCallback as user_data argument.
	void* user_data;
};

// Optional memory allocation callbacks. If they're not provided the system falls back to C realloc().
struct VgtSystemCallbacks {
	//
	// Temporary allocator that is used as a stack. Falls back to C realloc() if not provided.
	// Memory blocks are allocated and freed from the end.
	//
	struct VgtAllocatorCallbacks temp_allocator;
	
	//
	// Heap allocator used for small number of growable arrays and all output allocations. Falls back to C realloc() if not provided.
	// Memory blocks are allocated and freed in arbitrary order.
	//
	struct VgtAllocatorCallbacks heap_allocator;
};


// See normalize_vertex_attributes in VgtTriangleMeshDesc for reference.
typedef void (*VgtNormalizeVertexAttributes)(float* attributes);

struct VgtTriangleGeometryDesc {
	// Array of vertex indices. 3 consecutive indices form a triangle.
	const uint32_t* indices;
	
	// Array of vertices. Stride is equal to 'vertex_stride_bytes' and provided via VgtTriangleMeshDesc.
	const float* vertices;
	
	// Size of the 'indices' array. Must be a multiple of 3.
	uint32_t index_count;
	
	// Size of the 'vertices' array, in vertices of size 'vertex_stride_bytes' (NOT in floats).
	uint32_t vertex_count;
};

struct VgtTriangleMeshDesc {
	//
	// Array of geometry descriptions. One mesh may contain multiple geometries that get simplified
	// without forming cracks in between. This is useful when different geometries use different
	// materials or rendered using different techniques. For example a window might have transparent
	// glass as one geometry and opaque wood frame as another geometry. This also matches the way
	// raytracing acceleration structure build APIs work.
	//
	const struct VgtTriangleGeometryDesc* geometry_descs;
	
	// Size of the 'geometry_descs' array.
	uint32_t geometry_desc_count;
	
	// Byte offset between individual vertices in geometries.
	uint32_t vertex_stride_bytes;
	
	//
	// Relative error weights of vertex attributes. Size must be equal to VGT_MAX_ATTRIBUTE_STRIDE_DWORDS
	// or set to NULL for default weights==1.0.
	// Vertex positions are internally scaled such that the average face area is equal to one square unit.
	// Default weights==1.0 work well in most cases (unit vectors, quaternions, UV coordinates, colors, etc).
	// 
	float* attribute_weights;
	
	//
	// Vertex attribute normalization callback. Note that only attributes are passed in. Can be set to NULL.
	// This callback is called when a new set of attributes is computed and can be used to normalize unit
	// vectors and quaternions, or clamp coordinates or colors.
	//
	VgtNormalizeVertexAttributes normalize_vertex_attributes;
};

struct VgtVirtualGeometryBuildInputs {
	// Source triangle mesh containing multiple geometries. See VgtTriangleMeshDesc definition for reference.
	struct VgtTriangleMeshDesc mesh;
	
	// Target triangle count for meshlet builder. It will try to get as close to this value,
	// but never go above it. Internally clamped between 1 and 128 triangles.
	uint32_t meshlet_target_triangle_count;
	
	// Target vertex count for meshlet builder. It will try to get as close to this value,
	// but never go above it. Internally clamped between 3 and 254 vertices.
	uint32_t meshlet_target_vertex_count;
};

struct VgtLevelOfDetailTargetDesc {
	// Target face count for decimated mesh. Decimation algorithm will terminate once face count is
	// below or equal to the target face count.
	uint32_t target_face_count;
	
	// Target error limit for decimated mesh. Decimation algorithm will terminate before the limit is exceeded.
	float target_error_limit;
};

struct VgtMeshDecimationInputs {
	// Source triangle mesh containing multiple geometries. See VgtTriangleMeshDesc definition for reference.
	struct VgtTriangleMeshDesc mesh;
	
	// Array of target limits for each level of detail.
	VgtLevelOfDetailTargetDesc* level_of_detail_descs;
	
	// Size of 'level_of_detail_descs' array.
	uint32_t level_of_detail_count;
};

struct VgtMeshletTriangle {
	// Three indices into meshlet_vertex_indices array. See VgtMeshlet and VgtVirtualGeometryBuildResult definitions for reference.
	uint8_t i0;
	uint8_t i1;
	uint8_t i2;
};

struct VgtMeshlet {
	// Bounding box over meshlet vertex positions.
	struct VgtVector3 aabb_min;
	struct VgtVector3 aabb_max;
	
	// Bounding sphere over meshlet vertex positions.
	struct VgtSphereBounds geometric_sphere_bounds;
	
	//
	// Error metric of this meshlet. Transferred from the group this meshlet was built from.
	// For the first level meshlets the error is set to 0.
	// Meshlet should be drawn if (EvaluateErrorMetric(meshlet.current_level_error_metric) <= target_error).
	//
	struct VgtErrorMetric current_level_error_metric;
	//
	// Index of a meshlet group from which this meshlet was built. Set to UINT32_MAX for the first level meshlets.
	// current_level_error_metric is extracted from this meshlet group.
	//
	uint32_t current_level_meshlet_group_index;
	
	
	//
	// Error metric of one level coarser representation of this meshlet. Transferred from the group that was built using this meshlet.
	// For the last level meshlet groups the error is set to FLT_MAX.
	// Meshlet should be drawn if (EvaluateErrorMetric(meshlet.coarser_level_error_metric) > target_error).
	//
	struct VgtErrorMetric coarser_level_error_metric;
	//
	// Index of a meshlet group that was built from and contains this meshlet. This index is always valid, even for the last level meshlets.
	// coarser_level_error_metric is extracted from this meshlet group.
	//
	uint32_t coarser_level_meshlet_group_index;
	
	//
	// Range of meshlet_vertex_indices for this meshlet.
	// To iterate over all meshlet vertices use this loop:
	// for (u32 i = meshlet.begin_vertex_indices_index; i < meshlet.end_vertex_indices_index; i += 1) {
	//     u32 vertex_index = result.meshlet_vertex_indices[i];
	//     float* vertex = &result.vertices[vertex_index * vertex_stride_dwords];
	// }
	//
	uint32_t begin_vertex_indices_index;
	uint32_t end_vertex_indices_index;
	
	//
	// Range of meshlet_triangles for this meshlet.
	// To iterate over all meshlet triangles use this loop:
	// for (u32 i = meshlet.begin_meshlet_triangles_index; i < meshlet.end_meshlet_triangles_index; i += 1) {
	//     VgtMeshletTriangle triangle = result.meshlet_triangles[i];
	//     u32 vertex_index0 = result.meshlet_vertex_indices[triangle.i0 + meshlet.begin_vertex_indices_index];
	//     u32 vertex_index1 = result.meshlet_vertex_indices[triangle.i1 + meshlet.begin_vertex_indices_index];
	//     u32 vertex_index2 = result.meshlet_vertex_indices[triangle.i2 + meshlet.begin_vertex_indices_index];
	//     float* vertex0 = &result.vertices[vertex_index0 * vertex_stride_dwords];
	//     float* vertex1 = &result.vertices[vertex_index1 * vertex_stride_dwords];
	//     float* vertex2 = &result.vertices[vertex_index2 * vertex_stride_dwords];
	// }
	//
	uint32_t begin_meshlet_triangles_index;
	uint32_t end_meshlet_triangles_index;
	
	//
	// Index of the source geometry. Each meshlet is built from faces and vertices of only one geometry.
	// Note that meshlet groups may contain meshlets from different geometries. See definition of
	// VgtTriangleMeshDesc for more information.
	//
	uint32_t geometry_index;
};

struct VgtMeshletGroup {
	// Bounding box over source meshlet bounding boxes.
	struct VgtVector3 aabb_min;
	struct VgtVector3 aabb_max;
	
	// Bounding sphere over source meshlet bounding spheres.
	struct VgtSphereBounds geometric_sphere_bounds;
	
	//
	// Union of source meshlet current_level_error_metrics and decimation error for this meshlet group.
	// Source meshlet current_level_error_metric is the same as this error_metric. For the last level
	// meshlet groups the error is set to FLT_MAX.
	//
	// This error_metric can be used to accelerate LOD tests by first checking that it's error is
	// larger than the target error, and only then checking source meshlets if their error is
	// smaller than the target error:
	// if (EvaluateErrorMetric(error_metric) > target_error) {
	//     for (u32 i = group.begin_meshlet_index; i < group.end_meshlet_index; i += 1) {
	//         if (EvaluateErrorMetric(meshlets[i].current_level_error_metric) <= target_error) {
	//             DrawMeshlet(i);
	//         }
	//     }
	// }
	//
	struct VgtErrorMetric error_metric;
	
	// Range of source meshlets for this meshlet group. Groups can contain up to 32 meshlets.
	uint32_t begin_meshlet_index;
	uint32_t end_meshlet_index;
	
	//
	// LOD of source meshlets in range [0, 16).
	// Level 0 is the highest quality (i.e. source geometry), level 15 is the lowest quality.
	//
	uint32_t level_of_detail_index;
};

struct VgtVirtualGeometryLevel {
	// Range of meshlet groups for this level of detail.
	uint32_t begin_meshlet_groups_index;
	uint32_t end_meshlet_groups_index;
	
	// Range of meshlets for this level of detail.
	uint32_t begin_meshlets_index;
	uint32_t end_meshlets_index;
};

struct VgtVirtualGeometryBuildResult {
	//
	// Array of meshlet groups.
	// Meshlet groups are the unit of mesh decimation.
	// See definition of VgtMeshletGroup for more details.
	//
	struct VgtMeshletGroup* meshlet_groups;
	
	//
	// Array of meshlets.
	// Meshlets are spatially and topologically small regions of a mesh. They have a limited
	// number of faces and vertices. See definition of VgtMeshlet for more details.
	//
	struct VgtMeshlet* meshlets;
	
	//
	// Flattened arrays of vertex indices for each meshlet. Use [begin_vertex_indices_index, end_vertex_indices_index)
	// range to extract vertex indices of a given meshlet. See definition of VgtMeshlet for more details.
	//
	uint32_t* meshlet_vertex_indices;
	
	//
	// Flattened arrays of triangles for each meshlet. Each triangle contains indices into meshlet_vertex_indices array.
	// Use [begin_meshlet_triangles_index, end_meshlet_triangles_index) range to extract triangles of a given meshlet.
	// See definition of VgtMeshlet for more details.
	//
	struct VgtMeshletTriangle* meshlet_triangles;
	
	//
	// Array of vertices shared by all meshlets across all levels. Indexed using elements from
	// meshlet_vertex_indices array. Vertex stride matches the stride passed in VgtTriangleMeshDesc.
	//
	float* vertices;
	
	// Meshlet and meshlet group ranges for each level of detail.
	struct VgtVirtualGeometryLevel* levels;
	
	
	//
	// Sizes for each corresponding array defined above.
	// Vertex count is in vertices of size 'vertex_stride_bytes' (NOT in floats).
	//
	uint32_t meshlet_group_count;
	uint32_t meshlet_count;
	uint32_t meshlet_vertex_index_count;
	uint32_t meshlet_triangle_count;
	uint32_t vertex_count;
	uint32_t level_count;
};

struct VgtLevelOfDetailResultDesc {
	// Maximum edge collapse error encountered during simplification.
	float max_error;
	
	// Range of decimated geometry descs for this level of detail.
	uint32_t begin_geometry_index;
	uint32_t end_geometry_index;
};

struct VgtDecimatedGeometryDesc {
	// Range of vertex indices corresponding to a single geometry.
	uint32_t begin_indices_index;
	uint32_t end_indices_index;
};

struct VgtMeshDecimationResult {
	// Array of level of detail descriptions.
	struct VgtLevelOfDetailResultDesc* level_of_detail_descs;
	
	//
	// Array of per geometry ranges of vertex indices across all levels of detail.
	// Use 'level_of_detail_descs' to iterate over geometries of a specific level of detail.
	//
	struct VgtDecimatedGeometryDesc* geometry_descs;
	
	//
	// Array of vertex indices for all geometries across all levels of detail.
	// Use 'geometry_descs' to iterate over index ranges for a specific geometries.
	//
	uint32_t* indices;
	
	//
	// Array of vertices for all geometries across all levels of detail. Vertices that are not changed between levels of detail are not duplicated.
	// Vertex stride matches the stride passed in VgtTriangleMeshDesc.
	//
	float* vertices;
	
	//
	// Sizes for each corresponding array defined above.
	// Vertex count is in vertices of size 'vertex_stride_bytes' (NOT in floats).
	//
	uint32_t level_of_detail_count;
	uint32_t geometry_desc_count;
	uint32_t index_count;
	uint32_t vertex_count;
};

void VgtBuildVirtualGeometry(const struct VgtVirtualGeometryBuildInputs* inputs, struct VgtVirtualGeometryBuildResult* result, const struct VgtSystemCallbacks* callbacks);
void VgtDecimateMesh(const struct VgtMeshDecimationInputs* inputs, struct VgtMeshDecimationResult* result, const struct VgtSystemCallbacks* callbacks);

void VgtFreeVirtualGeometryBuildResult(const struct VgtVirtualGeometryBuildResult* result, const struct VgtSystemCallbacks* callbacks);
void VgtFreeMeshDecimationResult(const struct VgtMeshDecimationResult* result, const struct VgtSystemCallbacks* callbacks);

struct VgtSphereBounds VgtComputeSphereBoundsUnion(const struct VgtSphereBounds* source_sphere_bounds, uint32_t source_sphere_bounds_count);

#if defined(__cplusplus)
} // extern "C"
#endif // defined(__cplusplus)

#endif // MESHDECIMATION_H

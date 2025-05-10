#include "MeshDecimation.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unordered_map>
#include <vector>

#pragma warning(disable: 4996)

static bool IsLineEnding(char c) {
	return (c == '\n') || (c == '\r');
}

static bool IsWhiteSpace(char c) {
	return (c == ' ') || (c == '\t') || IsLineEnding(c);
}

static const char* EatWhiteSpace(const char* string) {
	while (*string && IsWhiteSpace(*string)) string += 1;
	return string;
}

static const char* EatSingleLineComment(const char* string) {
	while (*string && IsLineEnding(*string) == false) string += 1;
	while (IsLineEnding(*string)) string += 1;
	return string;
}


static const char* EatWhiteSpaceAndComments(const char* string) {
	string = EatWhiteSpace(string);
	while (string[0] == '#') {
		string = EatSingleLineComment(string + 1);
		string = EatWhiteSpace(string);
	}
	return string;
}


#define compile_const constexpr static const

using u32 = uint32_t;
using u64 = uint64_t;

struct Vector2 {
	float x = 0.f;
	float y = 0.f;
};

struct ObjVertex {
	Vector3 position;
	Vector2 texcoord;
	Vector3 normal;
};
compile_const u32 obj_vertex_stride_dwords = sizeof(ObjVertex) / sizeof(u32);

static void NormalizeObjVertexAttributes(float* attributes) {
	auto& normal = *(Vector3*)(attributes + 2);
	
	float length_square = normal.x * normal.x + normal.y * normal.y + normal.z * normal.z;
	
	if (length_square > 0.f) {
		float scale = 1.f / sqrtf(length_square);
		
		normal.x = normal.x * scale;
		normal.y = normal.y * scale;
		normal.z = normal.z * scale;
	}
}

struct ObjTriangleMesh {
	std::vector<u32> indices;
	std::vector<ObjVertex> vertices;
};

static ObjTriangleMesh ParseWavefrontObj(const char* string) {
	std::vector<Vector3> positions;
	std::vector<Vector3> normals;
	std::vector<Vector2> texcoords;
	
	ObjTriangleMesh mesh;
	
	std::unordered_map<u64, u32> index_map;

	while (*string) {
		string = EatWhiteSpaceAndComments(string);
		
		switch (*string) {
		case 'm': // "mtllib", not interested.
		case 'u': // "usemtl", not interested.
		case 'o': // "o" "ObjectName", not interested.
		case 's': // "s" "%d" smoothing group, not interested.
		{
			string = EatSingleLineComment(string + 1);
			break;
		} case 'v': {
			string += 1;
			
			if (*string == 't') { // "vt"
				string += 1;
				Vector2 texcoord;
				texcoord.x = strtof(string, (char**)&string);
				texcoord.y = strtof(string, (char**)&string);
				texcoords.push_back(texcoord);
			} else if (*string == 'n') { // "vn"
				string += 1;
				Vector3 normal;
				normal.x = strtof(string, (char**)&string);
				normal.y = strtof(string, (char**)&string);
				normal.z = strtof(string, (char**)&string);
				normals.push_back(normal);
			} else { // "v"
				Vector3 position;
				position.x = strtof(string, (char**)&string);
				position.y = strtof(string, (char**)&string);
				position.z = strtof(string, (char**)&string);
				positions.push_back(position);
			}
			
			break;
		} case 'f': {
			string += 1;
			for (u32 i = 0; i < 3; i += 1) {
				u32 position_index = (u32)strtoull(string, (char**)&string, 10) - 1;
				if (*string) string += 1;
			
				u32 texcoord_index = (u32)strtoull(string, (char**)&string, 10) - 1;
				if (*string) string += 1;

				u32 normal_index = (u32)strtoull(string, (char**)&string, 10) - 1;
				if (*string) string += 1;

				u64 key = (u64)position_index | ((u64)texcoord_index << 21) | ((u64)normal_index << 42);

				auto [it, is_inserted] = index_map.emplace(key, 0u);

				if (is_inserted) {
					ObjVertex vertex;
					vertex.position = positions[position_index];
					vertex.texcoord = texcoords[texcoord_index];
					vertex.normal   = normals[normal_index];
					it->second = (u32)mesh.vertices.size();
					mesh.vertices.push_back(vertex);
				}

				mesh.indices.push_back(it->second);
			}
			break;
		} default: {
			return {};
		}
		}
	}
	
	return mesh;
}

void WriteWavefrontObjFile(const MeshDecimationResult& mesh) {
	auto* file = fopen("D:/Dev/MeshDecimation/Meshes/Output/StanfordDragon.obj", "wb");
	if (file == nullptr) return;
	
	fprintf(file, "# MeshDecimation\n");
	fprintf(file, "o Object\n");
	
	u32 vertex_count = mesh.vertices.count / obj_vertex_stride_dwords;
	auto* vertices = (ObjVertex*)mesh.vertices.data;
	for (u32 index = 0; index < vertex_count; index += 1) {
		auto& v = vertices[index];
		fprintf(file, "v %f %f %f\n", v.position.x, v.position.y, v.position.z);
		fprintf(file, "vn %f %f %f\n", v.normal.x, v.normal.y, v.normal.z);
		fprintf(file, "vt %f %f\n", v.texcoord.x, v.texcoord.y);
	}
	
	for (u32 corner = 0; corner < mesh.indices.count; corner += 3) {
		u32 i0 = mesh.indices[corner + 0] + 1;
		u32 i1 = mesh.indices[corner + 1] + 1;
		u32 i2 = mesh.indices[corner + 2] + 1;
		fprintf(file, "f %u/%u/%u %u/%u/%u %u/%u/%u\n", i0, i0, i0, i1, i1, i1, i2, i2, i2);
	}
	
	fclose(file);
}

void WriteWavefrontObjFile(const VirtualGeometryBuildResult& mesh) {
	auto* file = fopen("D:/Dev/MeshDecimation/Meshes/Output/StanfordDragon.obj", "wb");
	if (file == nullptr) return;
	
	fprintf(file, "# MeshDecimation\n");
	fprintf(file, "o Object\n");
	
	u32 vertex_count = mesh.vertices.count / obj_vertex_stride_dwords;
	auto* vertices = (ObjVertex*)mesh.vertices.data;
	for (u32 index = 0; index < vertex_count; index += 1) {
		auto& v = vertices[index];
		fprintf(file, "v %f %f %f\n", v.position.x, v.position.y, v.position.z);
		fprintf(file, "vn %f %f %f\n", v.normal.x, v.normal.y, v.normal.z);
		fprintf(file, "vt %f %f\n", v.texcoord.x, v.texcoord.y);
	}
	
	// Output an LOD with a given target error.
	float target_error = 0.05f;
	
	u32 group_index = ~0u;
	for (u32 meshlet_index = 0; meshlet_index < mesh.meshlets.count; meshlet_index += 1) {
		auto& meshlet = mesh.meshlets[meshlet_index];
		
		bool draw_current_level = (meshlet.current_level_error_metric.error <= target_error);
		bool draw_coarser_level = (meshlet.coarser_level_error_metric.error <= target_error);
		bool draw_meshlet       = draw_current_level && !draw_coarser_level;
		
		if (draw_meshlet == false) continue;
		
#if 0
		if (group_index != meshlet.coarser_level_meshlet_group_index) {
			group_index = meshlet.coarser_level_meshlet_group_index;
			fprintf(file, "o Group%u\n", group_index);
		}
#else
		fprintf(file, "o Meshlet%u\n", meshlet_index);
#endif
		
		for (u32 i = meshlet.begin_meshlet_triangles_index; i < meshlet.end_meshlet_triangles_index; i += 3) {
			u32 i0 = mesh.meshlet_vertex_indices[mesh.meshlet_triangles[i + 0] + meshlet.begin_vertex_indices_index] + 1;
			u32 i1 = mesh.meshlet_vertex_indices[mesh.meshlet_triangles[i + 1] + meshlet.begin_vertex_indices_index] + 1;
			u32 i2 = mesh.meshlet_vertex_indices[mesh.meshlet_triangles[i + 2] + meshlet.begin_vertex_indices_index] + 1;
			
			fprintf(file, "f %u/%u/%u %u/%u/%u %u/%u/%u\n", i0, i0, i0, i1, i1, i1, i2, i2, i2);
		}
	}
	
	fclose(file);
}

#define ENABLE_ALLOCATOR_VALIDATION 1

struct ValidatedAllocator {
#if ENABLE_ALLOCATOR_VALIDATION
	u32 allocation_count   = 0;
	u32 deallocation_count = 0;
#endif // ENABLE_ALLOCATOR_VALIDATION
};

static void* ValidatedAllocatorRealloc(void* old_memory_block, u64 size_bytes, void* user_data) {
#if ENABLE_ALLOCATOR_VALIDATION
	auto* allocator = (ValidatedAllocator*)user_data;
	if (old_memory_block != nullptr && size_bytes != 0) { // Reallocate.
		allocator->allocation_count   += 1;
		allocator->deallocation_count += 1;
	} else if (old_memory_block && size_bytes == 0) { // Deallocate.
		allocator->deallocation_count += 1;
	} else if (old_memory_block == nullptr && size_bytes != 0) { // Allocate.
		allocator->allocation_count += 1;
	} else {
		assert(old_memory_block == nullptr && size_bytes == 0); // No op.
	}
#endif // ENABLE_ALLOCATOR_VALIDATION
	
	return realloc(old_memory_block, size_bytes);
}

#include <chrono>

int main() {
	auto t0 = std::chrono::high_resolution_clock::now();
	
	auto* file = fopen("D:/Dev/MeshDecimation/Meshes/StanfordDragon.obj", "rb");
	if (file == nullptr) return -1;
	
	fseek(file, 0, SEEK_END);
	u64 file_size = ftell(file);
	fseek(file, 0, SEEK_SET);
	
	std::vector<char> file_data;
	file_data.resize(file_size + 1);
	fread(file_data.data(), 1, file_size, file);
	file_data[file_size] = 0;
	
	auto t1 = std::chrono::high_resolution_clock::now();
	printf("Read File Time: %llums\n", std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
	
	t0 = std::chrono::high_resolution_clock::now();
	auto triangle_mesh = ParseWavefrontObj(file_data.data());
	t1 = std::chrono::high_resolution_clock::now();
	printf("Parse File Time: %llums\n", std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
	
	t0 = std::chrono::high_resolution_clock::now();
	
#if 0
	u32 split_index = (((u32)triangle_mesh.indices.size() / 3) / 2) * 3;
	
	compile_const u32 geometry_desc_count = 2;
	TriangleGeometryDesc geometry_descs[geometry_desc_count];
	geometry_descs[0].indices      = triangle_mesh.indices.data() + split_index;
	geometry_descs[0].index_count  = (u32)triangle_mesh.indices.size() - split_index;
	geometry_descs[0].vertices     = (float*)triangle_mesh.vertices.data();
	geometry_descs[0].vertex_count = (u32)triangle_mesh.vertices.size();
	
	geometry_descs[1].indices      = triangle_mesh.indices.data();
	geometry_descs[1].index_count  = split_index;
	geometry_descs[1].vertices     = (float*)triangle_mesh.vertices.data();
	geometry_descs[1].vertex_count = (u32)triangle_mesh.vertices.size();
#else
	compile_const u32 geometry_desc_count = 1;
	TriangleGeometryDesc geometry_descs[geometry_desc_count];
	geometry_descs[0].indices      = triangle_mesh.indices.data();
	geometry_descs[0].index_count  = (u32)triangle_mesh.indices.size();
	geometry_descs[0].vertices     = (float*)triangle_mesh.vertices.data();
	geometry_descs[0].vertex_count = (u32)triangle_mesh.vertices.size();
#endif
	
	compile_const float uv_weight     = 1.f;
	compile_const float normal_weight = 1.f;
	
	float attribute_weights[5];
	attribute_weights[0] = uv_weight;
	attribute_weights[1] = uv_weight;
	attribute_weights[2] = normal_weight;
	attribute_weights[3] = normal_weight;
	attribute_weights[4] = normal_weight;
	
	TriangleMeshDesc mesh_desc;
	mesh_desc.geometry_descs      = geometry_descs;
	mesh_desc.geometry_desc_count = geometry_desc_count;
	mesh_desc.vertex_stride_bytes = sizeof(ObjVertex);
	mesh_desc.attribute_weights   = attribute_weights;
	mesh_desc.normalize_vertex_attributes = &NormalizeObjVertexAttributes;
	
	ValidatedAllocator temp_allocator;
	ValidatedAllocator heap_allocator;
	
	SystemCallbacks callbacks;
	callbacks.temp_allocator.realloc   = &ValidatedAllocatorRealloc;
	callbacks.temp_allocator.user_data = &temp_allocator;
	callbacks.heap_allocator.realloc   = &ValidatedAllocatorRealloc;
	callbacks.heap_allocator.user_data = &heap_allocator;
	
#if 1
	VirtualGeometryBuildInputs inputs;
	inputs.mesh = mesh_desc;
	
	VirtualGeometryBuildResult result;
	BuildVirtualGeometry(inputs, result, callbacks);

#if ENABLE_ALLOCATOR_VALIDATION
	assert(heap_allocator.allocation_count == heap_allocator.deallocation_count + 6); // 6 live heap allocations.
#endif // ENABLE_ALLOCATOR_VALIDATION
#else
	MeshDecimationInputs inputs;
	inputs.mesh = mesh_desc;
	inputs.target_face_count = ((u32)triangle_mesh.indices.size() / 3) / 138;
	
	MeshDecimationResult result;
	DecimateMesh(inputs, result, callbacks);
	
#if ENABLE_ALLOCATOR_VALIDATION
	assert(heap_allocator.allocation_count == heap_allocator.deallocation_count + 2); // 2 live heap allocations.
#endif // ENABLE_ALLOCATOR_VALIDATION
#endif
	
	t1 = std::chrono::high_resolution_clock::now();
	printf("Decimation Time: %llums\n", std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
	
#if ENABLE_ALLOCATOR_VALIDATION
	assert(temp_allocator.allocation_count == temp_allocator.deallocation_count); // No live temp allocations.
#endif // ENABLE_ALLOCATOR_VALIDATION
	
	WriteWavefrontObjFile(result);
	FreeResultBuffers(result, callbacks);

#if ENABLE_ALLOCATOR_VALIDATION
	printf("Temp Allocation Count: %u\n", temp_allocator.allocation_count);
	printf("Heap Allocation Count: %u\n", heap_allocator.allocation_count);
	assert(heap_allocator.allocation_count == heap_allocator.deallocation_count); // No live heap allocations.
#endif // ENABLE_ALLOCATOR_VALIDATION
	
	fclose(file);
	
	return 0;
}




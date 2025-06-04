#include "MeshDecimation.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unordered_map>
#include <vector>
#include <chrono>

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

using Vector3 = VgtVector3;

struct ObjVertex {
	Vector3 position;
	Vector2 texcoord;
	Vector3 normal;
};

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


struct ObjMeshIndexRange {
	u32 begin_indices_index = 0;
	u32 end_indices_index   = 0;
};

struct ObjTriangleMesh {
	std::vector<u32> indices;
	std::vector<ObjVertex> vertices;
	std::vector<ObjMeshIndexRange> index_ranges;
};

static ObjTriangleMesh ParseWavefrontObj(FILE* source_file) {
	fseek(source_file, 0, SEEK_END);
	u64 file_size = ftell(source_file);
	fseek(source_file, 0, SEEK_SET);
	
	std::vector<char> file_data;
	file_data.resize(file_size + 1);
	fread(file_data.data(), 1, file_size, source_file);
	file_data[file_size] = 0;
	
	const char* string = file_data.data();
	
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
		} case 'o': {
			if (mesh.index_ranges.size()) {
				mesh.index_ranges.back().end_indices_index = (u32)mesh.indices.size();
			}
			
			mesh.index_ranges.emplace_back();
			mesh.index_ranges.back().begin_indices_index = (u32)mesh.indices.size();
			
			string = EatSingleLineComment(string + 1);
			break;
		} default: {
			return {};
		}
		}
	}
	
	if (mesh.index_ranges.size() == 0) mesh.index_ranges.emplace_back();
	mesh.index_ranges.back().end_indices_index = (u32)mesh.indices.size();
	
	return mesh;
}



struct CommandLineOptions {
	const char* source_file_name = nullptr;
	const char* result_file_name = nullptr;
	
	bool dlod = false;
	bool clod = false;
};

static bool ParseCommandLineOptions(u32 argument_count, char** arguments, CommandLineOptions& options) {
	if (argument_count != 4) {
		printf("Error: Unexpected number of arguments: %u, expected: 3.\n", argument_count - 1);
		
		printf("\nTo build discrete LOD use this command. Output file will contain all levels of detail.\n");
		printf("    MeshDecimationDemo -d SourceMeshPath.obj ResultMeshPath.obj\n");
		
		printf("\nTo build continuous LOD use this command. Output file will contain a slice of CLOD tree with 0.05 unit error.\n");
		printf("    MeshDecimationDemo -c SourceMeshPath.obj ResultMeshPath.obj\n");
		return false;
	}
	
	options.dlod = strcmp(arguments[1], "-d") == 0;
	options.clod = strcmp(arguments[1], "-c") == 0;
	
	if (options.dlod == false && options.clod == false) {
		printf("Error: Unexpected value of argument 2. Expected '-d' for Discrete LOD or '-c' for Continuous LOD. Given '%s'.", arguments[1]);
		return false;
	}
	
	options.source_file_name = arguments[2];
	options.result_file_name = arguments[3];
	
	return true;
}


void WriteWavefrontObjFileDLOD(const VgtMeshDecimationResult& mesh, FILE* file) {
	fprintf(file, "# MeshDecimation\n");
	
	auto* vertices = (ObjVertex*)mesh.vertices;
	for (u32 index = 0; index < mesh.vertex_count; index += 1) {
		auto& v = vertices[index];
		fprintf(file, "v %f %f %f\n", v.position.x, v.position.y, v.position.z);
		fprintf(file, "vn %f %f %f\n", v.normal.x, v.normal.y, v.normal.z);
		fprintf(file, "vt %f %f\n", v.texcoord.x, v.texcoord.y);
	}
	
	for (u32 level_index = 0; level_index < mesh.level_of_detail_count; level_index += 1) {
		auto& lod_desc = mesh.level_of_detail_descs[level_index];
		
		for (u32 geometry_index = lod_desc.begin_geometry_index; geometry_index < lod_desc.end_geometry_index; geometry_index += 1) {
			auto& geometry_desc = mesh.geometry_descs[geometry_index];
			
			fprintf(file, "o LOD%u-Geometry%u\n", level_index, geometry_index - lod_desc.begin_geometry_index);
			
			for (u32 corner = geometry_desc.begin_indices_index; corner < geometry_desc.end_indices_index; corner += 3) {
				u32 i0 = mesh.indices[corner + 0] + 1;
				u32 i1 = mesh.indices[corner + 1] + 1;
				u32 i2 = mesh.indices[corner + 2] + 1;
				fprintf(file, "f %u/%u/%u %u/%u/%u %u/%u/%u\n", i0, i0, i0, i1, i1, i1, i2, i2, i2);
			}
		}
	}
}

void WriteWavefrontObjFileCLOD(const VgtVirtualGeometryBuildResult& mesh, FILE* file) {
	fprintf(file, "# MeshDecimation\n");
	fprintf(file, "o Object\n");
	
	auto* vertices = (ObjVertex*)mesh.vertices;
	for (u32 index = 0; index < mesh.vertex_count; index += 1) {
		auto& v = vertices[index];
		fprintf(file, "v %f %f %f\n", v.position.x, v.position.y, v.position.z);
		fprintf(file, "vn %f %f %f\n", v.normal.x, v.normal.y, v.normal.z);
		fprintf(file, "vt %f %f\n", v.texcoord.x, v.texcoord.y);
	}
	
	// Output an LOD with a given target error.
	float target_error = 0.05f;
	
	u32 group_index = ~0u;
	for (u32 meshlet_index = 0; meshlet_index < mesh.meshlet_count; meshlet_index += 1) {
		auto& meshlet = mesh.meshlets[meshlet_index];
		
		bool draw_current_level = (meshlet.current_level_error_metric.error <= target_error);
		bool draw_coarser_level = (meshlet.coarser_level_error_metric.error <= target_error);
		bool draw_meshlet       = draw_current_level && !draw_coarser_level;
		
		if (draw_meshlet == false) continue;
		
#if 0
		if (group_index != meshlet.coarser_level_meshlet_group_index) {
			group_index = meshlet.coarser_level_meshlet_group_index;
			fprintf(file, "o Group%u\n", group_index); // Note that groups might span multiple geometries.
		}
#else
		fprintf(file, "o Meshlet%u-Geometry%u\n", meshlet_index, meshlet.geometry_index);
#endif
		
		for (u32 i = meshlet.begin_meshlet_triangles_index; i < meshlet.end_meshlet_triangles_index; i += 1) {
			auto triangle = mesh.meshlet_triangles[i];
			u32 i0 = mesh.meshlet_vertex_indices[triangle.i0 + meshlet.begin_vertex_indices_index] + 1;
			u32 i1 = mesh.meshlet_vertex_indices[triangle.i1 + meshlet.begin_vertex_indices_index] + 1;
			u32 i2 = mesh.meshlet_vertex_indices[triangle.i2 + meshlet.begin_vertex_indices_index] + 1;
			
			fprintf(file, "f %u/%u/%u %u/%u/%u %u/%u/%u\n", i0, i0, i0, i1, i1, i1, i2, i2, i2);
		}
	}
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
		VGT_ASSERT(old_memory_block == nullptr && size_bytes == 0); // No op.
	}
#endif // ENABLE_ALLOCATOR_VALIDATION
	
	return realloc(old_memory_block, size_bytes);
}


int main(int argument_count, char** arguments) {
	CommandLineOptions options = {};
	if (ParseCommandLineOptions(argument_count, arguments, options) == false) return -1;
	
	auto* source_file = options.source_file_name ? fopen(options.source_file_name, "rb") : nullptr;
	if (source_file == nullptr) {
		printf("Error: Cannot open source file '%s' for read.\n", options.source_file_name);
		return -1;
	}
	
	auto* result_file = options.result_file_name ? fopen(options.result_file_name, "wb") : nullptr;
	if (result_file == nullptr) {
		printf("Error: Cannot open result file '%s' for write.\n", options.result_file_name);
		return -1;
	}
	
	printf("Building %s LOD:\n", options.dlod ? "Discrete" : "Continuous");
	printf("    Source Mesh Path: '%s'\n", options.source_file_name);
	printf("    Result Mesh Path: '%s'\n", options.result_file_name);
	printf("\n");
	
	
	auto t0 = std::chrono::high_resolution_clock::now();
	auto triangle_mesh = ParseWavefrontObj(source_file);
	auto t1 = std::chrono::high_resolution_clock::now();
	printf("Parse File Time: %llums\n", std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
	
	
	std::vector<VgtTriangleGeometryDesc> geometry_descs;
	geometry_descs.reserve(triangle_mesh.index_ranges.size());
	for (auto& range : triangle_mesh.index_ranges) {
		auto& geometry_desc = geometry_descs.emplace_back();
		geometry_desc.indices      = triangle_mesh.indices.data() + range.begin_indices_index;
		geometry_desc.index_count  = range.end_indices_index - range.begin_indices_index;
		geometry_desc.vertices     = (float*)triangle_mesh.vertices.data();
		geometry_desc.vertex_count = (u32)triangle_mesh.vertices.size();
	}
	
	compile_const float uv_weight     = 1.f;
	compile_const float normal_weight = 1.f;
	
	float attribute_weights[VGT_MAX_ATTRIBUTE_STRIDE_DWORDS] = {};
	attribute_weights[0] = uv_weight;
	attribute_weights[1] = uv_weight;
	attribute_weights[2] = normal_weight;
	attribute_weights[3] = normal_weight;
	attribute_weights[4] = normal_weight;
	
	VgtTriangleMeshDesc mesh_desc = {};
	mesh_desc.geometry_descs      = geometry_descs.data();
	mesh_desc.geometry_desc_count = (u32)geometry_descs.size();
	mesh_desc.vertex_stride_bytes = sizeof(ObjVertex);
	mesh_desc.attribute_weights   = attribute_weights;
	mesh_desc.normalize_vertex_attributes = &NormalizeObjVertexAttributes;
	
	ValidatedAllocator temp_allocator = {};
	ValidatedAllocator heap_allocator = {};
	
	VgtSystemCallbacks callbacks = {};
	callbacks.temp_allocator.realloc   = &ValidatedAllocatorRealloc;
	callbacks.temp_allocator.user_data = &temp_allocator;
	callbacks.heap_allocator.realloc   = &ValidatedAllocatorRealloc;
	callbacks.heap_allocator.user_data = &heap_allocator;
	
	if (options.clod) {
		VgtVirtualGeometryBuildInputs inputs = {};
		inputs.mesh                          = mesh_desc;
		inputs.meshlet_target_vertex_count   = 128;
		inputs.meshlet_target_triangle_count = 128;
		
		t0 = std::chrono::high_resolution_clock::now();
		
		VgtVirtualGeometryBuildResult result = {};
		VgtBuildVirtualGeometry(&inputs, &result, &callbacks);
		
		t1 = std::chrono::high_resolution_clock::now();
		printf("CLOD Build Time: %llums\n", std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
		
#if ENABLE_ALLOCATOR_VALIDATION
		VGT_ASSERT(temp_allocator.allocation_count == temp_allocator.deallocation_count); // No live temp allocations.
		VGT_ASSERT(heap_allocator.allocation_count == heap_allocator.deallocation_count + 6); // 6 live heap allocations.
#endif // ENABLE_ALLOCATOR_VALIDATION
		
		WriteWavefrontObjFileCLOD(result, result_file);
		VgtFreeVirtualGeometryBuildResult(&result, &callbacks);
	} else if (options.dlod) {
		compile_const u32 max_level_of_detail_desc_count = 16;
		VgtLevelOfDetailTargetDesc level_of_detail_descs[max_level_of_detail_desc_count] = {};
		
		u32 source_mesh_face_count = (u32)(triangle_mesh.indices.size() / 3);
		u32 level_of_detail_count = 0;
		
		for (u32 level_index = 0; level_index < max_level_of_detail_desc_count; level_index += 1, level_of_detail_count = level_index) {
			u32 target_face_count = (source_mesh_face_count >> (level_index + 1));
			if (target_face_count <= 32) break;
			
			level_of_detail_descs[level_index].target_face_count  = target_face_count;
			level_of_detail_descs[level_index].target_error_limit = FLT_MAX;
		}
		
		VgtMeshDecimationInputs inputs = {};
		inputs.mesh                  = mesh_desc;
		inputs.level_of_detail_descs = level_of_detail_descs;
		inputs.level_of_detail_count = level_of_detail_count;
		
		t0 = std::chrono::high_resolution_clock::now();
		
		VgtMeshDecimationResult result = {};
		VgtDecimateMesh(&inputs, &result, &callbacks);
		
		t1 = std::chrono::high_resolution_clock::now();
		printf("DLOD Build Time: %llums\n", std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
		
#if ENABLE_ALLOCATOR_VALIDATION
		VGT_ASSERT(temp_allocator.allocation_count == temp_allocator.deallocation_count); // No live temp allocations.
		VGT_ASSERT(heap_allocator.allocation_count == heap_allocator.deallocation_count + 4); // 4 live heap allocations.
#endif // ENABLE_ALLOCATOR_VALIDATION
		
		WriteWavefrontObjFileDLOD(result, result_file);
		VgtFreeMeshDecimationResult(&result, &callbacks);
	}
	
#if ENABLE_ALLOCATOR_VALIDATION
	printf("Temp Allocation Count: %u\n", temp_allocator.allocation_count);
	printf("Heap Allocation Count: %u\n", heap_allocator.allocation_count);
	VGT_ASSERT(heap_allocator.allocation_count == heap_allocator.deallocation_count); // No live heap allocations.
#endif // ENABLE_ALLOCATOR_VALIDATION
	
	fclose(result_file);
	fclose(source_file);
	
	return 0;
}

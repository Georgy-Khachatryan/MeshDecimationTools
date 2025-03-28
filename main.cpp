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


ObjTriangleMesh ParseWavefrontObj(const char* string) {
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

void WriteWavefrontObjFile(ObjTriangleMesh mesh) {
	auto* file = fopen("D:/Dev/MeshDecimation/Meshes/Output/NonManifoldCorner.obj", "wb");
	if (file == nullptr) return;
	
	fprintf(file, "# MeshDecimation\n");
	fprintf(file, "o Object\n");
	
	for (u32 index = 0; index < mesh.vertices.size(); index += 1) {
		auto& v = mesh.vertices[index];
		fprintf(file, "v %f %f %f\n", v.position.x, v.position.y, v.position.z);
		fprintf(file, "vn %f %f %f\n", v.normal.x, v.normal.y, v.normal.z);
		fprintf(file, "vt %f %f\n", v.texcoord.x, v.texcoord.y);
	}
	
	for (u32 corner = 0; corner < mesh.indices.size(); corner += 3) {
		u32 i0 = mesh.indices[corner + 0] + 1;
		u32 i1 = mesh.indices[corner + 1] + 1;
		u32 i2 = mesh.indices[corner + 2] + 1;
		fprintf(file, "f %u/%u/%u %u/%u/%u %u/%u/%u\n", i0, i0, i0, i1, i1, i1, i2, i2, i2);
	}
	
	fclose(file);
}

void WriteWavefrontObjFile(MeshView mesh) {
	auto* file = fopen("D:/Dev/MeshDecimation/Meshes/Output/NonManifoldCorner.obj", "wb");
	if (file == nullptr) return;
	
	fprintf(file, "# MeshDecimation\n");
	fprintf(file, "o Object\n");
	
	for (VertexID vertex_id = { 0 }; vertex_id.index < mesh.vertex_count; vertex_id.index += 1) {
		auto& v = mesh[vertex_id];
		fprintf(file, "v %f %f %f\n", v.position.x, v.position.y, v.position.z);
	}
	
	for (AttributesID attribute_id = { 0 }; attribute_id.index < mesh.attribute_count; attribute_id.index += 1) {
		auto* a = mesh[attribute_id];
		fprintf(file, "vn %f %f %f\n", a[2], a[3], a[4]);
		fprintf(file, "vt %f %f\n", a[0], a[1]);
	}
	
	for (FaceID face_id = { 0 }; face_id.index < mesh.face_count; face_id.index += 1) {
		auto& f = mesh[face_id];
		if (f.corner_list_base.index == u32_max) continue;
		
		fprintf(file, "f");

		auto face_corner_list_base_id = f.corner_list_base;
		auto current_corner_id = face_corner_list_base_id;
		do {
			auto& c0 = mesh[current_corner_id];
			u32 vi0 = c0.vertex_id.index + 1;
			u32 ai0 = c0.attributes_id.index + 1;
			
			fprintf(file, " %u/%u/%u", vi0, ai0, ai0);
		
			current_corner_id = mesh[current_corner_id].corner_list_around[(u32)ElementType::Face].next;
		} while (current_corner_id.index != face_corner_list_base_id.index);

		fprintf(file, "\n");
	}
	
	fclose(file);
}

int main() {
	auto* file = fopen("D:/Dev/MeshDecimation/Meshes/NonManifoldCorner.obj", "rb");
	if (file == nullptr) return -1;
	
	fseek(file, 0, SEEK_END);
	u64 file_size = ftell(file);
	fseek(file, 0, SEEK_SET);
	
	std::vector<char> file_data;
	file_data.resize(file_size + 1);
	fread(file_data.data(), 1, file_size, file);
	file_data[file_size] = 0;
	
	auto triangle_mesh = ParseWavefrontObj(file_data.data());
	// WriteWavefrontObjFile(triangle_mesh);
	
	auto editable_mesh = ObjMeshToEditableMesh(triangle_mesh);

	auto mesh = MeshToMeshView(editable_mesh);
	PerformRandomEdgeCollapse(mesh);
	
	WriteWavefrontObjFile(mesh);
	
	fclose(file);
	
	return 0;
}




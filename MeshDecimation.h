#ifndef MESHDECIMATION_H
#define MESHDECIMATION_H

#include <vector>

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
};

struct Vector2 {
	float x = 0.f;
	float y = 0.f;
};


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
compile_const u32 attribute_stride_dwords = 5; // 3 normal + 2 texcoords.


struct Face {
	CornerID corner_list_base; // Corner list around a face.
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

struct Vertex {
	Vector3 position;
	CornerID corner_list_base; // Corner list around a vertex.
};


struct Mesh {
	std::vector<Face>   faces;
	std::vector<Edge>   edges;
	std::vector<Vertex> vertices;
	std::vector<Corner> corners;
	std::vector<float>  attributes;
};

struct MeshView {
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
	
	Face&   operator[] (FaceID face_id)             { return faces[face_id.index]; }
	Edge&   operator[] (EdgeID edge_id)             { return edges[edge_id.index]; }
	Vertex& operator[] (VertexID vertex_id)         { return vertices[vertex_id.index]; }
	Corner& operator[] (CornerID corner_id)         { return corners[corner_id.index]; }
	float*  operator[] (AttributesID attributes_id) { return attributes + attributes_id.index * attribute_stride_dwords; }
};

MeshView MeshToMeshView(Mesh& mesh);
void PerformRandomEdgeCollapse(MeshView mesh);


struct ObjVertex {
	Vector3 position;
	Vector2 texcoord;
	Vector3 normal;
};

struct ObjTriangleMesh {
	std::vector<ObjVertex> vertices;
	std::vector<u32>       indices;
};

Mesh ObjMeshToEditableMesh(ObjTriangleMesh mesh);
ObjTriangleMesh EditableMeshToObjMesh(MeshView mesh);

#endif // MESHDECIMATION_H

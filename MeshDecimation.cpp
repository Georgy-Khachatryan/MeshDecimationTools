#include "MeshDecimation.h"

#include <unordered_map>
#include <vector>
#include <intrin.h>
#include <assert.h>
#include <math.h>

//
// References:
// - Michael Garland, Paul S. Heckbert. 1997. Surface Simplification Using Quadric Error Metrics.
// - Hugues Hoppe. 1999. New Quadric Metric for Simplifying Meshes with Appearance Attributes.
// - HSUEH-TI DEREK LIU, XIAOTING ZHANG, CEM YUKSEL. 2024. Simplifying Triangle Meshes in the Wild.
//

struct EdgeKey {
	VertexID vertex_0;
	VertexID vertex_1;
	
	friend bool operator== (EdgeKey lh, EdgeKey rh) {
		return // Edges are non directional. EdgeKey(A, B) is the same as EdgeKey(B, A)
			(lh.vertex_0.index == rh.vertex_0.index) && (lh.vertex_1.index == rh.vertex_1.index) ||
			(lh.vertex_0.index == rh.vertex_1.index) && (lh.vertex_1.index == rh.vertex_0.index);
	}
};

// fenv pragmas are an attempt at making compiler not optimize away (x + 0.f).
#pragma fenv_access(on)
struct VertexHasher {
	static u64 PositionHash(const Vector3& v) {
		compile_const u64 knuth_golden_ratio = 0x9e3779b97f4a7c55;
		
		static union {
			u64 scalars[2];
			__m128i vector;
		} seed = { knuth_golden_ratio, knuth_golden_ratio };
		
		// Add zero to turn -0.0 to +0.0.
		auto hash = _mm_castps_si128(_mm_add_ps(_mm_setr_ps(v.z, v.z, v.y, v.x), _mm_setzero_ps()));
		hash = _mm_aesdec_si128(hash, seed.vector);
		hash = _mm_aesdec_si128(hash, seed.vector);
		
		return _mm_cvtsi128_si64(hash);
	}
	
	u64 operator() (const EdgeKey& e) const { return std::hash<u64>{}(e.vertex_0.index) + std::hash<u64>{}(e.vertex_1.index); }
	u64 operator() (const Vector3& v) const { return PositionHash(v); }
};
#pragma fenv_access(off)


template<ElementType element_type_t> auto GetElementID(const Corner& corner) {
	if constexpr (element_type_t == ElementType::Vertex) {
		return corner.vertex_id;
	} else if constexpr (element_type_t == ElementType::Edge) {
		return corner.edge_id;
	} else if constexpr (element_type_t == ElementType::Face) {
		return corner.face_id;
	} else {
		static_assert(false, "Unknown element_type_t.");
	}
}

template<typename ElementID>
static void CornerListInsert(MeshView mesh, ElementID element_id, CornerID new_corner_id) {
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

template<ElementType element_type_t>
static bool CornerListRemove(MeshView mesh, CornerID corner_id) {
	compile_const u32 element_type = (u32)element_type_t;
	
	auto& corner = mesh[corner_id];
	auto prev_corner_id = corner.corner_list_around[element_type].prev;
	auto next_corner_id = corner.corner_list_around[element_type].next;
	
	mesh[next_corner_id].corner_list_around[element_type].prev = prev_corner_id;
	mesh[prev_corner_id].corner_list_around[element_type].next = next_corner_id;
	
	// Remove the element if it lost it's last corner.
	bool is_last_reference = (prev_corner_id.index == corner_id.index);
	auto new_corner_list_base = is_last_reference ? CornerID{ u32_max } : prev_corner_id;
	
	mesh[GetElementID<element_type_t>(corner)].corner_list_base = new_corner_list_base;
	
	corner.corner_list_around[element_type].prev.index = u32_max;
	corner.corner_list_around[element_type].next.index = u32_max;
	
	return is_last_reference;
}

enum struct IterationControl : u32 {
	Continue = 0,
	Break    = 1,
};

// Iterate linked list around a given element type starting with the base corner id. 
// Removal while iterating is allowed.
template<ElementType element_type_t, typename Lambda>
static void IterateCornerList(MeshView mesh, CornerID corner_list_base, Lambda&& lambda) {
	auto& element = mesh[GetElementID<element_type_t>(mesh[corner_list_base])];
	
	auto current_corner_id = corner_list_base;
	IterationControl control = IterationControl::Continue;
	do {
		auto next_corner_id = mesh[current_corner_id].corner_list_around[(u32)element_type_t].next;
		
		if constexpr (std::is_invocable_r<IterationControl, Lambda&, CornerID>::value) {
			control = lambda(current_corner_id);
		} else if constexpr (std::is_invocable_r<void, Lambda&, CornerID>::value) {
			lambda(current_corner_id);
		} else {
			static_assert(false, "Lambda passed to IterateCornerList should have signature (CornerID) -> void or (CornerID) -> IterationControl.");
		}
		
		current_corner_id = next_corner_id;
	} while (current_corner_id.index != corner_list_base.index && element.corner_list_base.index != u32_max && control == IterationControl::Continue);
}

// Merge linked lists around element_0 and element_1 and remove element_1.
// Patch up references to element_1 with a reference to element_0.
template<typename ElementID>
static void CornerListMerge(MeshView mesh, ElementID element_0, ElementID element_1) {
	auto base_id_0 = mesh[element_0].corner_list_base;
	auto base_id_1 = mesh[element_1].corner_list_base;
	
	compile_const ElementType element_type_t = ElementID::element_type;
	compile_const u32 element_type = (u32)element_type_t;
	
	if (base_id_0.index != u32_max && base_id_1.index != u32_max) {
		IterateCornerList<element_type_t>(mesh, base_id_1, [&](CornerID corner_id) {
			if constexpr (element_type_t == ElementType::Vertex) {
				mesh[corner_id].vertex_id = element_0;
			
				// TODO: This can be iteration over just incoming and outgoing edges of a corner.
				IterateCornerList<ElementType::Face>(mesh, corner_id, [&](CornerID corner_id) {
					auto& edge = mesh[mesh[corner_id].edge_id];
					if (edge.vertex_0.index == element_1.index) edge.vertex_0 = element_0;
					if (edge.vertex_1.index == element_1.index) edge.vertex_1 = element_0;
				});
			} else if constexpr (element_type_t == ElementType::Edge) {
				mesh[corner_id].edge_id = element_0;
			} else {
				static_assert(false, "Unknown element_type_t.");
			}
		});
		
		auto base_id_0_prev = mesh[base_id_0].corner_list_around[element_type].prev;
		auto base_id_1_prev = mesh[base_id_1].corner_list_around[element_type].prev;
		
		mesh[base_id_0].corner_list_around[element_type].prev = base_id_1_prev;
		mesh[base_id_1_prev].corner_list_around[element_type].next = base_id_0;
		
		mesh[base_id_1].corner_list_around[element_type].prev = base_id_0_prev;
		mesh[base_id_0_prev].corner_list_around[element_type].next = base_id_1;
		
		mesh[element_1].corner_list_base.index = u32_max;
	}
}


Mesh ObjMeshToEditableMesh(ObjTriangleMesh triangle_mesh) {
	std::vector<VertexID> src_vertex_index_to_vertex_id;
	src_vertex_index_to_vertex_id.resize(triangle_mesh.vertices.size());
	
	Mesh result_mesh;
	
	result_mesh.vertices.reserve(triangle_mesh.vertices.size());
	result_mesh.corners.resize(triangle_mesh.indices.size());
	result_mesh.faces.resize(triangle_mesh.indices.size() / 3);
	result_mesh.edges.reserve(triangle_mesh.indices.size());
	result_mesh.attributes.resize(triangle_mesh.vertices.size() * attribute_stride_dwords);
	
	auto mesh = MeshToMeshView(result_mesh);
	
	auto vertex_id_allocator = VertexID{ 0 };
	std::unordered_map<Vector3, VertexID, VertexHasher> vertex_position_to_id;

	for (u32 vertex_index = 0; vertex_index < triangle_mesh.vertices.size(); vertex_index += 1) {
		auto& position = triangle_mesh.vertices[vertex_index].position;
		memcpy(mesh[AttributesID{ vertex_index }], (float*)&triangle_mesh.vertices[vertex_index] + 3, attribute_stride_dwords * sizeof(u32));
		
		auto [it, is_inserted] = vertex_position_to_id.emplace(position, vertex_id_allocator);
		if (is_inserted) {
			auto& vertex = result_mesh.vertices.emplace_back();
			vertex.position = position;
			vertex.corner_list_base.index = u32_max;
			
			vertex_id_allocator.index += 1;
		}
		
		src_vertex_index_to_vertex_id[vertex_index] = it->second;
	}
	
	
	auto edge_id_allocator = EdgeID{ 0 };
	std::unordered_map<EdgeKey, EdgeID, VertexHasher> edge_to_edge_id;

	u32 triangle_count = (u32)(triangle_mesh.indices.size() / 3);
	for (u32 triangle_index = 0; triangle_index < triangle_count; triangle_index += 1) {
		auto face_id = FaceID{ triangle_index };
		
		u32 indices[3] = {
			triangle_mesh.indices[triangle_index * 3 + 0],
			triangle_mesh.indices[triangle_index * 3 + 1],
			triangle_mesh.indices[triangle_index * 3 + 2],
		};
		
		VertexID vertex_ids[3] = {
			src_vertex_index_to_vertex_id[indices[0]],
			src_vertex_index_to_vertex_id[indices[1]],
			src_vertex_index_to_vertex_id[indices[2]],
		};
		
		EdgeKey edge_keys[3] = {
			EdgeKey{ vertex_ids[0], vertex_ids[1] },
			EdgeKey{ vertex_ids[1], vertex_ids[2] },
			EdgeKey{ vertex_ids[2], vertex_ids[0] },
		};

		auto& face = mesh[face_id];
		face.corner_list_base.index = u32_max;
		
		for (u32 corner_index = 0; corner_index < 3; corner_index += 1) {
			auto corner_id = CornerID{ triangle_index * 3 + corner_index };
			
			auto [it, is_inserted] = edge_to_edge_id.emplace(edge_keys[corner_index], edge_id_allocator);
			if (is_inserted) {
				auto& edge = result_mesh.edges.emplace_back();
				edge.vertex_0 = edge_keys[corner_index].vertex_0;
				edge.vertex_1 = edge_keys[corner_index].vertex_1;
				edge.corner_list_base.index = u32_max;
				
				edge_id_allocator.index += 1;
			}

			auto& corner = mesh[corner_id];
			corner.face_id       = face_id;
			corner.edge_id       = it->second;
			corner.vertex_id     = vertex_ids[corner_index];
			corner.attributes_id = { indices[corner_index] };

			CornerListInsert<VertexID>(mesh, corner.vertex_id, corner_id);
			CornerListInsert<EdgeID>(mesh, corner.edge_id, corner_id);
			CornerListInsert<FaceID>(mesh, corner.face_id, corner_id);
		}
	}

	return result_mesh;
}

ObjTriangleMesh EditableMeshToObjMesh(MeshView mesh) {
	std::vector<VertexID> attribute_vertex_ids;
	attribute_vertex_ids.resize(mesh.attribute_count, VertexID{ u32_max });
	
	std::vector<u32> attribute_id_remap;
	attribute_id_remap.resize(mesh.attribute_count);
	
	
	for (VertexID vertex_id = { 0 }; vertex_id.index < mesh.vertex_count; vertex_id.index += 1) {
		auto& vertex = mesh[vertex_id];
		
		if (vertex.corner_list_base.index != u32_max) {
			IterateCornerList<ElementType::Vertex>(mesh, vertex.corner_list_base, [&](CornerID corner_id) {
				u32 attribute_index = mesh[corner_id].attributes_id.index;
				attribute_vertex_ids[attribute_index] = vertex_id;
			});
		}
	}
	
	u32 attribute_count = 0;
	for (AttributesID attribute_id = { 0 }; attribute_id.index < mesh.attribute_count; attribute_id.index += 1) {
		bool is_active = (attribute_vertex_ids[attribute_id.index].index != u32_max);
		attribute_id_remap[attribute_id.index] = attribute_count;
		attribute_count += is_active ? 1 : 0;
	}
	
	u32 face_count = 0;
	for (FaceID face_id = { 0 }; face_id.index < mesh.face_count; face_id.index += 1) {
		face_count +=  mesh[face_id].corner_list_base.index != u32_max ? 1 : 0;
	}
	
	ObjTriangleMesh triangle_mesh;
	triangle_mesh.indices.reserve(face_count * 3);
	triangle_mesh.vertices.reserve(attribute_count);
	
	for (FaceID face_id = { 0 }; face_id.index < mesh.face_count; face_id.index += 1) {
		auto& face = mesh[face_id];
		if (face.corner_list_base.index == u32_max) continue;
		
		IterateCornerList<ElementType::Face>(mesh, face.corner_list_base, [&](CornerID corner_id) {
			auto& corner = mesh[corner_id];
			triangle_mesh.indices.emplace_back(attribute_id_remap[corner.attributes_id.index]);
		});
	}
	
	for (AttributesID attribute_id = { 0 }; attribute_id.index < mesh.attribute_count; attribute_id.index += 1) {
		auto vertex_id = attribute_vertex_ids[attribute_id.index];
		if (vertex_id.index != u32_max) {
			auto& vertex = triangle_mesh.vertices.emplace_back();
			vertex.position = mesh[vertex_id].position;
			memcpy((float*)&vertex + 3, mesh[attribute_id], attribute_stride_dwords * sizeof(u32));
		}
	}
	
	return triangle_mesh;
}

MeshView MeshToMeshView(Mesh& mesh) {
	MeshView view;
	view.faces           = mesh.faces.data();
	view.edges           = mesh.edges.data();
	view.vertices        = mesh.vertices.data();
	view.corners         = mesh.corners.data();
	view.attributes      = mesh.attributes.data();
	view.face_count      = (u32)mesh.faces.size();
	view.edge_count      = (u32)mesh.edges.size();
	view.vertex_count    = (u32)mesh.vertices.size();
	view.corner_count    = (u32)mesh.corners.size();
	view.attribute_count = (u32)mesh.attributes.size() / attribute_stride_dwords;
	return view;
}


static CornerID PerformEdgeCollapse(MeshView mesh, EdgeID edge_id) {
	auto& edge = mesh[edge_id];
	
	auto& vertex_0 = mesh[edge.vertex_0];
	auto& vertex_1 = mesh[edge.vertex_1];

	assert(edge.corner_list_base.index != u32_max);
	assert(vertex_0.corner_list_base.index != u32_max);
	assert(vertex_1.corner_list_base.index != u32_max);
	
	IterateCornerList<ElementType::Edge>(mesh, edge.corner_list_base, [&](CornerID corner_id) {
		auto& corner = mesh[corner_id];
		auto& face   = mesh[corner.face_id];
		
		IterateCornerList<ElementType::Face>(mesh, face.corner_list_base, [&](CornerID corner_id) {
			CornerListRemove<ElementType::Vertex>(mesh, corner_id);
			CornerListRemove<ElementType::Edge>(mesh, corner_id);
			CornerListRemove<ElementType::Face>(mesh, corner_id);
		});
	});
	
	CornerListMerge<VertexID>(mesh, edge.vertex_0, edge.vertex_1);
	
	auto remaning_base_id = vertex_0.corner_list_base.index != u32_max ? vertex_0.corner_list_base : vertex_1.corner_list_base;
	if (remaning_base_id.index != u32_max) {
		std::unordered_map<EdgeKey, EdgeID, VertexHasher> edge_duplicate_map;
		
		IterateCornerList<ElementType::Vertex>(mesh, remaning_base_id, [&](CornerID corner_id) {
			// TODO: This can be iteration over just incoming and outgoing edges of a corner.
			IterateCornerList<ElementType::Face>(mesh, corner_id, [&](CornerID corner_id) {
				auto edge_id_1 = mesh[corner_id].edge_id;
				auto& edge_1 = mesh[edge_id_1];
				
				auto [it, is_inserted] = edge_duplicate_map.emplace(EdgeKey{ edge_1.vertex_0, edge_1.vertex_1 }, edge_id_1);
				auto edge_id_0 = it->second;
				
				if (is_inserted == false && edge_id_0.index != edge_id_1.index) {
					CornerListMerge<EdgeID>(mesh, edge_id_0, edge_id_1);
				}
			});
		});
	}
	
	return remaning_base_id;
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

#define ENABLE_ATTRIBUTE_SUPPORT 1

#if ENABLE_ATTRIBUTE_SUPPORT
struct QuadricAttributeData {
	Vector3 g = { 0.f, 0.f, 0.f };
	float d = 0.f;
};
#endif // ENABLE_ATTRIBUTE_SUPPORT


static Vector3 operator+ (const Vector3& lh, const Vector3& rh) { return Vector3{ lh.x + rh.x, lh.y + rh.y, lh.z + rh.z }; }
static Vector3 operator- (const Vector3& lh, const Vector3& rh) { return Vector3{ lh.x - rh.x, lh.y - rh.y, lh.z - rh.z }; }
static Vector3 operator* (const Vector3& lh, float rh) { return Vector3{ lh.x * rh, lh.y * rh, lh.z * rh }; }
static float DotProduct(const Vector3& lh, const Vector3& rh) { return lh.x * rh.x + lh.y * rh.y + lh.z * rh.z; }
static float Length(const Vector3& v) { return sqrtf(DotProduct(v, v)); }
static Vector3 Normalize(const Vector3& v) { float length = Length(v); return length < FLT_EPSILON ? v : v * (1.f / length); }
static Vector3 CrossProduct(const Vector3& lh, const Vector3& rh) { return Vector3{ lh.y * rh.z - lh.z * rh.y, lh.z * rh.x - lh.x * rh.z, lh.x * rh.y - lh.y * rh.x }; }


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

#if ENABLE_ATTRIBUTE_SUPPORT
static void AccumulateQuadricAttribute(QuadricAttributeData* accumulators, const QuadricAttributeData* quadrics) {
	for (u32 i = 0; i < attribute_stride_dwords; i += 1) {
		auto& accumulator = accumulators[i];
		auto& quadric     = quadrics[i];

		accumulator.g.x += quadric.g.x;
		accumulator.g.y += quadric.g.y;
		accumulator.g.z += quadric.g.z;
		accumulator.d   += quadric.d;
	}
}
#endif // ENABLE_ATTRIBUTE_SUPPORT

static Quadric ComputeQuadric(const Vector3& normal, float distance_to_triangle, float weight) {
	//
	// For reference see [Hugues Hoppe 1999] Section 3 Previous Quadric Error Metrics.
	//
	// A = normal * normal^T
	// b = distance_to_triangle * normal
	// c = distance_to_triangle^2
	//
	// (n.x)
	// (n.y) * (n.x, n.y, n.z)
	// (n.z)
	//
	Quadric quadric;
	quadric.a00 = (normal.x * normal.x) * weight;
	quadric.a11 = (normal.y * normal.y) * weight;
	quadric.a22 = (normal.z * normal.z) * weight;
	quadric.a01 = (normal.x * normal.y) * weight;
	quadric.a02 = (normal.x * normal.z) * weight;
	quadric.a12 = (normal.y * normal.z) * weight;
	
	quadric.b.x = (normal.x * distance_to_triangle) * weight;
	quadric.b.y = (normal.y * distance_to_triangle) * weight;
	quadric.b.z = (normal.z * distance_to_triangle) * weight;
	
	quadric.c = (distance_to_triangle * distance_to_triangle) * weight;
	quadric.weight = weight;
	
	return quadric;
}

#if ENABLE_ATTRIBUTE_SUPPORT
compile_const float attribute_weight = 1.f / (1024.f * 1024.f);

static void ComputeAttributeQuadric(Quadric& quadric, QuadricAttributeData* quadric_attribute, const Vector3& p0, const Vector3& p1, const Vector3& p2, float* a0, float* a1, float* a2, float weight) {
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
	auto p10 = p1 - p0;
	auto p20 = p2 - p0;
	auto n   = Normalize(CrossProduct(p10, p20));
	
	// Compute determinant of a 3x3 matrix A with rows p10, p20, n.
	float det0 = p10.x * (p20.y * n.z - p20.z * n.y);
	float det1 = p10.y * (p20.z * n.x - p20.x * n.z);
	float det2 = p10.z * (p20.x * n.y - p20.y * n.x);
	float determinant = det0 + det1 + det2;
	float determinant_rcp = determinant == 0.f ? 0.f : 1.f / determinant;
	
	// Compute first two colums of A^-1.
	float a_inv_00 = (p20.y * n.z - p20.z * n.y) * determinant_rcp;
	float a_inv_01 = (p10.z * n.y - p10.y * n.z) * determinant_rcp;
	float a_inv_10 = (p20.z * n.x - p20.x * n.z) * determinant_rcp;
	float a_inv_11 = (p10.x * n.z - p10.z * n.x) * determinant_rcp;
	float a_inv_20 = (p20.x * n.y - p20.y * n.x) * determinant_rcp;
	float a_inv_21 = (p10.y * n.x - p10.x * n.y) * determinant_rcp;
	
	for (u32 i = 0; i < attribute_stride_dwords; i += 1) {
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
		
		quadric_attribute[i].g = g * weight;
		quadric_attribute[i].d = d * weight;
	}
}
#endif // ENABLE_ATTRIBUTE_SUPPORT

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

#if ENABLE_ATTRIBUTE_SUPPORT
static float ComputeQuadricError(const Quadric& q, const QuadricAttributeData* a, const Vector3& p, float* attributes) {
	//
	// error = p^T * A * p + 2 * b * v + c
	//
	//                           ( a00,   a01,   a02,  -g0.x, -gi.x)   (p.x)
	// (p.x, p.y, p.z, s0, si) * ( a01,   a11,   a12,  -g0.y, -gi.y) * (p.y) + 2 * (b, -d0, -di) * (p, s0, si) + (c + d0^2 + di^2)
	//                           ( a02,   a12,   a22,  -g0.z, -gi.z)   (p.z)
	//                           (-g0.x, -g0.y, -g0.z,  1.0,   0.0 )   (s0 )
	//                           (-gi.x, -gi.y, -gi.z,  0.0,   1.0 )   (si )
	// 
	float weighted_error = ComputeQuadricError(q, p);
	
	for (u32 i = 0; i < attribute_stride_dwords; i += 1) {
		auto g = a[i].g;
		auto d = a[i].d;
		if (q.weight <= 0.f) continue;
		
		float s = (DotProduct(g, p) + d) * (1.f / q.weight);
		
		if constexpr (attribute_weight > 0.f) {
			attributes[i] = s * (1.f / attribute_weight);
		}
		
		//
		// Simplified by replacing first three lines with a dot product, and substituting -DotProduct(g, p) for (d - s * q.weight).
		//
		// p.x * (-g.x * s) +
		// p.y * (-g.y * s) +
		// p.z * (-g.z * s) +
		// s * (-DotProduct(g, p) + s) +
		// -2.f * d * s;
		//
		float weighted_attribute_error = s * s * (1.f - 2.f * q.weight);
		
		weighted_error += weighted_attribute_error;
	}
	
	return fabsf(weighted_error);
}
#endif // ENABLE_ATTRIBUTE_SUPPORT


// Check if any triangle around the collapsed edge is flipped or becomes zero area, excluding collapsed triangles.
static bool ValidateEdgeCollapsePosition(MeshView mesh, Edge edge, const Vector3& new_position) {
	bool reject_edge_collapse = false;
	
	auto check_triangle_flip_for_vertex = [&](CornerID corner_id)-> IterationControl {
		auto& c1 = mesh[corner_id];
		
		auto v0 = mesh[c1.corner_list_around[(u32)ElementType::Face].prev].vertex_id;
		auto v1 = c1.vertex_id;
		auto v2 = mesh[c1.corner_list_around[(u32)ElementType::Face].next].vertex_id;
		
		auto p0 = mesh[v0].position;
		auto p1 = mesh[v1].position;
		auto p2 = mesh[v2].position;
		
		auto n0 = CrossProduct(p1 - p0, p2 - p0);
		
		u32 replaced_vertex_count = 0;
		if (v0.index == edge.vertex_0.index || v0.index == edge.vertex_1.index) { replaced_vertex_count += 1; p0 = new_position; }
		if (v1.index == edge.vertex_0.index || v1.index == edge.vertex_1.index) { replaced_vertex_count += 1; p1 = new_position; }
		if (v2.index == edge.vertex_0.index || v2.index == edge.vertex_1.index) { replaced_vertex_count += 1; p2 = new_position; }
		
		// Replaced vertex count == 0 is impossible.
		// Replaced vertex count == 2 is true for triangles that would get collapsed, we don't need to check if they flip.
		if (replaced_vertex_count == 1) {
			auto n1 = CrossProduct(p1 - p0, p2 - p0);
			
			// Prevent flipped or zero area triangles.
			reject_edge_collapse |= (DotProduct(n0, n1) < 0.f) || (DotProduct(n0, n0) > FLT_EPSILON && DotProduct(n1, n1) < FLT_EPSILON);
		}
		
		return reject_edge_collapse ? IterationControl::Break : IterationControl::Continue;
	};
	
	if (reject_edge_collapse == false) IterateCornerList<ElementType::Vertex>(mesh, mesh[edge.vertex_0].corner_list_base, check_triangle_flip_for_vertex);
	if (reject_edge_collapse == false) IterateCornerList<ElementType::Vertex>(mesh, mesh[edge.vertex_1].corner_list_base, check_triangle_flip_for_vertex);
	
	return reject_edge_collapse == false;
}

struct State {
	// Edge quadrics accumulated on vertices.
	std::vector<Quadric> vertex_edge_quadrics;
	
	// Face quadrics accumulated on attributes.
	std::vector<Quadric> attribute_face_quadrics;
	
#if ENABLE_ATTRIBUTE_SUPPORT
	// Face quadrics accumulated on attributes.
	std::vector<QuadricAttributeData> attribute_face_quadrics_a;
#endif // ENABLE_ATTRIBUTE_SUPPORT
	
	std::vector<u8> wedge_attribute_set;
	
	std::vector<Quadric> wedge_quadrics;
	std::vector<AttributesID> wedge_attributes_ids;
	
#if ENABLE_ATTRIBUTE_SUPPORT
	// Face quadrics accumulated on attributes.
	std::vector<QuadricAttributeData> wedge_quadrics_a;
#endif // ENABLE_ATTRIBUTE_SUPPORT
};

struct EdgeSelectInfo {
	EdgeID edge_to_collapse = { u32_max };
	float min_error = FLT_MAX;
	
	Vector3 new_position;
};

void ComputeEdgeCollapseError(MeshView mesh, State& state, EdgeSelectInfo& info, EdgeID edge_id) {
	memset(state.wedge_attribute_set.data(), 0, state.wedge_attribute_set.size());
	state.wedge_quadrics.clear();
	state.wedge_attributes_ids.clear();
#if ENABLE_ATTRIBUTE_SUPPORT
	state.wedge_quadrics_a.clear();
#endif // ENABLE_ATTRIBUTE_SUPPORT
	
	auto& edge = mesh[edge_id];
	
	IterateCornerList<ElementType::Edge>(mesh, edge.corner_list_base, [&](CornerID corner_id) {
		auto& corner_0 = mesh[corner_id];
		auto& corner_1 = mesh[corner_0.corner_list_around[(u32)ElementType::Face].next];
		
		u8 wedge_index_0 = state.wedge_attribute_set[corner_0.attributes_id.index];
		u8 wedge_index_1 = state.wedge_attribute_set[corner_1.attributes_id.index];
		
		if (wedge_index_0 == 0 && wedge_index_1 == 0) {
			u8 index = (u8)(state.wedge_quadrics.size() + 1);
			state.wedge_attribute_set[corner_0.attributes_id.index] = index;
			state.wedge_attribute_set[corner_1.attributes_id.index] = index;
			
			state.wedge_attributes_ids.emplace_back(corner_0.attributes_id);
			auto& quadric = state.wedge_quadrics.emplace_back(state.attribute_face_quadrics[corner_0.attributes_id.index]);
			AccumulateQuadric(quadric, state.attribute_face_quadrics[corner_1.attributes_id.index]);
			
#if ENABLE_ATTRIBUTE_SUPPORT
			QuadricAttributeData quadric_attributes[attribute_stride_dwords] = {};
			memcpy(quadric_attributes, &state.attribute_face_quadrics_a[corner_0.attributes_id.index * attribute_stride_dwords], attribute_stride_dwords * sizeof(QuadricAttributeData));
			AccumulateQuadricAttribute(quadric_attributes, &state.attribute_face_quadrics_a[corner_1.attributes_id.index * attribute_stride_dwords]);
			
			for (auto& a : quadric_attributes) state.wedge_quadrics_a.push_back(a);
#endif // ENABLE_ATTRIBUTE_SUPPORT
		} else if (wedge_index_0 == 0 && wedge_index_1 != 0) {
			state.wedge_attribute_set[corner_0.attributes_id.index] = wedge_index_1;
			AccumulateQuadric(state.wedge_quadrics[wedge_index_1 - 1], state.attribute_face_quadrics[corner_0.attributes_id.index]);
			
#if ENABLE_ATTRIBUTE_SUPPORT
			AccumulateQuadricAttribute(&state.wedge_quadrics_a[(wedge_index_1 - 1) * attribute_stride_dwords], &state.attribute_face_quadrics_a[corner_0.attributes_id.index * attribute_stride_dwords]);
#endif // ENABLE_ATTRIBUTE_SUPPORT
		} else if (wedge_index_0 != 0 && wedge_index_1 == 0) {
			state.wedge_attribute_set[corner_1.attributes_id.index] = wedge_index_0;
			AccumulateQuadric(state.wedge_quadrics[wedge_index_0 - 1], state.attribute_face_quadrics[corner_1.attributes_id.index]);
#if ENABLE_ATTRIBUTE_SUPPORT
			AccumulateQuadricAttribute(&state.wedge_quadrics_a[(wedge_index_0 - 1) * attribute_stride_dwords], &state.attribute_face_quadrics_a[corner_1.attributes_id.index * attribute_stride_dwords]);
#endif // ENABLE_ATTRIBUTE_SUPPORT
		}
	});
	
	auto accumulate_quadrics = [&](CornerID corner_id) {
		u32 attribute_index = mesh[corner_id].attributes_id.index;
		
		if (state.wedge_attribute_set[attribute_index] == 0) {
			state.wedge_attribute_set[attribute_index] = (u8)(state.wedge_quadrics.size() + 1);
			state.wedge_quadrics.push_back(state.attribute_face_quadrics[attribute_index]);
			state.wedge_attributes_ids.emplace_back(AttributesID{ attribute_index });
			
#if ENABLE_ATTRIBUTE_SUPPORT
			QuadricAttributeData quadric_attributes[attribute_stride_dwords] = {};
			memcpy(quadric_attributes, &state.attribute_face_quadrics_a[attribute_index * attribute_stride_dwords], attribute_stride_dwords * sizeof(QuadricAttributeData));
			for (auto& a : quadric_attributes) state.wedge_quadrics_a.push_back(a);
#endif // ENABLE_ATTRIBUTE_SUPPORT
		}
	};
	
	auto& v0 = mesh[edge.vertex_0];
	auto& v1 = mesh[edge.vertex_1];
	
	IterateCornerList<ElementType::Vertex>(mesh, v0.corner_list_base, accumulate_quadrics);
	IterateCornerList<ElementType::Vertex>(mesh, v1.corner_list_base, accumulate_quadrics);
	
	auto edge_quadrics = state.vertex_edge_quadrics[edge.vertex_0.index];
	AccumulateQuadric(edge_quadrics, state.vertex_edge_quadrics[edge.vertex_1.index]);
	
	// Try a few different positions for the new vertex.
	compile_const u32 candidate_position_count = 3;
	Vector3 candidate_positions[candidate_position_count];
	candidate_positions[0] = v0.position;
	candidate_positions[1] = v1.position;
	candidate_positions[2] = (candidate_positions[0] + candidate_positions[1]) * 0.5f;
	
	u32 wedge_count = (u32)state.wedge_quadrics.size();
	for (u32 i = 0; i < candidate_position_count; i += 1) {
		auto& p = candidate_positions[i];
		
		float error = ComputeQuadricError(edge_quadrics, p);
		
		for (u32 i = 0; i < wedge_count; i += 1) {
#if ENABLE_ATTRIBUTE_SUPPORT
			float attributes[attribute_stride_dwords];
			error += ComputeQuadricError(state.wedge_quadrics[i], &state.wedge_quadrics_a[i * attribute_stride_dwords], p, attributes);
#else // !ENABLE_ATTRIBUTE_SUPPORT
			error += ComputeQuadricError(state.wedge_quadrics[i], p);
#endif // ENABLE_ATTRIBUTE_SUPPORT
		}
		
		if (info.min_error > error && ValidateEdgeCollapsePosition(mesh, edge, p)) {
			info.min_error = error;
			info.edge_to_collapse = edge_id;
			info.new_position = p;
		}
	}
}

//
// TODO:
// - Support for attribute quadrics.
// - Optimization of vertex location.
// - Priority queue (don't recompute all quadrics after each collapse).
// - Output an obj sequence for edge to verify edge collapses.
//
void DecimateMesh(MeshView mesh) {
	State state;
	state.vertex_edge_quadrics.resize(mesh.vertex_count);
	state.attribute_face_quadrics.resize(mesh.attribute_count);
	state.wedge_attribute_set.resize(mesh.attribute_count);
	state.wedge_quadrics.reserve(64);

#if ENABLE_ATTRIBUTE_SUPPORT
	state.attribute_face_quadrics_a.resize(mesh.attribute_count * attribute_stride_dwords);
	state.wedge_quadrics_a.reserve(64 * attribute_stride_dwords);
#endif // ENABLE_ATTRIBUTE_SUPPORT
	
	for (u32 i = 0; i < 405; i += 1) {
		memset(state.vertex_edge_quadrics.data(), 0, state.vertex_edge_quadrics.size() * sizeof(Quadric));
		memset(state.attribute_face_quadrics.data(), 0, state.attribute_face_quadrics.size() * sizeof(Quadric));
		
#if ENABLE_ATTRIBUTE_SUPPORT
		memset(state.attribute_face_quadrics_a.data(), 0, state.attribute_face_quadrics_a.size() * sizeof(QuadricAttributeData));
#endif // ENABLE_ATTRIBUTE_SUPPORT
		
		for (FaceID face_id = { 0 }; face_id.index < mesh.face_count; face_id.index += 1) {
			auto& face = mesh[face_id];
			if (face.corner_list_base.index == u32_max) continue;
			
			auto& c1 = mesh[face.corner_list_base];
			auto& c0 = mesh[c1.corner_list_around[(u32)ElementType::Face].prev];
			auto& c2 = mesh[c1.corner_list_around[(u32)ElementType::Face].next];
			
			auto p0 = mesh[c0.vertex_id].position;
			auto p1 = mesh[c1.vertex_id].position;
			auto p2 = mesh[c2.vertex_id].position;
			
#if ENABLE_ATTRIBUTE_SUPPORT
			auto* a0 = mesh[c0.attributes_id];
			auto* a1 = mesh[c1.attributes_id];
			auto* a2 = mesh[c2.attributes_id];
#endif // ENABLE_ATTRIBUTE_SUPPORT
			
			auto e10 = p1 - p0;
			auto e20 = p2 - p0;
			
			auto normal_direction     = CrossProduct(e10, e20);
			auto twice_triangle_area  = Length(normal_direction);
			auto normal               = twice_triangle_area < FLT_EPSILON ? normal_direction : normal_direction * (1.f / twice_triangle_area); // TODO: Potential division by zero. Handle better.
			auto distance_to_triangle = -DotProduct(normal, p0);
			float weight = twice_triangle_area * 0.5f;
			
			auto quadric = ComputeQuadric(normal, distance_to_triangle, weight);
			
#if ENABLE_ATTRIBUTE_SUPPORT
			QuadricAttributeData quadric_attributes[attribute_stride_dwords] = {};
			ComputeAttributeQuadric(quadric, quadric_attributes, p0, p1, p2, a0, a1, a2, weight);
#endif // ENABLE_ATTRIBUTE_SUPPORT
			
			AccumulateQuadric(state.attribute_face_quadrics[c0.attributes_id.index], quadric);
			AccumulateQuadric(state.attribute_face_quadrics[c1.attributes_id.index], quadric);
			AccumulateQuadric(state.attribute_face_quadrics[c2.attributes_id.index], quadric);
			
#if ENABLE_ATTRIBUTE_SUPPORT
			AccumulateQuadricAttribute(&state.attribute_face_quadrics_a[c0.attributes_id.index * attribute_stride_dwords], quadric_attributes);
			AccumulateQuadricAttribute(&state.attribute_face_quadrics_a[c1.attributes_id.index * attribute_stride_dwords], quadric_attributes);
			AccumulateQuadricAttribute(&state.attribute_face_quadrics_a[c2.attributes_id.index * attribute_stride_dwords], quadric_attributes);
#endif // ENABLE_ATTRIBUTE_SUPPORT
		}
		
		for (EdgeID edge_id = { 0 }; edge_id.index < mesh.edge_count; edge_id.index += 1) {
			auto& edge = mesh[edge_id];
			if (edge.corner_list_base.index == u32_max) continue;
			
			u32 degree = 0;
			IterateCornerList<ElementType::Edge>(mesh, edge.corner_list_base, [&](CornerID) {
				degree += 1;
			});
			if (degree != 1) continue;
			
			auto& c1 = mesh[edge.corner_list_base];
			
			// v1---v2 is the edge.
			auto v0 = mesh[c1.corner_list_around[(u32)ElementType::Face].prev].vertex_id;
			auto v1 = c1.vertex_id;
			auto v2 = mesh[c1.corner_list_around[(u32)ElementType::Face].next].vertex_id;
			
			assert(v1.index == edge.vertex_0.index || v1.index == edge.vertex_1.index);
			assert(v2.index == edge.vertex_0.index || v2.index == edge.vertex_1.index);
			
			auto p0 = mesh[v0].position;
			auto p1 = mesh[v1].position;
			auto p2 = mesh[v2].position;
			
			auto e10 = p1 - p0;
			auto e20 = p2 - p0;
			auto e21 = p1 - p2;
			
			auto face_normal_direction = CrossProduct(e10, e20);
			auto normal_direction      = CrossProduct(e21, face_normal_direction);
			auto normal_length         = Length(normal_direction);
			auto normal                = normal_length < FLT_EPSILON ? normal_direction : normal_direction * (1.f / normal_length); // TODO: Potential division by zero. Handle better.
			auto distance_to_triangle  = -DotProduct(normal, p1);
			
			auto quadric = ComputeQuadric(normal, distance_to_triangle, Length(e21));
			
			AccumulateQuadric(state.vertex_edge_quadrics[v1.index], quadric);
			AccumulateQuadric(state.vertex_edge_quadrics[v2.index], quadric);
		}
		
		EdgeSelectInfo select_info;
		
		for (EdgeID edge_id = { 0 }; edge_id.index < mesh.edge_count; edge_id.index += 1) {
			auto& edge = mesh[edge_id];
			if (edge.corner_list_base.index == u32_max) continue;
			
			ComputeEdgeCollapseError(mesh, state, select_info, edge_id);
		}
		
		if (select_info.edge_to_collapse.index != u32_max) {
#if ENABLE_ATTRIBUTE_SUPPORT
			EdgeSelectInfo new_select_info = {};
			ComputeEdgeCollapseError(mesh, state, new_select_info, select_info.edge_to_collapse);
#endif // ENABLE_ATTRIBUTE_SUPPORT
			
			// printf("CollapseID: %u, Error: %.3f\n", edge_to_collapse.index, min_error);
			auto remaning_base_id = PerformEdgeCollapse(mesh, select_info.edge_to_collapse);
			
			auto& edge = mesh[select_info.edge_to_collapse];
			mesh[edge.vertex_0].position = select_info.new_position;
			mesh[edge.vertex_1].position = select_info.new_position;
				
#if ENABLE_ATTRIBUTE_SUPPORT
			if (remaning_base_id.index != u32_max) {
				u32 wedge_count = (u32)state.wedge_quadrics.size();
				
				std::vector<float> attributes;
				attributes.resize(wedge_count * attribute_stride_dwords);
				
				for (u32 i = 0; i < wedge_count; i += 1) {
					ComputeQuadricError(state.wedge_quadrics[i], &state.wedge_quadrics_a[i * attribute_stride_dwords], select_info.new_position, &attributes[i * attribute_stride_dwords]);
				}
				
				IterateCornerList<ElementType::Vertex>(mesh, remaning_base_id, [&](CornerID corner_id) {
					auto attributes_id = mesh[corner_id].attributes_id;
					
					u8 index = state.wedge_attribute_set[attributes_id.index];
					if (index != 0) {
						memcpy(mesh[attributes_id], &attributes[(index - 1) * attribute_stride_dwords], sizeof(u32) * attribute_stride_dwords);
						mesh[corner_id].attributes_id = state.wedge_attributes_ids[index - 1];
					}
				});
			}
#endif // ENABLE_ATTRIBUTE_SUPPORT
		} else {
			break;
		}
	}
}


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

// Iterate linked list around a given element type starting with the base corner id. 
// Removal while iterating is allowed.
template<ElementType element_type_t, typename Lambda>
static void IterateCornerList(MeshView mesh, CornerID corner_list_base, Lambda&& lambda) {
	auto& element = mesh[GetElementID<element_type_t>(mesh[corner_list_base])];
	
	auto current_corner_id = corner_list_base;
	do {
		auto next_corner_id = mesh[current_corner_id].corner_list_around[(u32)element_type_t].next;
		
		lambda(current_corner_id);
		
		current_corner_id = next_corner_id;
	} while (current_corner_id.index != corner_list_base.index && element.corner_list_base.index != u32_max);
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


static void PerformEdgeCollapse(MeshView mesh, EdgeID edge_id) {
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

static Vector3 operator+ (const Vector3& lh, const Vector3& rh) { return Vector3{ lh.x + rh.x, lh.y + rh.y, lh.z + rh.z }; }
static Vector3 operator- (const Vector3& lh, const Vector3& rh) { return Vector3{ lh.x - rh.x, lh.y - rh.y, lh.z - rh.z }; }
static Vector3 operator* (const Vector3& lh, float rh) { return Vector3{ lh.x * rh, lh.y * rh, lh.z * rh }; }
static float DotProduct(const Vector3& lh, const Vector3& rh) { return lh.x * rh.x + lh.y * rh.y + lh.z * rh.z; }
static float Length(const Vector3& v) { return sqrtf(DotProduct(v, v)); }
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

static Quadric ComputeQuadric(const Vector3& normal, float distance_to_triangle, float weight) {
	//
	// For reference see [Hugues Hoppe 1999] Simplification of geometry.
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
	
	return fabsf(weighted_error) * (1.f / q.weight);
}

void DecimateMesh(MeshView mesh) {
	std::vector<Quadric> vertex_quadrics;
	vertex_quadrics.resize(mesh.vertex_count);
	
	for (u32 i = 0; i < 375; i += 1) {
		memset(vertex_quadrics.data(), 0, vertex_quadrics.size() * sizeof(Quadric));
		
		for (FaceID face_id = { 0 }; face_id.index < mesh.face_count; face_id.index += 1) {
			auto& face = mesh[face_id];
			if (face.corner_list_base.index == u32_max) continue;
			
			auto& c1 = mesh[face.corner_list_base];
			
			auto v0 = mesh[c1.corner_list_around[(u32)ElementType::Face].prev].vertex_id;
			auto v1 = c1.vertex_id;
			auto v2 = mesh[c1.corner_list_around[(u32)ElementType::Face].next].vertex_id;
			
			auto p0 = mesh[v0].position;
			auto p1 = mesh[v1].position;
			auto p2 = mesh[v2].position;
			
			auto e10 = p1 - p0;
			auto e20 = p2 - p0;
			
			auto normal_direction     = CrossProduct(e10, e20);
			auto twice_triangle_area  = Length(normal_direction);
			auto normal               = normal_direction * (1.f / twice_triangle_area); // TODO: Potential division by zero.
			auto distance_to_triangle = -DotProduct(normal, p0);
			
			auto quadric = ComputeQuadric(normal, distance_to_triangle, twice_triangle_area * 0.5f);
			
			AccumulateQuadric(vertex_quadrics[v0.index], quadric);
			AccumulateQuadric(vertex_quadrics[v1.index], quadric);
			AccumulateQuadric(vertex_quadrics[v2.index], quadric);
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
			auto normal                = normal_direction * (1.f / Length(normal_direction)); // TODO: Potential division by zero.
			auto distance_to_triangle  = -DotProduct(normal, p1);
			
			auto quadric = ComputeQuadric(normal, distance_to_triangle, Length(e21)/* * 10.f*/);
			
			AccumulateQuadric(vertex_quadrics[v1.index], quadric);
			AccumulateQuadric(vertex_quadrics[v2.index], quadric);
		}
		
		auto edge_to_collapse = EdgeID{ u32_max };
		float min_error = FLT_MAX;
		Vector3 new_position;
		
		for (EdgeID edge_id = { 0 }; edge_id.index < mesh.edge_count; edge_id.index += 1) {
			auto& edge = mesh[edge_id];
			if (edge.corner_list_base.index == u32_max) continue;
			
			auto q = vertex_quadrics[edge.vertex_0.index];
			AccumulateQuadric(q, vertex_quadrics[edge.vertex_1.index]);
			
			// Try a few different positions for the new vertex.
			auto p0 = mesh[edge.vertex_0].position;
			auto p1 = mesh[edge.vertex_1].position;
			auto p2 = (p0 + p1) * 0.5f;
			
			float error0 = ComputeQuadricError(q, p0);
			if (min_error > error0) {
				min_error = error0;
				edge_to_collapse = edge_id;
				new_position = p0;
			}
			
			float error1 = ComputeQuadricError(q, p1);
			if (min_error > error1) {
				min_error = error1;
				edge_to_collapse = edge_id;
				new_position = p1;
			}
			
			float error2 = ComputeQuadricError(q, p2);
			if (min_error > error2) {
				min_error = error2;
				edge_to_collapse = edge_id;
				new_position = p2;
			}
		}
		
		if (edge_to_collapse.index != u32_max) {
			printf("CollapseID: %u, Error: %.3f\n", edge_to_collapse.index, min_error);
			PerformEdgeCollapse(mesh, edge_to_collapse);
			
			auto& edge = mesh[edge_to_collapse];
			mesh[edge.vertex_0].position = new_position;
			mesh[edge.vertex_1].position = new_position;
		} else {
			break;
		}
	}
}


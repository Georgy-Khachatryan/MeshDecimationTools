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
// - Hugues Hoppe, Steve Marschner. 2000. Efficient Minimization of New Quadric Metric for Simplifying Meshes with Appearance Attributes.
//

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
	
	u64 operator() (const Vector3& v) const { return PositionHash(v); }
};
#pragma fenv_access(off)

static u64 PackEdgeKey(VertexID vertex_id_0, VertexID vertex_id_1) {
	// Always pack VertexIDs in ascending order to ensure that PackEdgeKey(A, B) == PackEdgeKey(B, A) and they hash to the same value.
	return vertex_id_1.index > vertex_id_0.index ? ((u64)vertex_id_1.index << u32_bit_count) | (u64)vertex_id_0.index : ((u64)vertex_id_0.index << u32_bit_count) | (u64)vertex_id_1.index;
}

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
					assert(edge.vertex_0.index != edge.vertex_1.index);
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
	std::unordered_map<u64, EdgeID> edge_to_edge_id;
	
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
		
		u64 edge_keys[3] = {
			PackEdgeKey(vertex_ids[0], vertex_ids[1]),
			PackEdgeKey(vertex_ids[1], vertex_ids[2]),
			PackEdgeKey(vertex_ids[2], vertex_ids[0]),
		};

		auto& face = mesh[face_id];
		face.corner_list_base.index = u32_max;
		
		bool has_duplicate_vertices = 
			vertex_ids[0].index == vertex_ids[1].index ||
			vertex_ids[0].index == vertex_ids[2].index ||
			vertex_ids[1].index == vertex_ids[2].index;
		
		// Skip primitives that reduce to a single line or a point. PerformEdgeCollapse(...) can't reliably handle them.
		if (has_duplicate_vertices) continue;
		
		
		for (u32 corner_index = 0; corner_index < 3; corner_index += 1) {
			auto corner_id = CornerID{ triangle_index * 3 + corner_index };
			u64 edge_key = edge_keys[corner_index];
			
			auto [it, is_inserted] = edge_to_edge_id.emplace(edge_key, edge_id_allocator);
			if (is_inserted) {
				auto& edge = result_mesh.edges.emplace_back();
				edge.vertex_0 = VertexID{ (u32)(edge_key >> 0) };
				edge.vertex_1 = VertexID{ (u32)(edge_key >> u32_bit_count) };
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

using RemovedEdgeArray = std::vector<EdgeID>;
using EdgeDuplicateMap = std::unordered_map<u64, EdgeID>;

static CornerID PerformEdgeCollapse(MeshView mesh, EdgeID edge_id, EdgeDuplicateMap& edge_duplicate_map, RemovedEdgeArray& removed_edge_array) {
	auto& edge = mesh[edge_id];
	
	assert(edge.vertex_0.index != edge.vertex_1.index);
	auto& vertex_0 = mesh[edge.vertex_0];
	auto& vertex_1 = mesh[edge.vertex_1];

	assert(edge.corner_list_base.index != u32_max);
	assert(vertex_0.corner_list_base.index != u32_max);
	assert(vertex_1.corner_list_base.index != u32_max);
	assert(vertex_0.corner_list_base.index != vertex_1.corner_list_base.index);
	
	removed_edge_array.clear();
	IterateCornerList<ElementType::Edge>(mesh, edge.corner_list_base, [&](CornerID corner_id) {
		auto& corner = mesh[corner_id];
		auto& face   = mesh[corner.face_id];
		
		IterateCornerList<ElementType::Face>(mesh, face.corner_list_base, [&](CornerID corner_id) {
			CornerListRemove<ElementType::Vertex>(mesh, corner_id);
			bool edge_removed = CornerListRemove<ElementType::Edge>(mesh, corner_id);
			CornerListRemove<ElementType::Face>(mesh, corner_id);
			
			if (edge_removed) removed_edge_array.push_back(mesh[corner_id].edge_id);
		});
	});
	
	CornerListMerge<VertexID>(mesh, edge.vertex_0, edge.vertex_1);
	
	auto remaning_base_id = vertex_0.corner_list_base.index != u32_max ? vertex_0.corner_list_base : vertex_1.corner_list_base;
	if (remaning_base_id.index != u32_max) {
		edge_duplicate_map.clear();
		
		IterateCornerList<ElementType::Vertex>(mesh, remaning_base_id, [&](CornerID corner_id) {
			// TODO: This can be iteration over just incoming and outgoing edges of a corner.
			IterateCornerList<ElementType::Face>(mesh, corner_id, [&](CornerID corner_id) {
				auto edge_id_1 = mesh[corner_id].edge_id;
				auto& edge_1 = mesh[edge_id_1];
				
				auto [it, is_inserted] = edge_duplicate_map.emplace(PackEdgeKey(edge_1.vertex_0, edge_1.vertex_1), edge_id_1);
				auto edge_id_0 = it->second;
				
				if (is_inserted == false && edge_id_0.index != edge_id_1.index) {
					CornerListMerge<EdgeID>(mesh, edge_id_0, edge_id_1);
					removed_edge_array.push_back(edge_id_1);
				}
			});
		});
		
		IterateCornerList<ElementType::Vertex>(mesh, remaning_base_id, [&](CornerID corner_id_0) {
			IterateCornerList<ElementType::Face>(mesh, corner_id_0, [&](CornerID corner_id_1) {
				if (remaning_base_id.index == corner_id_1.index) return;
				
				IterateCornerList<ElementType::Vertex>(mesh, corner_id_1, [&](CornerID corner_id_2) {
					if (corner_id_1.index == corner_id_2.index) return;
					
					IterateCornerList<ElementType::Face>(mesh, corner_id_2, [&](CornerID corner_id) {
						auto edge_id = mesh[corner_id].edge_id;
						auto& edge = mesh[edge_id];
						
						edge_duplicate_map.emplace(PackEdgeKey(edge.vertex_0, edge.vertex_1), edge_id);
					});
				});
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

struct QuadricWithAttributes : Quadric {
#if ENABLE_ATTRIBUTE_SUPPORT
	struct {
		Vector3 g = { 0.f, 0.f, 0.f };
		float d = 0.f;
	} attributes[attribute_stride_dwords];
#endif // ENABLE_ATTRIBUTE_SUPPORT
};


//
// Matt Pharr's blog. 2019. Accurate Differences of Products with Kahan's Algorithm.
//
static float DifferenceOfProducts(float a, float b, float c, float d) {
	float cd = c * d;
	float err = fmaf(c, -d,  cd);
	float dop = fmaf(a,  b, -cd);
	return dop + err;
}

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

static void AccumulateQuadricWithAttributes(QuadricWithAttributes& accumulator, const QuadricWithAttributes& quadric) {
	AccumulateQuadric(accumulator, quadric);
	
#if ENABLE_ATTRIBUTE_SUPPORT
	for (u32 i = 0; i < attribute_stride_dwords; i += 1) {
		auto& attribute_accumulator = accumulator.attributes[i];
		auto& attribute_quadric     = quadric.attributes[i];

		attribute_accumulator.g.x += attribute_quadric.g.x;
		attribute_accumulator.g.y += attribute_quadric.g.y;
		attribute_accumulator.g.z += attribute_quadric.g.z;
		attribute_accumulator.d   += attribute_quadric.d;
	}
#endif // ENABLE_ATTRIBUTE_SUPPORT
}

static void ComputePlanarQuadric(Quadric& quadric, const Vector3& n, float d, float weight) {
	//
	// For reference see [Hugues Hoppe 1999] Section 3 Previous Quadric Error Metrics.
	//
	// A = n * n^T
	// b = d * n
	// c = d^2
	//
	// (n.x)
	// (n.y) * (n.x, n.y, n.z)
	// (n.z)
	//
	quadric.a00 = (n.x * n.x) * weight;
	quadric.a11 = (n.y * n.y) * weight;
	quadric.a22 = (n.z * n.z) * weight;
	quadric.a01 = (n.x * n.y) * weight;
	quadric.a02 = (n.x * n.z) * weight;
	quadric.a12 = (n.y * n.z) * weight;
	
	quadric.b.x = (n.x * d) * weight;
	quadric.b.y = (n.y * d) * weight;
	quadric.b.z = (n.z * d) * weight;
	
	quadric.c = (d * d) * weight;
	quadric.weight = weight;
}

// Assumes that the quadric edge is (p0, p1)
static void ComputeEdgeQuadric(Quadric& quadric, const Vector3& p0, const Vector3& p1, const Vector3& p2) {
	auto p10 = p1 - p0;
	auto p20 = p2 - p0;
	
	auto face_normal_direction = CrossProduct(p10, p20);
	auto normal_direction      = CrossProduct(p10, face_normal_direction);
	auto normal_length         = Length(normal_direction);
	auto normal                = normal_length < FLT_EPSILON ? normal_direction : normal_direction * (1.f / normal_length); // TODO: Potential division by zero. Handle better.
	auto distance_to_triangle  = -DotProduct(normal, p1);
	
	ComputePlanarQuadric(quadric, normal, distance_to_triangle, Length(p10));
}

// TODO: Convert these constants into user editable settings.
compile_const float attribute_weight = 1.f / (1024.f * 1024.f);
compile_const float inversion_error  = 1.f;

static void ComputeFaceQuadricWithAttributes(QuadricWithAttributes& quadric, const Vector3& p0, const Vector3& p1, const Vector3& p2, float* a0, float* a1, float* a2) {
	auto p10 = p1 - p0;
	auto p20 = p2 - p0;
	
	auto scaled_normal       = CrossProduct(p10, p20);
	auto twice_triangle_area = Length(scaled_normal);
	auto n                   = twice_triangle_area < FLT_EPSILON ? scaled_normal : scaled_normal * (1.f / twice_triangle_area); // TODO: Potential division by zero. Handle better.
	
	float weight = twice_triangle_area * 0.5f;
	ComputePlanarQuadric(quadric, n, -DotProduct(n, p0), weight);
	
	
#if ENABLE_ATTRIBUTE_SUPPORT
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
	
	// Compute determinant of a 3x3 matrix A with rows p10, p20, n.
	float det0 = p10.x * DifferenceOfProducts(p20.y, n.z, p20.z, n.y);
	float det1 = p10.y * DifferenceOfProducts(p20.z, n.x, p20.x, n.z);
	float det2 = p10.z * DifferenceOfProducts(p20.x, n.y, p20.y, n.x);
	float determinant = det0 + det1 + det2;
	float determinant_rcp = fabsf(determinant) < FLT_EPSILON ? 0.f : 1.f / determinant;
	
	// Compute first two colums of A^-1.
	float a_inv_00 = DifferenceOfProducts(p20.y, n.z, p20.z, n.y) * determinant_rcp;
	float a_inv_01 = DifferenceOfProducts(p10.z, n.y, p10.y, n.z) * determinant_rcp;
	float a_inv_10 = DifferenceOfProducts(p20.z, n.x, p20.x, n.z) * determinant_rcp;
	float a_inv_11 = DifferenceOfProducts(p10.x, n.z, p10.z, n.x) * determinant_rcp;
	float a_inv_20 = DifferenceOfProducts(p20.x, n.y, p20.y, n.x) * determinant_rcp;
	float a_inv_21 = DifferenceOfProducts(p10.y, n.x, p10.x, n.y) * determinant_rcp;
	
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
		
		quadric.attributes[i].g = g * weight;
		quadric.attributes[i].d = d * weight;
	}
#endif // ENABLE_ATTRIBUTE_SUPPORT
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
	
	return fabsf(weighted_error);
}

static float ComputeQuadricErrorWithAttributes(const QuadricWithAttributes& q, const Vector3& p, float* attributes) {
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
	
#if ENABLE_ATTRIBUTE_SUPPORT
	float rcp_weight = q.weight < FLT_EPSILON ? 0.f : (1.f / q.weight);
	for (u32 i = 0; i < attribute_stride_dwords; i += 1) {
		auto g = q.attributes[i].g;
		auto d = q.attributes[i].d;
		if (q.weight <= 0.f) continue; // TODO: Check if we hit this and how we should handle fallback.
		
		float s = (DotProduct(g, p) + d) * rcp_weight;
		
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
#endif // ENABLE_ATTRIBUTE_SUPPORT
	
	return fabsf(weighted_error);
}

static bool ComputeOptimalVertexPosition(const QuadricWithAttributes& quadric, Vector3& optimal_position) {
	//
	// For reference see [Hugues Hoppe 2000] Section 3 Quadric Metric Minimization.
	//
	// Note that definition of b = (d, d0, di) is negated in [Hugues Hoppe 2000] relative to [Hugues Hoppe 1999].
	// We're using definitions from [Hugues Hoppe 2000] where g*p + d = s.
	//
	
	if (quadric.weight < FLT_EPSILON) return false;
	
	// K = B * B^T
	float k00 = 0.f;
	float k11 = 0.f;
	float k22 = 0.f;
	float k01 = 0.f;
	float k02 = 0.f;
	float k12 = 0.f;
	
	// h = B * b2
	float h0 = 0.f;
	float h1 = 0.f;
	float h2 = 0.f;
	
#if ENABLE_ATTRIBUTE_SUPPORT
	for (u32 i = 0; i < attribute_stride_dwords; i += 1) {
		auto g = quadric.attributes[i].g;
		auto d = quadric.attributes[i].d;
		
		// B * B^T
		k00 += (g.x * g.x);
		k11 += (g.y * g.y);
		k22 += (g.z * g.z);
		k01 += (g.x * g.y);
		k02 += (g.x * g.z);
		k12 += (g.y * g.z);
		
		// B * b2
		h0 += ((float)g.x * (float)d);
		h1 += ((float)g.y * (float)d);
		h2 += ((float)g.z * (float)d);
	}
#endif // ENABLE_ATTRIBUTE_SUPPORT
	
	// M = C - B * B^T * (1.0 / alpha)
	float rcp_weight = 1.f / quadric.weight;
	float m00 = quadric.a00 - k00 * rcp_weight;
	float m11 = quadric.a11 - k11 * rcp_weight;
	float m22 = quadric.a22 - k22 * rcp_weight;
	float m01 = quadric.a01 - k01 * rcp_weight;
	float m02 = quadric.a02 - k02 * rcp_weight;
	float m12 = quadric.a12 - k12 * rcp_weight;
	
	// Note that this expression is negated relative to [Hugues Hoppe 2000] because our b is using notation
	// from [Hugues Hoppe 1999] where it's negated. So both h and quadric.b are negated.
	float j0 = (h0 * rcp_weight - quadric.b.x);
	float j1 = (h1 * rcp_weight - quadric.b.y);
	float j2 = (h2 * rcp_weight - quadric.b.z);
	
	// Determinant of M.
	float det0 = m00 * DifferenceOfProducts(m11, m22, m12, m12);
	float det1 = m01 * DifferenceOfProducts(m01, m22, m12, m02);
	float det2 = m02 * DifferenceOfProducts(m01, m12, m11, m02);
	float determinant = det0 - det1 + det2;
	
	//
	// As an alternative to inverting M [HSUEH-TI DEREK LIU 2024] suggests Cholesky decomposition.
	// LU decomposition should work too. SVD could be used to find least squares solution.
	//
	if (fabsf(determinant) < FLT_EPSILON) return false;
	float determinant_rcp = 1.f / determinant;
	
	//
	// M inverse. Has to be computed with extra precision, otherwise vertex placement is all over the place.
	//
	float m_inv_00 = DifferenceOfProducts(m11, m22, m12, m12);
	float m_inv_01 = DifferenceOfProducts(m02, m12, m01, m22);
	float m_inv_02 = DifferenceOfProducts(m01, m12, m02, m11);
	float m_inv_11 = DifferenceOfProducts(m00, m22, m02, m02);
	float m_inv_12 = DifferenceOfProducts(m02, m01, m00, m12);
	float m_inv_22 = DifferenceOfProducts(m00, m11, m01, m01);
	
	optimal_position.x = (m_inv_00 * j0 + m_inv_01 * j1 + m_inv_02 * j2) * determinant_rcp;
	optimal_position.y = (m_inv_01 * j0 + m_inv_11 * j1 + m_inv_12 * j2) * determinant_rcp;
	optimal_position.z = (m_inv_02 * j0 + m_inv_12 * j1 + m_inv_22 * j2) * determinant_rcp;
	
	return true;
}

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


struct AttributeWedgeMap {
	compile_const u32 capacity = 256;
	
	AttributesID keys[capacity];
	u8 values[capacity];
	u32 count = 0;
};

static void AttributeWedgeMapAdd(AttributeWedgeMap& small_set, AttributesID key, u8 value) {
	u32 index = (small_set.count < AttributeWedgeMap::capacity) ? small_set.count++ : (AttributeWedgeMap::capacity - 1);
	small_set.keys[index]   = key;
	small_set.values[index] = value;
}

static u8 AttributeWedgeMapFind(AttributeWedgeMap& small_set, AttributesID key) {
	for (u32 i = 0; i < small_set.count; i += 1) {
		if (small_set.keys[i].index == key.index) return small_set.values[i];
	}
	
	return 0;
}

static void ResetAttributeWedgeMap(AttributeWedgeMap& small_set) {
	small_set.count = 0;
}


struct MeshDecimationState {
	// Edge quadrics accumulated on vertices.
	std::vector<Quadric> vertex_edge_quadrics;
	
	// Face quadrics accumulated on attributes.
	std::vector<QuadricWithAttributes> attribute_face_quadrics;
	
	EdgeDuplicateMap edge_duplicate_map;
	RemovedEdgeArray removed_edge_array;
	
	AttributeWedgeMap wedge_attribute_set;
	std::vector<QuadricWithAttributes> wedge_quadrics;
	std::vector<float> wedge_attributes;
	std::vector<AttributesID> wedge_attributes_ids;
};

struct EdgeSelectInfo {
	float min_error = FLT_MAX;
	Vector3 new_position;
};

void ComputeEdgeCollapseError(MeshView mesh, MeshDecimationState& state, EdgeID edge_id, EdgeSelectInfo* info = nullptr) {
	ResetAttributeWedgeMap(state.wedge_attribute_set);
	state.wedge_quadrics.clear();
	state.wedge_attributes_ids.clear();
	
	auto& edge = mesh[edge_id];
	
	// Wedges spanning collapsed edge must be unified. Manually set their wedge index to the same value and accumulate quadrics.
	// For reference see [Hugues Hoppe 1999] Section 5 Attribute Discontinuities, Figure 5.
	IterateCornerList<ElementType::Edge>(mesh, edge.corner_list_base, [&](CornerID corner_id) {
		auto& corner_0 = mesh[corner_id];
		auto& corner_1 = mesh[corner_0.corner_list_around[(u32)ElementType::Face].next];
		
		u8 wedge_index_0 = AttributeWedgeMapFind(state.wedge_attribute_set, corner_0.attributes_id);
		u8 wedge_index_1 = AttributeWedgeMapFind(state.wedge_attribute_set, corner_1.attributes_id);
		
		if (wedge_index_0 == 0 && wedge_index_1 == 0) {
			u8 index = (u8)(state.wedge_quadrics.size() + 1);
			AttributeWedgeMapAdd(state.wedge_attribute_set, corner_0.attributes_id, index);
			AttributeWedgeMapAdd(state.wedge_attribute_set, corner_1.attributes_id, index);
			
			state.wedge_attributes_ids.emplace_back(corner_0.attributes_id);
			auto& quadric = state.wedge_quadrics.emplace_back(state.attribute_face_quadrics[corner_0.attributes_id.index]);
			AccumulateQuadricWithAttributes(quadric, state.attribute_face_quadrics[corner_1.attributes_id.index]);
		} else if (wedge_index_0 == 0 && wedge_index_1 != 0) {
			AttributeWedgeMapAdd(state.wedge_attribute_set, corner_0.attributes_id, wedge_index_1);
			AccumulateQuadricWithAttributes(state.wedge_quadrics[wedge_index_1 - 1], state.attribute_face_quadrics[corner_0.attributes_id.index]);
		} else if (wedge_index_0 != 0 && wedge_index_1 == 0) {
			AttributeWedgeMapAdd(state.wedge_attribute_set, corner_1.attributes_id, wedge_index_0);
			AccumulateQuadricWithAttributes(state.wedge_quadrics[wedge_index_0 - 1], state.attribute_face_quadrics[corner_1.attributes_id.index]);
		}
	});
	
	auto accumulate_quadrics = [&](CornerID corner_id) {
		auto attribute_id = mesh[corner_id].attributes_id;
		
		if (AttributeWedgeMapFind(state.wedge_attribute_set, attribute_id) == 0) {
			AttributeWedgeMapAdd(state.wedge_attribute_set, attribute_id, (u8)(state.wedge_quadrics.size() + 1));
			state.wedge_quadrics.push_back(state.attribute_face_quadrics[attribute_id.index]);
			state.wedge_attributes_ids.emplace_back(attribute_id);
		}
	};
	
	auto& v0 = mesh[edge.vertex_0];
	auto& v1 = mesh[edge.vertex_1];
	
	IterateCornerList<ElementType::Vertex>(mesh, v0.corner_list_base, accumulate_quadrics);
	IterateCornerList<ElementType::Vertex>(mesh, v1.corner_list_base, accumulate_quadrics);
	
	if (info != nullptr) {
		auto edge_quadrics = state.vertex_edge_quadrics[edge.vertex_0.index];
		AccumulateQuadric(edge_quadrics, state.vertex_edge_quadrics[edge.vertex_1.index]);
		
		// Try a few different positions for the new vertex.
		compile_const u32 candidate_position_count = 3;
		Vector3 candidate_positions[candidate_position_count];
		candidate_positions[0] = v0.position;
		candidate_positions[1] = v1.position;
		candidate_positions[2] = (candidate_positions[0] + candidate_positions[1]) * 0.5f;
		
		QuadricWithAttributes total_quadric;
		AccumulateQuadric(total_quadric, edge_quadrics);
		
		u32 wedge_count = (u32)state.wedge_quadrics.size();
		for (u32 i = 0; i < wedge_count; i += 1) {
			 AccumulateQuadricWithAttributes(total_quadric, state.wedge_quadrics[i]);
		}
		
		// Override average with optimal position if it can be computed.
		ComputeOptimalVertexPosition(total_quadric, candidate_positions[2]);
		
		for (u32 i = 0; i < candidate_position_count; i += 1) {
			auto& p = candidate_positions[i];
			
			float error = ComputeQuadricError(edge_quadrics, p);
			
			for (u32 i = 0; i < wedge_count; i += 1) {
				float attributes[attribute_stride_dwords];
				error += ComputeQuadricErrorWithAttributes(state.wedge_quadrics[i], p, attributes);
			}
			if (error > info->min_error) continue;
			
			
			error += ValidateEdgeCollapsePosition(mesh, edge, p) ? 0.f : inversion_error;
			if (error > info->min_error) continue;

			info->min_error = error;
			info->new_position = p;
		}
	}
}

struct EdgeCollapseHeap {
	std::vector<EdgeID> heap_index_to_edge_id;
	std::vector<u32>    edge_id_to_heap_index;
	std::vector<float>  edge_collapse_errors;
};

static u32 HeapChildIndex0(u32 node_index) { return node_index * 2 + 1; }
static u32 HeapChildIndex1(u32 node_index) { return node_index * 2 + 2; }
static u32 HeapParentIndex(u32 node_index) { return (node_index - 1) / 2; }

static void EdgeCollapseHeapSwapElements(EdgeCollapseHeap& heap, u32 node_index_0, u32 node_index_1) {
	auto node_0_edge_id = heap.heap_index_to_edge_id[node_index_0];
	auto node_1_edge_id = heap.heap_index_to_edge_id[node_index_1];
	auto node_0_error   = heap.edge_collapse_errors[node_index_0];
	auto node_1_error   = heap.edge_collapse_errors[node_index_1];
	
	heap.heap_index_to_edge_id[node_index_0] = node_1_edge_id;
	heap.edge_collapse_errors[node_index_0]  = node_1_error;
	
	heap.heap_index_to_edge_id[node_index_1] = node_0_edge_id;
	heap.edge_collapse_errors[node_index_1]  = node_0_error;
	
	heap.edge_id_to_heap_index[node_1_edge_id.index] = node_index_0;
	heap.edge_id_to_heap_index[node_0_edge_id.index] = node_index_1;
}

static void EdgeCollapseHeapSiftUp(EdgeCollapseHeap& heap, u32 node_index) {
	while (node_index) {
		u32 parent_node_index = HeapParentIndex(node_index);
		
		if (heap.edge_collapse_errors[node_index] < heap.edge_collapse_errors[parent_node_index]) {
			EdgeCollapseHeapSwapElements(heap, node_index, parent_node_index);
		}
		
		node_index = parent_node_index;
	}
}

static void EdgeCollapseHeapSiftDown(EdgeCollapseHeap& heap, u32 node_index) {
	u32 element_count = (u32)heap.edge_collapse_errors.size();
	while (HeapChildIndex0(node_index) < element_count) {
		u32 index_0 = HeapChildIndex0(node_index);
		u32 index_1 = HeapChildIndex1(node_index);
		
		u32 smallest_child_index = index_1 >= element_count ? index_0 : (heap.edge_collapse_errors[index_0] < heap.edge_collapse_errors[index_1] ? index_0 : index_1);
		
		if (heap.edge_collapse_errors[node_index] > heap.edge_collapse_errors[smallest_child_index]) {
			EdgeCollapseHeapSwapElements(heap, node_index, smallest_child_index);
		}
		
		node_index = smallest_child_index;
	}
}

static EdgeID EdgeCollapseHeapPop(EdgeCollapseHeap& heap) {
	auto edge_id = heap.heap_index_to_edge_id.front();
	
	if (heap.edge_collapse_errors.size() > 1) {
		heap.edge_collapse_errors[0]  = heap.edge_collapse_errors.back();
		heap.heap_index_to_edge_id[0] = heap.heap_index_to_edge_id.back();
	
		heap.edge_id_to_heap_index[edge_id.index] = u32_max;
		heap.edge_id_to_heap_index[heap.heap_index_to_edge_id[0].index] = 0;
	}

	heap.edge_collapse_errors.pop_back();
	heap.heap_index_to_edge_id.pop_back();
	
	EdgeCollapseHeapSiftDown(heap, 0);
	
	return edge_id;
}

static void EdgeCollapseHeapRemove(EdgeCollapseHeap& heap, u32 heap_index) {
	auto edge_id = heap.heap_index_to_edge_id[heap_index];
	
	bool sift_up = true;
	if (heap.edge_collapse_errors.size() > heap_index) {
		float prev_error = heap.edge_collapse_errors[heap_index];

		heap.edge_collapse_errors[heap_index]  = heap.edge_collapse_errors.back();
		heap.heap_index_to_edge_id[heap_index] = heap.heap_index_to_edge_id.back();

		float new_error = heap.edge_collapse_errors[heap_index];
		sift_up = new_error < prev_error;
	
		heap.edge_id_to_heap_index[edge_id.index] = u32_max;
		heap.edge_id_to_heap_index[heap.heap_index_to_edge_id[heap_index].index] = heap_index;
	}

	heap.edge_collapse_errors.pop_back();
	heap.heap_index_to_edge_id.pop_back();
	
	if (sift_up) {
		EdgeCollapseHeapSiftUp(heap, heap_index);
	} else {
		EdgeCollapseHeapSiftDown(heap, heap_index);
	}
}

static void EdgeCollapseHeapUpdate(EdgeCollapseHeap& heap, u32 node_index, float error) {
	bool sift_up = error < heap.edge_collapse_errors[node_index];
	
	heap.edge_collapse_errors[node_index] = error;
	
	if (sift_up) {
		EdgeCollapseHeapSiftUp(heap, node_index);
	} else {
		EdgeCollapseHeapSiftDown(heap, node_index);
	}
}

static void EdgeCollapseHeapInitialize(EdgeCollapseHeap& heap) {
	if (heap.edge_collapse_errors.size() == 0) return;
	
	u32 node_index = HeapParentIndex((u32)heap.edge_collapse_errors.size() - 1);
	
	for (u32 i = node_index; i > 0; i -= 1) {
		EdgeCollapseHeapSiftDown(heap, i);
	}
	EdgeCollapseHeapSiftDown(heap, 0);
}

#define ENABLE_EDGE_COLLAPSE_HEAP_VALIDATION 0
#if ENABLE_EDGE_COLLAPSE_HEAP_VALIDATION
static void EdgeCollapseHeapValidate(EdgeCollapseHeap& heap) {
	for (u32 i = 0; i < heap.edge_collapse_errors.size(); i += 1) {
		float e0 = heap.edge_collapse_errors[i];
		float e1 = HeapChildIndex0(i) < heap.edge_collapse_errors.size() ? heap.edge_collapse_errors[HeapChildIndex0(i)] : FLT_MAX;
		float e2 = HeapChildIndex1(i) < heap.edge_collapse_errors.size() ? heap.edge_collapse_errors[HeapChildIndex1(i)] : FLT_MAX;

		assert(e0 <= e1);
		assert(e0 <= e2);

		assert(heap.edge_id_to_heap_index[heap.heap_index_to_edge_id[i].index] == i);
	}
}
#endif // ENABLE_EDGE_COLLAPSE_HEAP_VALIDATION

//
// TODO:
// - Optimization of vertex location.
// - Scale mesh and attributes before simplification.
// - Memory-less quadrics.
// - Output an obj sequence for edge to verify edge collapses.
//
void DecimateMesh(MeshView mesh) {
	MeshDecimationState state;
	state.vertex_edge_quadrics.resize(mesh.vertex_count);
	state.attribute_face_quadrics.resize(mesh.attribute_count);
	state.wedge_quadrics.reserve(64);
	state.wedge_attributes.reserve(64 * attribute_stride_dwords);
	state.wedge_attributes_ids.reserve(64 * attribute_stride_dwords);
	
	memset(state.vertex_edge_quadrics.data(), 0, state.vertex_edge_quadrics.size() * sizeof(Quadric));
	memset(state.attribute_face_quadrics.data(), 0, state.attribute_face_quadrics.size() * sizeof(QuadricWithAttributes));
	
	{
		// ScopedTimer t("- Compute Face Quadrics");
		
		for (FaceID face_id = { 0 }; face_id.index < mesh.face_count; face_id.index += 1) {
			auto& face = mesh[face_id];
			if (face.corner_list_base.index == u32_max) continue;
			
			auto& c1 = mesh[face.corner_list_base];
			auto& c0 = mesh[c1.corner_list_around[(u32)ElementType::Face].prev];
			auto& c2 = mesh[c1.corner_list_around[(u32)ElementType::Face].next];
			
			auto p0 = mesh[c0.vertex_id].position;
			auto p1 = mesh[c1.vertex_id].position;
			auto p2 = mesh[c2.vertex_id].position;
			
			auto* a0 = mesh[c0.attributes_id];
			auto* a1 = mesh[c1.attributes_id];
			auto* a2 = mesh[c2.attributes_id];
			
			QuadricWithAttributes quadric;
			ComputeFaceQuadricWithAttributes(quadric, p0, p1, p2, a0, a1, a2);
			
			AccumulateQuadricWithAttributes(state.attribute_face_quadrics[c0.attributes_id.index], quadric);
			AccumulateQuadricWithAttributes(state.attribute_face_quadrics[c1.attributes_id.index], quadric);
			AccumulateQuadricWithAttributes(state.attribute_face_quadrics[c2.attributes_id.index], quadric);
		}
	}
	
	{
		// ScopedTimer t("- Compute Edge Quadrics");
		
		for (EdgeID edge_id = { 0 }; edge_id.index < mesh.edge_count; edge_id.index += 1) {
			auto& edge = mesh[edge_id];
			if (edge.corner_list_base.index == u32_max) continue;
			
			u32 degree = 0;
			IterateCornerList<ElementType::Edge>(mesh, edge.corner_list_base, [&](CornerID) {
				degree += 1;
			});
			if (degree != 1) continue;
			
			auto& c1 = mesh[edge.corner_list_base];
			
			// (v0, v1) is the current edge.
			auto v0 = c1.vertex_id;
			auto v1 = mesh[c1.corner_list_around[(u32)ElementType::Face].next].vertex_id;
			auto v2 = mesh[c1.corner_list_around[(u32)ElementType::Face].prev].vertex_id;
			
			auto p0 = mesh[v0].position;
			auto p1 = mesh[v1].position;
			auto p2 = mesh[v2].position;
			
			Quadric quadric;
			ComputeEdgeQuadric(quadric, p0, p1, p2);
			
			AccumulateQuadric(state.vertex_edge_quadrics[v0.index], quadric);
			AccumulateQuadric(state.vertex_edge_quadrics[v1.index], quadric);
		}
	}
	
	
	EdgeCollapseHeap edge_collapse_heap;
	edge_collapse_heap.edge_collapse_errors.resize(mesh.edge_count);
	edge_collapse_heap.edge_id_to_heap_index.resize(mesh.edge_count);
	edge_collapse_heap.heap_index_to_edge_id.resize(mesh.edge_count);
	
	{
		// ScopedTimer t("- Rank Edge Collapses");
		
		for (EdgeID edge_id = { 0 }; edge_id.index < mesh.edge_count; edge_id.index += 1) {
			EdgeSelectInfo select_info;
			ComputeEdgeCollapseError(mesh, state, edge_id, &select_info);
			
			edge_collapse_heap.edge_collapse_errors[edge_id.index]  = select_info.min_error;
			edge_collapse_heap.edge_id_to_heap_index[edge_id.index] = edge_id.index;
			edge_collapse_heap.heap_index_to_edge_id[edge_id.index] = edge_id;
		}
		
		EdgeCollapseHeapInitialize(edge_collapse_heap);
	}
	
	u32 edges_collapsed = 0;
	u32 target_edges_collapsed = mesh.edge_count * 21 / 64;
	
	u64 time0 = 0;
	u64 time1 = 0;
	
	float max_error = 0.f;
	while (edges_collapsed < target_edges_collapsed && edge_collapse_heap.edge_collapse_errors.size()) {
		if ((target_edges_collapsed - edges_collapsed) % 10000 == 0) {
			printf("Remaining: %u\n", target_edges_collapsed - edges_collapsed);
			
#if ENABLE_EDGE_COLLAPSE_HEAP_VALIDATION
			EdgeCollapseHeapValidate(edge_collapse_heap);
#endif // ENABLE_EDGE_COLLAPSE_HEAP_VALIDATION
		}
		
		u64 t0 = __rdtsc();
		
		// ~80% of the execution time.
		for (auto& [edge_key, edge_id] : state.edge_duplicate_map) {
			u32 heap_index = edge_collapse_heap.edge_id_to_heap_index[edge_id.index];
			if (heap_index == u32_max) continue;
			
			EdgeSelectInfo select_info;
			ComputeEdgeCollapseError(mesh, state, edge_id, &select_info);
			
			EdgeCollapseHeapUpdate(edge_collapse_heap, heap_index, select_info.min_error);
		}
		state.edge_duplicate_map.clear();
			
		u64 t1 = __rdtsc();
		time0 += (t1 - t0);
		
		auto edge_id = EdgeCollapseHeapPop(edge_collapse_heap);
		
		// 2% of the execution time
		EdgeSelectInfo select_info;
		ComputeEdgeCollapseError(mesh, state, edge_id, &select_info);
		
		max_error = max_error < select_info.min_error ? select_info.min_error : max_error;
		
		// 15% of the execution time
		auto remaning_base_id = PerformEdgeCollapse(mesh, edge_id, state.edge_duplicate_map, state.removed_edge_array);
		edges_collapsed += 1;
		
		u32 edge_duplicate_map_size = (u32)state.edge_duplicate_map.size();
		if (edge_duplicate_map_size > 255) edge_duplicate_map_size = 255;
		
		for (auto edge_id : state.removed_edge_array) {
			u32 heap_index = edge_collapse_heap.edge_id_to_heap_index[edge_id.index];
			if (heap_index != u32_max) EdgeCollapseHeapRemove(edge_collapse_heap, heap_index);
		}
		
		{
			auto& edge = mesh[edge_id];
			mesh[edge.vertex_0].position = select_info.new_position;
			mesh[edge.vertex_1].position = select_info.new_position;
			
			Quadric quadric = state.vertex_edge_quadrics[edge.vertex_0.index];
			AccumulateQuadric(quadric, state.vertex_edge_quadrics[edge.vertex_1.index]);
			
			state.vertex_edge_quadrics[edge.vertex_0.index] = quadric;
			state.vertex_edge_quadrics[edge.vertex_1.index] = quadric;
		
#if ENABLE_ATTRIBUTE_SUPPORT
			if (remaning_base_id.index != u32_max) {
				u32 wedge_count = (u32)state.wedge_quadrics.size();
				state.wedge_attributes.resize(wedge_count * attribute_stride_dwords);
				
				for (u32 i = 0; i < wedge_count; i += 1) {
					ComputeQuadricErrorWithAttributes(state.wedge_quadrics[i], select_info.new_position, &state.wedge_attributes[i * attribute_stride_dwords]);
					state.attribute_face_quadrics[state.wedge_attributes_ids[i].index] = state.wedge_quadrics[i];
				}
				
				IterateCornerList<ElementType::Vertex>(mesh, remaning_base_id, [&](CornerID corner_id) {
					auto attributes_id = mesh[corner_id].attributes_id;
					
					u8 index = AttributeWedgeMapFind(state.wedge_attribute_set, attributes_id);
					if (index != 0) {
						memcpy(mesh[attributes_id], &state.wedge_attributes[(index - 1) * attribute_stride_dwords], sizeof(u32) * attribute_stride_dwords);
						mesh[corner_id].attributes_id = state.wedge_attributes_ids[index - 1];
					}
				});
			}
#endif // ENABLE_ATTRIBUTE_SUPPORT
		}
		
		u64 t2 = __rdtsc();
		
		time1 += (t2 - t0);
	}
	
	printf("%.3f\n", (float)time0 / (float)time1);
	printf("Max Error: %e\n", max_error);
}


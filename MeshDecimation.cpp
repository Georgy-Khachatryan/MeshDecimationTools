#include "MeshDecimation.h"

#include <unordered_map>
#include <vector>
#include <intrin.h>
#include <assert.h>

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

void PerformRandomEdgeCollapse(MeshView mesh) {
#if 0
	auto edge_id = EdgeID{ 1 };
	
	printf("CollapseID: %u\n", edge_id.index);
	PerformEdgeCollapse(mesh, edge_id);

	edge_id = EdgeID{ 5 };
	printf("CollapseID: %u\n", edge_id.index);
	PerformEdgeCollapse(mesh, edge_id);

	int k = 2;
#else
	u32 to_collapse = 1800;
	if (to_collapse >= mesh.edge_count) to_collapse = mesh.edge_count;
	for (u32 i = 0; i < to_collapse; i += 1) {
		auto edge_id = EdgeID{ i };
		if (mesh[edge_id].corner_list_base.index == u32_max) continue;
		
		printf("CollapseID: %u\n", edge_id.index);
		PerformEdgeCollapse(mesh, edge_id);
	}
#endif
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


#include "MeshDecimation.h"

#include <unordered_map>
#include <vector>
#include <intrin.h>
#include <assert.h>

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
	
	u64 operator() (const Edge& e) const { return std::hash<u64>{}(e.vertex_0.index) + std::hash<u64>{}(e.vertex_1.index); }
	u64 operator() (const Vector3& v) const { return PositionHash(v); }
};
#pragma fenv_access(off)

static void VertexCornerListInsert(MeshView mesh, VertexID vertex_id, CornerID new_corner_id) {
	auto& corner = mesh[new_corner_id];
	auto& vertex = mesh[vertex_id];
	
	if (vertex.vertex_corner_list_base.index == u32_max) {
		vertex.vertex_corner_list_base = new_corner_id;
		corner.prev_corner_around_a_vertex = new_corner_id;
		corner.next_corner_around_a_vertex = new_corner_id;
	} else {
		auto& existing_corner = mesh[vertex.vertex_corner_list_base];
		mesh[existing_corner.prev_corner_around_a_vertex].next_corner_around_a_vertex = new_corner_id;
		
		corner.prev_corner_around_a_vertex = existing_corner.prev_corner_around_a_vertex;
		corner.next_corner_around_a_vertex = vertex.vertex_corner_list_base;
		existing_corner.prev_corner_around_a_vertex = new_corner_id;
	}
}

static void EdgeCornerListInsert(MeshView mesh, EdgeID edge_id, CornerID new_corner_id) {
	auto& corner = mesh[new_corner_id];
	auto& edge = mesh[edge_id];
	
	if (edge.edge_corner_list_base.index == u32_max) {
		edge.edge_corner_list_base = new_corner_id;
		corner.prev_corner_around_an_edge = new_corner_id;
		corner.next_corner_around_an_edge = new_corner_id;
	} else {
		auto& existing_corner = mesh[edge.edge_corner_list_base];
		mesh[existing_corner.prev_corner_around_an_edge].next_corner_around_an_edge = new_corner_id;
		
		corner.prev_corner_around_an_edge = existing_corner.prev_corner_around_an_edge;
		corner.next_corner_around_an_edge = edge.edge_corner_list_base;
		existing_corner.prev_corner_around_an_edge = new_corner_id;
	}
}

static void FaceCornerListInsert(MeshView mesh, FaceID face_id, CornerID new_corner_id) {
	auto& corner = mesh[new_corner_id];
	auto& face = mesh[face_id];
	
	if (face.face_corner_list_base.index == u32_max) {
		face.face_corner_list_base = new_corner_id;
		corner.prev_corner_around_a_face = new_corner_id;
		corner.next_corner_around_a_face = new_corner_id;
	} else {
		auto& existing_corner = mesh[face.face_corner_list_base];
		mesh[existing_corner.prev_corner_around_a_face].next_corner_around_a_face = new_corner_id;
		
		corner.prev_corner_around_a_face = existing_corner.prev_corner_around_a_face;
		corner.next_corner_around_a_face = face.face_corner_list_base;
		existing_corner.prev_corner_around_a_face = new_corner_id;
	}
}


bool EdgeCornerListRemove(MeshView mesh, CornerID corner_id) {
	auto& corner = mesh[corner_id];
	auto prev_corner_around_an_edge = corner.prev_corner_around_an_edge;
	auto next_corner_around_an_edge = corner.next_corner_around_an_edge;
	
	mesh[next_corner_around_an_edge].prev_corner_around_an_edge = prev_corner_around_an_edge;
	mesh[prev_corner_around_an_edge].next_corner_around_an_edge = next_corner_around_an_edge;
	
	// Remove the edge if it lost it's last corner.
	bool is_last_reference = (prev_corner_around_an_edge.index == corner_id.index);
	auto new_edge_corner_list_base = is_last_reference ? CornerID{ u32_max } : prev_corner_around_an_edge;
	
	mesh[corner.edge_id].edge_corner_list_base = new_edge_corner_list_base;
	
	corner.prev_corner_around_an_edge.index = u32_max;
	corner.next_corner_around_an_edge.index = u32_max;
	
	return is_last_reference;
}

bool FaceCornerListRemove(MeshView mesh, CornerID corner_id) {
	auto& corner = mesh[corner_id];
	auto prev_corner_around_a_face = corner.prev_corner_around_a_face;
	auto next_corner_around_a_face = corner.next_corner_around_a_face;
	
	mesh[next_corner_around_a_face].prev_corner_around_a_face = prev_corner_around_a_face;
	mesh[prev_corner_around_a_face].next_corner_around_a_face = next_corner_around_a_face;
	
	// Remove the face if it lost it's last corner.
	bool is_last_reference = (prev_corner_around_a_face.index == corner_id.index);
	auto new_face_corner_list_base = is_last_reference ? CornerID{ u32_max } : prev_corner_around_a_face;
	
	mesh[corner.face_id].face_corner_list_base = new_face_corner_list_base;
	
	corner.prev_corner_around_a_face.index = u32_max;
	corner.next_corner_around_a_face.index = u32_max;
	
	return is_last_reference;
}

bool VertexCornerListRemove(MeshView mesh, CornerID corner_id) {
	auto& corner = mesh[corner_id];
	auto prev_corner_around_a_vertex = corner.prev_corner_around_a_vertex;
	auto next_corner_around_a_vertex = corner.next_corner_around_a_vertex;

	mesh[next_corner_around_a_vertex].prev_corner_around_a_vertex = prev_corner_around_a_vertex;
	mesh[prev_corner_around_a_vertex].next_corner_around_a_vertex = next_corner_around_a_vertex;
	
	// Remove the vertex if it lost it's last corner.
	bool is_last_reference = (prev_corner_around_a_vertex.index == corner_id.index);
	auto new_vertex_corner_list_base = is_last_reference ? CornerID{ u32_max } : prev_corner_around_a_vertex;
	
	mesh[corner.vertex_id].vertex_corner_list_base = new_vertex_corner_list_base;
	
	corner.prev_corner_around_a_vertex.index = u32_max;
	corner.next_corner_around_a_vertex.index = u32_max;
	
	return is_last_reference;
}


template<typename Lambda>
static void IterateEdgeCornerList(MeshView mesh, CornerID edge_corner_list_base_id, Lambda&& lambda) {
	auto current_corner_id = edge_corner_list_base_id;
	auto& edge = mesh[mesh[edge_corner_list_base_id].edge_id];
	bool edge_is_valid = true;
	do {
		auto next_corner_around_an_edge = mesh[current_corner_id].next_corner_around_an_edge;
		lambda(current_corner_id);
		edge_is_valid = edge.edge_corner_list_base.index != u32_max;
		
		current_corner_id = next_corner_around_an_edge;
	} while (current_corner_id.index != edge_corner_list_base_id.index && edge_is_valid);
}

template<typename Lambda>
static void IterateVertexCornerList(MeshView mesh, CornerID vertex_corner_list_base_id, Lambda&& lambda) {
	auto current_corner_id = vertex_corner_list_base_id;
	auto& vertex = mesh[mesh[vertex_corner_list_base_id].vertex_id];
	bool vertex_is_valid = true;
	do {
		auto next_corner_around_a_vertex = mesh[current_corner_id].next_corner_around_a_vertex;
		lambda(current_corner_id);
		vertex_is_valid = vertex.vertex_corner_list_base.index != u32_max;
		
		current_corner_id = next_corner_around_a_vertex;
	} while (current_corner_id.index != vertex_corner_list_base_id.index && vertex_is_valid);
}

template<typename Lambda>
static void IterateFaceCornerList(MeshView mesh, CornerID face_corner_list_base_id, Lambda&& lambda) {
	auto current_corner_id = face_corner_list_base_id;
	auto& face = mesh[mesh[face_corner_list_base_id].face_id];
	bool face_is_valid = true;
	do {
		auto next_corner_around_a_face = mesh[current_corner_id].next_corner_around_a_face;
		lambda(current_corner_id);
		face_is_valid = face.face_corner_list_base.index != u32_max;
		
		current_corner_id = next_corner_around_a_face;
	} while (current_corner_id.index != face_corner_list_base_id.index && face_is_valid);
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
	
	VertexID vertex_id_allocator;
	vertex_id_allocator.index = 0;
	std::unordered_map<Vector3, VertexID, VertexHasher> vertex_position_to_id;

	for (u32 i = 0; i < triangle_mesh.vertices.size(); i += 1) {
		auto& position = triangle_mesh.vertices[i].position;
		
		auto [it, is_inserted] = vertex_position_to_id.emplace(position, vertex_id_allocator);
		if (is_inserted) {
			auto& vertex = result_mesh.vertices.emplace_back();
			vertex.position = position;
			vertex.vertex_corner_list_base.index = u32_max;
			
			vertex_id_allocator.index += 1;
		}
		
		src_vertex_index_to_vertex_id[i] = it->second;
	}
	
	
	
	for (u32 i = 0; i < triangle_mesh.indices.size(); i += 1) {
		u32 src_vertex_index = triangle_mesh.indices[i];
		VertexID vertex_id = src_vertex_index_to_vertex_id[src_vertex_index];
		CornerID corner_id = { i };
		AttributesID attributes_id = { src_vertex_index };
		
		VertexCornerListInsert(mesh, vertex_id, corner_id);
		
		auto& corner = mesh[corner_id];
		corner.vertex_id     = vertex_id;
		corner.attributes_id = attributes_id;
		
		memcpy(mesh[attributes_id], (float*)&triangle_mesh.vertices[src_vertex_index] + 3, attribute_stride_dwords * sizeof(u32));
	}



	EdgeID edge_id_allocator;
	edge_id_allocator.index = 0;
	std::unordered_map<Edge, EdgeID, VertexHasher> edge_to_edge_id;

	for (u32 i = 0; i < triangle_mesh.indices.size(); i += 3) {
		FaceID face_id = { i / 3 };
		auto& face = mesh[face_id];
		face.face_corner_list_base.index = u32_max;
		
		for (u32 corner_index = 0; corner_index < 3; corner_index += 1) {
			u32 src_vertex_index_0 = triangle_mesh.indices[i + corner_index];
			u32 src_vertex_index_1 = triangle_mesh.indices[i + (corner_index + 1) % 3];

			VertexID vertex_id_0 = src_vertex_index_to_vertex_id[src_vertex_index_0];
			VertexID vertex_id_1 = src_vertex_index_to_vertex_id[src_vertex_index_1];

			Edge edge_0 = { vertex_id_0, vertex_id_1 };

			auto [it, is_inserted] = edge_to_edge_id.emplace(edge_0, edge_id_allocator);
			if (is_inserted) {
				auto& edge = result_mesh.edges.emplace_back();
				edge.vertex_0 = vertex_id_0;
				edge.vertex_1 = vertex_id_1;
				edge.edge_corner_list_base.index = u32_max;
				
				edge_id_allocator.index += 1;
			}

			auto& corner = mesh[CornerID{ i + corner_index }];
			corner.face_id = face_id;
			corner.edge_id = it->second;

			EdgeCornerListInsert(mesh, it->second, CornerID{ i + corner_index });
			FaceCornerListInsert(mesh, face_id, CornerID{ i + corner_index });
		}
	}

	
#if 0
	for (u32 i = 0; i < vertices.size(); i += 1) {
		auto vertex = vertices[VertexID{ i }];

		IterateVertexCornerList(mesh, vertex.vertex_corner_list_base, [](CornerID corner_id) {
			printf("%u -> ", corner_id.index);
		});
		printf("%u\n", vertex.vertex_corner_list_base.index);
	}
#endif

#if 0
	for (u32 i = 0; i < edge_id_allocator.index; i += 1) {
		auto edge = mesh[EdgeID{ i }];

		IterateEdgeCornerList(mesh, edge.edge_corner_list_base, [](CornerID corner_id) {
			printf("%u -> ", corner_id.index);
		});
		printf("%u\n", edge.edge_corner_list_base.index);
	}
#endif
	
#if 0
	for (u32 i = 0; i < triangle_mesh.indices.size(); i += 3) {
		auto face = mesh[FaceID{ i / 3 }];

		IterateFaceCornerList(mesh, face.face_corner_list_base, [](CornerID corner_id) {
			printf("%u -> ", corner_id.index);
		});
		printf("%u\n", face.face_corner_list_base.index);
	}
#endif
	
	return result_mesh;
}


static void PerformEdgeCollapse(MeshView mesh, EdgeID edge_id) {
	auto& edge = mesh[edge_id];
	
	auto& vertex_0 = mesh[edge.vertex_0];
	auto& vertex_1 = mesh[edge.vertex_1];

	assert(edge.edge_corner_list_base.index != u32_max);
	assert(vertex_0.vertex_corner_list_base.index != u32_max);
	assert(vertex_1.vertex_corner_list_base.index != u32_max);
	
	
	IterateEdgeCornerList(mesh, edge.edge_corner_list_base, [&](CornerID corner_id) {
		auto& corner = mesh[corner_id];
		auto& face   = mesh[corner.face_id];
		
		IterateFaceCornerList(mesh, face.face_corner_list_base, [&](CornerID corner_id) {
			VertexCornerListRemove(mesh, corner_id);
			EdgeCornerListRemove(mesh, corner_id);
			FaceCornerListRemove(mesh, corner_id);
		});
	});
	
	{
		// Splice vertex lists
		auto base0 = mesh[edge.vertex_0].vertex_corner_list_base;
		auto base1 = mesh[edge.vertex_1].vertex_corner_list_base;
		
		if (base0.index != u32_max && base1.index != u32_max) {
			IterateVertexCornerList(mesh, base1, [&](CornerID corner_id) {
				mesh[corner_id].vertex_id = edge.vertex_0;
				
				IterateFaceCornerList(mesh, corner_id, [&](CornerID corner_id) {
					auto& e0 = mesh[mesh[corner_id].edge_id];
					
					if (e0.vertex_0.index == edge.vertex_1.index) {
						e0.vertex_0 = edge.vertex_0;
					}
					
					if (e0.vertex_1.index == edge.vertex_1.index) {
						e0.vertex_1 = edge.vertex_0;
					}
				});
			});
			
			u32 degree0 = 0; IterateVertexCornerList(mesh, base0, [&](CornerID) { degree0 += 1; });
			u32 degree1 = 0; IterateVertexCornerList(mesh, base1, [&](CornerID) { degree1 += 1; });
			
			auto base0_prev_corner_around_a_vertex = mesh[base0].prev_corner_around_a_vertex;
			auto base1_prev_corner_around_a_vertex = mesh[base1].prev_corner_around_a_vertex;
			
			mesh[base0].prev_corner_around_a_vertex = base1_prev_corner_around_a_vertex;
			mesh[base1_prev_corner_around_a_vertex].next_corner_around_a_vertex = base0;
			
			mesh[base1].prev_corner_around_a_vertex = base0_prev_corner_around_a_vertex;
			mesh[base0_prev_corner_around_a_vertex].next_corner_around_a_vertex = base1;
			
			u32 degree2 = 0; IterateVertexCornerList(mesh, base0, [&](CornerID) { degree2 += 1; });
			assert(degree0 + degree1 == degree2);
		}
	}
	
	auto remaning_base_id   = vertex_0.vertex_corner_list_base.index != u32_max ? vertex_0.vertex_corner_list_base : vertex_1.vertex_corner_list_base;
	auto remaning_base_id_nocheckin = vertex_0.vertex_corner_list_base.index != u32_max ? 0 : 1;
	
	std::unordered_map<Edge, EdgeID, VertexHasher> edge_duplicate_map;
	
	if (remaning_base_id.index != u32_max) IterateVertexCornerList(mesh, remaning_base_id, [&](CornerID corner_id) {
		IterateFaceCornerList(mesh, corner_id, [&](CornerID corner_id) {
			auto edge_id_1 = mesh[corner_id].edge_id;
			
			auto [it, is_inserted] = edge_duplicate_map.emplace(mesh[edge_id_1], edge_id_1);
			auto edge_id_0 = it->second;
			
			if (is_inserted == false && edge_id_0.index != edge_id_1.index) {
				auto base0 = mesh[edge_id_0].edge_corner_list_base;
				auto base1 = mesh[edge_id_1].edge_corner_list_base;
				
				u32 degree0 = 0; IterateEdgeCornerList(mesh, base0, [&](CornerID) { degree0 += 1; });
				u32 degree1 = 0; IterateEdgeCornerList(mesh, base1, [&](CornerID) { degree1 += 1; });
				
				auto base0_prev_corner_around_an_edge = mesh[base0].prev_corner_around_an_edge;
				auto base1_prev_corner_around_an_edge = mesh[base1].prev_corner_around_an_edge;
				
				mesh[base0].prev_corner_around_an_edge = base1_prev_corner_around_an_edge;
				mesh[base1_prev_corner_around_an_edge].next_corner_around_an_edge = base0;
				
				mesh[base1].prev_corner_around_an_edge = base0_prev_corner_around_an_edge;
				mesh[base0_prev_corner_around_an_edge].next_corner_around_an_edge = base1;
				
				u32 degree2 = 0; IterateEdgeCornerList(mesh, base0, [&](CornerID) { degree2 += 1; });
				assert(degree0 + degree1 == degree2);
				
				IterateEdgeCornerList(mesh, base0, [&](CornerID corner_id) {
					mesh[corner_id].edge_id = edge_id_0;
				});
				mesh[edge_id_1].edge_corner_list_base.index = u32_max;
			}
		});
	});
	
	if (remaning_base_id_nocheckin == 0) {
		vertex_1.vertex_corner_list_base.index = u32_max; // Removed.
	} else {
		vertex_0.vertex_corner_list_base.index = u32_max; // Removed.
	}
}

void PerformRandomEdgeCollapse(MeshView mesh) {
#if 0
	auto edge_id = EdgeID{ 1 };
	
	printf("CollapseID: %u\n", edge_id.index);
	PerformEdgeCollapse(mesh, edge_id);

	edge_id = EdgeID{ 7 };
	printf("CollapseID: %u\n", edge_id.index);
	PerformEdgeCollapse(mesh, edge_id);

	int k = 2;
#else
	u32 to_collapse = 1800;
	if (to_collapse >= mesh.edge_count) to_collapse = mesh.edge_count;
	for (u32 i = 0; i < to_collapse; i += 4) {
		auto edge_id = EdgeID{ i };
		if (mesh[edge_id].edge_corner_list_base.index == u32_max) continue;
		
		printf("CollapseID: %u\n", edge_id.index);
		PerformEdgeCollapse(mesh, edge_id);
	}
#endif

#if 0
	for (u32 i = 0; i < mesh.vertex_count; i += 1) {
		auto vertex = mesh[VertexID{ i }];
		if (vertex.vertex_corner_list_base.index == u32_max) continue;

		printf("v%u: ", i);

		IterateVertexCornerList(mesh, vertex.vertex_corner_list_base, [](CornerID corner_id) {
			printf("%u -> ", corner_id.index);
		});
		printf("%u\n", vertex.vertex_corner_list_base.index);
	}
#endif

#if 0
	for (u32 i = 0; i < mesh.edge_count; i += 1) {
		auto edge = mesh[EdgeID{ i }];
		if (edge.edge_corner_list_base.index == u32_max) continue;

		printf("e%u: ", i);

		IterateEdgeCornerList(mesh, edge.edge_corner_list_base, [](CornerID corner_id) {
			printf("%u -> ", corner_id.index);
		});
		printf("%u\n", edge.edge_corner_list_base.index);
	}
#endif
	
#if 0
	for (u32 i = 0; i < mesh.face_count; i += 1) {
		auto face = mesh[FaceID{ i }];
		if (face.face_corner_list_base.index == u32_max) continue;

		printf("f%u: ", i);

		IterateFaceCornerList(mesh, face.face_corner_list_base, [](CornerID corner_id) {
			printf("%u -> ", corner_id.index);
		});
		printf("%u\n", face.face_corner_list_base.index);
	}
#endif

#if 0
	auto edge_id = EdgeID{ 1 };
	
	printf("CollapseID: %u\n", edge_id.index);
	PerformEdgeCollapse(mesh, edge_id);
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


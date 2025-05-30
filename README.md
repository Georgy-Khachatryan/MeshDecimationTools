
## Mesh Decimation and Virtual Geometry Construction
This library provides functionality for generating discrete and continuous levels of detail for triangle meshes using a custom edge collapsing mesh decimation algorithm. Main features:
- Robust handling of non manifold meshes.
- Vertex placement optimization post edge collapse.
- Support for vertex attributes and attribute discontinuities.
- Accurate estimation of introduced error using quadric error metric.
- Seamless decimation of multiple geometries.


## How to use
### Common level of detail build inputs
Geometries are defined by a 32 bit index buffer and a variable stride vertex buffer. Vertices are represented by arrays of 32 bit floats where the first three floats are the vertex position and all remaning floats are vertex attributes. The maximum number of attributes per vertex is defined by `VGT_MAX_ATTRIBUTE_STRIDE_DWORDS`. See [compile time configuration](##compile-time-configuration) for more details.
```
struct Vertex {
    float position_x, position_y, position_z;
    float texcoord_u, texcoord_v;
    float normal_x, normal_y, normal_z;
};

VgtTriangleGeometryDesc geometry_descs[1] = {};
geometry_descs[0].indices      = triangle_mesh.indices.data();
geometry_descs[0].index_count  = triangle_mesh.indices.size();
geometry_descs[0].vertices     = (float*)triangle_mesh.vertices.data();
geometry_descs[0].vertex_count = triangle_mesh.vertices.size();
```
LOD build algorithms take an array of geometries as an input. All geometries of a mesh are decimated together which results in no seams at higher LODs.
```
VgtTriangleMeshDesc mesh_desc = {};
mesh_desc.geometry_descs      = geometry_descs;
mesh_desc.geometry_desc_count = 1;
mesh_desc.vertex_stride_bytes = sizeof(Vertex); // Vertices in all geometries must have the same layout.
mesh_desc.attribute_weights   = NULL;
mesh_desc.normalize_vertex_attributes = NULL;
```
Optional attribute weights can be specified to control attribute error importance relative to geometric error. Default weights==1.0 work well for vertex normals, UV coordinates, colors, etc.
```
float attribute_weights[VGT_MAX_ATTRIBUTE_STRIDE_DWORDS] = {};
attribute_weights[0] = uv_weight;
attribute_weights[1] = uv_weight;
attribute_weights[2] = normal_weight;
attribute_weights[3] = normal_weight;
attribute_weights[4] = normal_weight;

mesh_desc.attribute_weights = attribute_weights;
```


Optional vertex normalization callback can be specified to normalize unit vectors, clamp uv coordinates or colors on newly computed sets of attributes.
```
static void NormalizeVertexAttributes(float* attributes) { // Only attributes are passed to the callback.
    auto& vertex_attributes = *(VertexAttributes*)attributes;
    vertex_attributes.normal = NormalizeVector3(vertex_attributes.normal);
}

mesh_desc.normalize_vertex_attributes = &NormalizeVertexAttributes;
```

### Memory allocation
TODO

### Continuous level of detail
Continuous levels of detail build requires you to specify the maximum number of triangles and vertices per meshlet. Meshlets are the unit of mesh rendering and LOD swapping. For decimation meshlets are combined into groups of 32.
```
VgtVirtualGeometryBuildInputs inputs;
inputs.mesh                          = mesh_desc;
inputs.meshlet_target_vertex_count   = 128;
inputs.meshlet_target_triangle_count = 128;

VgtVirtualGeometryBuildResult result;
VgtBuildVirtualGeometry(&inputs, &result, &callbacks);
```

### Discrete level of detail
Discrete levels of detail build requires you to specify target number of faces as well as maximum allowed decimation error.
```
VgtMeshDecimationInputs inputs;
inputs.mesh               = mesh_desc;
inputs.target_face_count  = (triangle_mesh.indices.size() / 3) / 2; // Decimate to half the number of faces.
inputs.target_error_limit = FLT_MAX; // Allow any error.

VgtMeshDecimationResult result;
VgtDecimateMesh(&inputs, &result, &callbacks);
```

## How to render
TODO

## Compile time configuration
The library has a small set of configurations that are performance sensitive and have to be done using global definitions. There are two ways to set them:
- Globally define `VGT_CONFIGURATION_FILE="Path/To/VgtConfigurationFile.h"` in your build system and add individual definitions to the configuration file.
- Globally set each definition in your build system.

### Vertex attribute count limit
Default vertex attribute count limit is set to 16. If you have more or less attributes per vertex, you can set a custom limit using `VGT_MAX_ATTRIBUTE_STRIDE_DWORDS`. For example:
```
// Default attribute count limit is set to 16.
#define VGT_MAX_ATTRIBUTE_STRIDE_DWORDS 16
```
### Meshlet group size
Meshlet groups size used for virtual geometry construction is defined by `VGT_MESHLET_GROUP_SIZE`. Larger group sizes result in higher quality decimation at the cost of less grannular LOD swapping at runtime.
```
#define VGT_MESHLET_GROUP_SIZE 32
```


## Third Party
Uses [meshoptimizer](https://github.com/zeux/meshoptimizer). Copyright (c) 2016-2025, Arseny Kapoulkine. Avaliable under MIT license. See full license text in [THIRD_PARTY_LICENSES.md](https://github.com/Georgy-Khachatryan/MeshDecimation/blob/master/THIRD_PARTY_LICENSES.md)
- Meshlet generation algorithm is based on the implementation from meshoptimizer.
- Some internal data structures and utility functions are based on meshoptimizer code. Check references to [Kapoulkine 2025] for more detail.

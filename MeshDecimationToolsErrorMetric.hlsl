#ifndef MESHDECIMATIONTOOLSERRORMETRIC_HLSL
#define MESHDECIMATIONTOOLSERRORMETRIC_HLSL

struct MdtModelToWorldTransform {
	float3x3 rotation_and_scale;
	float3   position;
	float    max_scale;
};

struct MdtSphereBounds {
	float3 center;
	float  radius;
};

struct MdtErrorMetric {
	MdtSphereBounds bounds;
	float           error;
};

struct MdtPixelSpaceErrorMetric {
	float numerator;
	float denominator;
};

// 
// Example usage:
// 
// float target_error_pixels = 1.0;
// float world_to_pixel_space_scale = MdtComputeWorldToPixelSpaceScale(camera.view_to_clip[0][0], camera.render_target_size.x, target_error_pixels);
// 
// MdtModelToWorldTransform model_to_world;
// model_to_world.position           = instance.model_to_world.position;
// model_to_world.rotation_and_scale = mul(QuaternionToFloat3x3(instance.model_to_world.rotation), ScaleVectorToFloat3x3(instance.model_to_world.scale));
// model_to_world.max_scale          = Max3(instance.model_to_world.scale.x, instance.model_to_world.scale.y, instance.model_to_world.scale.z);
// 
// MdtPixelSpaceErrorMetric coarser_error = MdtEvaluateErrorMetric(model_to_world, meshlet.coarser_level_error_metric, camera_position_world_space, world_to_pixel_space_scale);
// MdtPixelSpaceErrorMetric current_error = MdtEvaluateErrorMetric(model_to_world, meshlet.current_level_error_metric, camera_position_world_space, world_to_pixel_space_scale);
// 
// bool is_visible = MdtLevelOfDetailCullCoarserLevelError(coarser_error) && MdtLevelOfDetailCullCurrentLevelError(current_error);
// 
MdtPixelSpaceErrorMetric MdtEvaluateErrorMetric(MdtModelToWorldTransform model_to_world, MdtErrorMetric error_metric, float3 camera_position_world_space, float world_to_pixel_space_scale) {
	const float3 center_world_space = mul(model_to_world.rotation_and_scale, error_metric.bounds.center) + model_to_world.position;
	const float  radius_world_space = error_metric.bounds.radius * model_to_world.max_scale;
	
	//
	// Error sphere is projected as if it was at the closest point on the bounding sphere.
	// 
	// Given that error_metric.bounds for coarser level meshlets always encloses finer 
	// level bounds, distance_to_sphere_world_space is always smaller for coarser level
	// meshlets. This makes the projected error for coarser levels always larger than
	// for finer ones.
	// 
	const float3 camera_to_center_world_space   = center_world_space - camera_position_world_space;
	const float  distance_to_sphere_world_space = max(length(camera_to_center_world_space) - radius_world_space, 0.f);
	
	// 
	// Both numerator and denominator are always positive, so we can easily subsitute:
	//   numerator / denominator <=> 1.0
	// with
	//   numerator <=> denominator
	// 
	MdtPixelSpaceErrorMetric result;
	result.numerator   = model_to_world.max_scale * error_metric.error * world_to_pixel_space_scale;
	result.denominator = distance_to_sphere_world_space;
	
	return result;
}

// This value can be precomputed on CPU and passed to shaders.
float MdtComputeWorldToPixelSpaceScale(float view_to_clip_0_0, float render_target_size_x, float target_error_pixels) {
	// World to pixel space projection scale assuming that the sphere is exactly at the center of the screen.
	return (view_to_clip_0_0 * render_target_size_x * 0.5f) / max(target_error_pixels, 1.f);
}

bool MdtLevelOfDetailCullCurrentLevelError(MdtPixelSpaceErrorMetric error_metric) { return error_metric.numerator <= error_metric.denominator; }
bool MdtLevelOfDetailCullCoarserLevelError(MdtPixelSpaceErrorMetric error_metric) { return error_metric.numerator >  error_metric.denominator; }

#endif // MESHDECIMATIONTOOLSERRORMETRIC_HLSL

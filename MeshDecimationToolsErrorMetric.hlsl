#ifndef MESHDECIMATIONTOOLSERRORMETRIC_HLSL
#define MESHDECIMATIONTOOLSERRORMETRIC_HLSL

struct MdtModelToWorldTransform {
	float3x3 rotation;
	float3   position;
	float    scale;
};

struct MdtErrorMetric {
	float3 aabb_min;
	float3 aabb_max;
	float  error;
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
// model_to_world.rotation = QuaternionToFloat3x3(instance.model_to_world.rotation);
// model_to_world.position = instance.model_to_world.position;
// model_to_world.scale    = instance.model_to_world.scale;
// 
// MdtPixelSpaceErrorMetric coarser_error = MdtEvaluateErrorMetric(model_to_world, meshlet.coarser_level_error_metric, camera_position_world_space, world_to_pixel_space_scale);
// MdtPixelSpaceErrorMetric current_error = MdtEvaluateErrorMetric(model_to_world, meshlet.current_level_error_metric, camera_position_world_space, world_to_pixel_space_scale);
// 
// bool is_visible = MdtLevelOfDetailCullCoarserLevelError(coarser_error) && MdtLevelOfDetailCullCurrentLevelError(current_error);
// 
MdtPixelSpaceErrorMetric MdtEvaluateErrorMetric(MdtModelToWorldTransform model_to_world, MdtErrorMetric error_metric, float3 camera_position_world_space, float world_to_pixel_space_scale) {
	const float3x3 world_to_model_rotation = transpose(model_to_world.rotation);
	const float3 camera_position_model_space = mul(world_to_model_rotation, camera_position_world_space - model_to_world.position) / model_to_world.scale;
	
	// 
	// Error sphere is projected as if it was at the closest point on the bounding box.
	// 
	// Given that error_metric.aabb_min/aabb_max for coarser level meshlets always encloses finer 
	// level bounds, distance_to_box_world_space is always smaller for coarser level
	// meshlets. This makes the projected error for coarser levels always larger than
	// for finer ones.
	// 
	const float3 camera_to_box_model_space   = clamp(camera_position_model_space, error_metric.aabb_min, error_metric.aabb_max) - camera_position_model_space;
	const float  distance_to_box_world_space = length(camera_to_box_model_space) * model_to_world.scale;
	
	// 
	// Both numerator and denominator are always positive, so we can easily subsitute:
	//   numerator / denominator <=> 1.0
	// with
	//   numerator <=> denominator
	// 
	MdtPixelSpaceErrorMetric result;
	result.numerator   = model_to_world.scale * error_metric.error * world_to_pixel_space_scale;
	result.denominator = distance_to_box_world_space;
	
	return result;
}

// This value can be precomputed on CPU and passed to shaders.
float MdtComputeWorldToPixelSpaceScale(float view_to_clip_0_0, float render_target_size_x, float target_error_pixels) {
	// World to pixel space projection scale assuming that the error sphere is exactly at the center of the screen.
	return (view_to_clip_0_0 * render_target_size_x * 0.5f) / max(target_error_pixels, 1.f);
}

bool MdtLevelOfDetailCullCurrentLevelError(MdtPixelSpaceErrorMetric error_metric) { return error_metric.numerator <= error_metric.denominator; }
bool MdtLevelOfDetailCullCoarserLevelError(MdtPixelSpaceErrorMetric error_metric) { return error_metric.numerator >  error_metric.denominator; }

#endif // MESHDECIMATIONTOOLSERRORMETRIC_HLSL

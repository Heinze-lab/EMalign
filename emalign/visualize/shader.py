FLOW_SHADER = '''#uicontrol bool diverging checkbox(default=true)
#uicontrol float vmax slider(min=0.1, max=100, default=10, step=0.1)
#uicontrol float opacity slider(min=0, max=1, default=1)

// diverging: RdBu over [-vmax, vmax] (flow x / y components)
// otherwise: cubehelix over [0, vmax] (peak ratio / sharpness, magnitude)

// Matplotlib RdBu: t=0 -> dark red, t=0.5 -> white, t=1 -> dark blue
vec3 rdbu(float t) {
  t = clamp(t, 0.0, 1.0);
  vec3 c0 = vec3(0.404, 0.000, 0.122);
  vec3 c1 = vec3(0.839, 0.376, 0.302);
  vec3 c2 = vec3(0.969, 0.969, 0.969);
  vec3 c3 = vec3(0.263, 0.576, 0.765);
  vec3 c4 = vec3(0.020, 0.188, 0.380);
  if (t < 0.25) return mix(c0, c1, t / 0.25);
  if (t < 0.50) return mix(c1, c2, (t - 0.25) / 0.25);
  if (t < 0.75) return mix(c2, c3, (t - 0.50) / 0.25);
  return mix(c3, c4, (t - 0.75) / 0.25);
}

void main() {
  float v = getDataValue();
  if (isnan(v)) {
    emitTransparent();
    return;
  }

  vec3 rgb;
  if (diverging) {
    rgb = rdbu(0.5 + 0.5 * v / vmax);
  } else {
    rgb = colormapCubehelix(clamp(v / vmax, 0.0, 1.0));
  }
  emitRGBA(vec4(rgb, opacity));
}
'''
// GLSL for the obsidian-chamber topology. Everything uses straight alpha blending and finishes with
// <colorspace_fragment> so hex colors land in sRGB exactly as designed.

export const conduitVertex = /* glsl */ `
  attribute float aSeed;
  varying vec2 vUv;
  varying float vSeed;
  varying float vFres;
  void main() {
    vUv = uv;
    vSeed = aSeed;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vec3 n = normalize(normalMatrix * normal);
    vFres = abs(dot(n, normalize(-mv.xyz)));
    gl_Position = projectionMatrix * mv;
  }
`;

export const conduitFragment = /* glsl */ `
  uniform float uTime;
  uniform float uBoost;
  uniform float uOpacity;
  uniform float uReveal;
  uniform float uFreq;
  uniform float uDim;
  uniform vec3 uColorA;
  uniform vec3 uColorB;
  varying vec2 vUv;
  varying float vSeed;
  varying float vFres;
  void main() {
    if (vUv.x > uReveal) discard;
    float speed = mix(0.3, 2.4, uBoost);
    float freq = uFreq * mix(1.0, 3.0, uBoost);
    float ph = fract(vUv.x * freq - uTime * speed + vSeed);
    float pulse = smoothstep(0.0, 0.05, ph) * (1.0 - smoothstep(0.05, 0.28, ph));
    float body = 0.5 + 0.5 * pow(vFres, 1.4);
    vec3 col = mix(uColorA, uColorB, pulse);
    float a = uOpacity * uDim * body * (0.38 + 0.62 * pulse + uBoost * 0.12);
    if (uReveal < 0.999) a += exp(-pow((uReveal - vUv.x) * 55.0, 2.0)) * 0.55;
    gl_FragColor = vec4(col, clamp(a, 0.0, 1.0));
    #include <colorspace_fragment>
  }
`;

export const planeVertex = /* glsl */ `
  varying vec3 vWorld;
  void main() {
    vec4 w = modelMatrix * vec4(position, 1.0);
    vWorld = w.xyz;
    gl_Position = projectionMatrix * viewMatrix * w;
  }
`;

export const planeFragment = /* glsl */ `
  uniform vec3 uLine;
  uniform vec3 uFill;
  uniform vec3 uRim;
  uniform float uRadius;
  uniform float uCell;
  uniform float uLineAlpha;
  uniform float uFillAlpha;
  uniform float uRimAlpha;
  varying vec3 vWorld;
  void main() {
    float r = length(vWorld.xz);
    if (r > uRadius * 1.04) discard;
    vec2 coord = vWorld.xz / uCell;
    vec2 g = abs(fract(coord - 0.5) - 0.5) / fwidth(coord);
    float line = 1.0 - min(min(g.x, g.y), 1.0);
    float fade = 1.0 - smoothstep(uRadius * 0.5, uRadius, r);
    float rim = smoothstep(0.4, 0.0, abs(r - uRadius));
    float aLine = line * uLineAlpha * fade;
    float aFill = uFillAlpha * (0.35 + 0.65 * fade);
    float aRim = rim * uRimAlpha;
    float a = max(max(aLine, aFill), aRim);
    vec3 col = uFill;
    col = mix(col, uLine, clamp(aLine / max(a, 1e-4), 0.0, 1.0));
    col = mix(col, uRim, clamp(aRim / max(a, 1e-4), 0.0, 1.0));
    gl_FragColor = vec4(col, a);
    #include <colorspace_fragment>
  }
`;

export const cloudVertex = /* glsl */ `
  attribute float aSeed;
  attribute float aSize;
  attribute float aTint;
  attribute float aBranch;
  uniform float uFocusBranch;
  uniform float uFocusAmt;
  uniform float uTime;
  uniform float uPixelRatio;
  uniform float uQuant;
  uniform vec3 uShockCenter;
  uniform float uShockTime;
  uniform float uHit;
  uniform float uHitRadius;
  uniform vec3 uColorA;
  uniform vec3 uColorB;
  uniform vec3 uHitColor;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vec3 p = position;
    // SQ8 demonstration: snap coordinates onto a coarse 8-bit style lattice.
    float cell = 0.7;
    p.xz = mix(p.xz, floor(p.xz / cell + 0.5) * cell, uQuant);
    p.y = mix(p.y, floor(p.y / (cell * 0.5) + 0.5) * cell * 0.5, uQuant);

    float d = distance(p.xz, uShockCenter.xz);
    float wave = 0.0;
    if (uShockTime >= 0.0) {
      float r = uShockTime * 14.0;
      wave = exp(-pow((d - r) * 0.6, 2.0)) * exp(-uShockTime * 0.9);
      p.y += wave * 1.1;
    }
    float near = (1.0 - smoothstep(0.0, uHitRadius, d)) * uHit;
    float tw = 0.85 + 0.15 * sin(uTime * 1.3 + aSeed * 43.0);
    vColor = mix(mix(uColorA, uColorB, aTint), uHitColor, clamp(near * 0.7 + wave, 0.0, 1.0));
    float match = 1.0 - step(0.5, abs(aBranch - uFocusBranch));
    float dim = mix(1.0, mix(0.15, 1.0, match), uFocusAmt);
    vAlpha = clamp((0.42 + uQuant * 0.2) * tw + wave * 0.5 + near * 0.3, 0.0, 0.95) * dim;

    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    float s = aSize * (1.0 + wave * 1.2 + near * 0.5 + uQuant * 0.6);
    gl_PointSize = min(s * uPixelRatio * (170.0 / -mv.z), 24.0 * uPixelRatio);
    gl_Position = projectionMatrix * mv;
  }
`;

export const cloudFragment = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    float a = smoothstep(0.5, 0.18, d) * vAlpha;
    gl_FragColor = vec4(vColor, a);
    #include <colorspace_fragment>
  }
`;

export const haloVertex = /* glsl */ `
  attribute float aSize;
  attribute float aSeed;
  attribute vec3 aColor;
  attribute float aBranch;
  uniform float uTime;
  uniform float uPixelRatio;
  uniform float uPulse;
  uniform float uFocusBranch;
  uniform float uFocusAmt;
  varying vec3 vColor;
  varying float vDim;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    float p = 1.0 + uPulse * 0.14 * sin(uTime * 1.8 + aSeed * 6.2831);
    gl_PointSize = min(aSize * p * uPixelRatio * (260.0 / -mv.z), 200.0 * uPixelRatio);
    gl_Position = projectionMatrix * mv;
    vColor = aColor;
    float match = 1.0 - step(0.5, abs(aBranch - uFocusBranch));
    vDim = mix(1.0, mix(0.15, 1.0, match), uFocusAmt);
  }
`;

export const haloFragment = /* glsl */ `
  uniform float uIntensity;
  varying vec3 vColor;
  varying float vDim;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    float a = pow(max(0.0, 1.0 - d * 2.0), 2.2) * uIntensity * vDim;
    gl_FragColor = vec4(vColor, a);
    #include <colorspace_fragment>
  }
`;

export const streamVertex = /* glsl */ `
  attribute float aSeed;
  attribute float aOffset;
  uniform float uTime;
  uniform float uPixelRatio;
  uniform float uStream;
  varying float vAlpha;
  void main() {
    vec3 p = position;
    float speed = 5.0 + aSeed * 9.0;
    float h = mod(aOffset + uTime * speed * (0.6 + uStream), 72.0);
    p.y = -6.0 + h;
    float shrink = smoothstep(30.0, 72.0, h);
    p.xz *= 1.0 - shrink * 0.35;
    vAlpha = (1.0 - smoothstep(48.0, 72.0, h)) * smoothstep(0.0, 6.0, h) * uStream;
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_PointSize = min((1.1 + aSeed) * uPixelRatio * (240.0 / -mv.z), 18.0 * uPixelRatio);
    gl_Position = projectionMatrix * mv;
  }
`;

export const streamFragment = /* glsl */ `
  uniform vec3 uColor;
  varying float vAlpha;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    float a = smoothstep(0.5, 0.15, d) * vAlpha * 0.8;
    gl_FragColor = vec4(uColor, a);
    #include <colorspace_fragment>
  }
`;

export const beamVertex = /* glsl */ `
  varying vec2 vUv;
  varying float vFres;
  void main() {
    vUv = uv;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vec3 n = normalize(normalMatrix * normal);
    vFres = abs(dot(n, normalize(-mv.xyz)));
    gl_Position = projectionMatrix * mv;
  }
`;

export const beamFragment = /* glsl */ `
  uniform vec3 uColor;
  uniform float uOpacity;
  varying vec2 vUv;
  varying float vFres;
  void main() {
    float core = pow(vFres, 1.6);
    float grad = mix(0.85, 0.0, pow(vUv.y, 0.55));
    gl_FragColor = vec4(uColor, core * grad * uOpacity);
    #include <colorspace_fragment>
  }
`;

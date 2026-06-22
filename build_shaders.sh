#!/bin/bash
# Regenerate all SPIR-V shaders from GLSL source and create C headers
set -e

# Recreate all shader source files
cat > cs_bayer_to_rgb.comp << 'SHADER'
#version 460
layout(local_size_x = 16, local_size_y = 16) in;
layout(set = 0, binding = 0, std430) readonly buffer IB { float bayer[]; };
layout(set = 0, binding = 1, std430) buffer OB { float rgb[]; };
layout(push_constant) uniform P { int width; int height; int bayer_pattern; float _pad; };
bool is_red(int y, int x)    { return (y % 2 == 1) && (x % 2 == 1); }
bool is_green_r(int y, int x){ return (y % 2 == 1) && (x % 2 == 0); }
bool is_green_b(int y, int x){ return (y % 2 == 0) && (x % 2 == 1); }
bool is_blue(int y, int x)   { return (y % 2 == 0) && (x % 2 == 0); }
float ld(int y, int x) {
    if (y < 0 || y >= height || x < 0 || x >= width) return 0.0;
    return bayer[y * width + x];
}
void main() {
    int y = int(gl_GlobalInvocationID.y);
    int x = int(gl_GlobalInvocationID.x);
    if (y >= height || x >= width) return;
    int idx = (y * width + x) * 3;
    float r, g, b;
    float dh = abs(ld(y, x-1) - ld(y, x+1));
    float dv = abs(ld(y-1, x) - ld(y+1, x));
    if (is_red(y, x)) {
        r = ld(y, x);
        float g_h = (ld(y, x-1) + ld(y, x+1)) / 2.0;
        float g_v = (ld(y-1, x) + ld(y+1, x)) / 2.0;
        g = (dh < dv) ? g_h : ((dv < dh) ? g_v : (g_h + g_v) / 2.0);
        b = (ld(y-1, x-1) + ld(y-1, x+1) + ld(y+1, x-1) + ld(y+1, x+1)) / 4.0;
    } else if (is_blue(y, x)) {
        b = ld(y, x);
        float g_h = (ld(y, x-1) + ld(y, x+1)) / 2.0;
        float g_v = (ld(y-1, x) + ld(y+1, x)) / 2.0;
        g = (dh < dv) ? g_h : ((dv < dh) ? g_v : (g_h + g_v) / 2.0);
        r = (ld(y-1, x-1) + ld(y-1, x+1) + ld(y+1, x-1) + ld(y+1, x+1)) / 4.0;
    } else {
        g = ld(y, x);
        if (is_green_r(y, x)) {
            r = (ld(y, x-1) + ld(y, x+1)) / 2.0;
            b = (ld(y-1, x) + ld(y+1, x)) / 2.0;
        } else {
            b = (ld(y, x-1) + ld(y, x+1)) / 2.0;
            r = (ld(y-1, x) + ld(y+1, x)) / 2.0;
        }
    }
    rgb[idx] = clamp(r, 0.0, 1.0);
    rgb[idx+1] = clamp(g, 0.0, 1.0);
    rgb[idx+2] = clamp(b, 0.0, 1.0);
}
SHADER

cat > cs_blc_wb.comp << 'SHADER'
#version 460
layout(local_size_x = 32, local_size_y = 8) in;
layout(set = 0, binding = 0, std430) readonly buffer IB { float raw[]; };
layout(set = 0, binding = 1, std430) buffer OB { float outbuf[]; };
layout(push_constant) uniform P { int width; int height; int bayer_pattern; float _pad; };
layout(set = 0, binding = 2, std430) readonly buffer GB { float blc[4]; float wb_gains[4]; };
int channel(int y, int x) {
    int p = bayer_pattern;
    if (p == 0) { if ((y & 1) == 0) return (x & 1) == 0 ? 0 : 1; else return (x & 1) == 0 ? 2 : 3; }
    else if (p == 1) { if ((y & 1) == 0) return (x & 1) == 0 ? 3 : 2; else return (x & 1) == 0 ? 1 : 0; }
    else if (p == 2) { if ((y & 1) == 0) return (x & 1) == 0 ? 1 : 0; else return (x & 1) == 0 ? 3 : 2; }
    else { if ((y & 1) == 0) return (x & 1) == 0 ? 2 : 3; else return (x & 1) == 0 ? 0 : 1; }
}
void main() {
    int y = int(gl_GlobalInvocationID.y);
    int x = int(gl_GlobalInvocationID.x);
    if (y >= height || x >= width) return;
    int ch = channel(y, x);
    float val = raw[y * width + x];
    float corrected = (val - blc[ch]) * wb_gains[ch];
    outbuf[y * width + x] = clamp(corrected, 0.0, 1.0);
}
SHADER

cat > cs_ccm.comp << 'SHADER'
#version 460
layout(local_size_x = 16, local_size_y = 16) in;
layout(set = 0, binding = 0, std430) readonly buffer IB { float rgb[]; };
layout(set = 0, binding = 1, std430) buffer OB { float outbuf[]; };
layout(push_constant) uniform P { int width; int height; int _pad0; float _pad1; };
layout(set = 0, binding = 2, std430) readonly buffer MB { float ccm[9]; };
void main() {
    int y = int(gl_GlobalInvocationID.y);
    int x = int(gl_GlobalInvocationID.x);
    if (y >= height || x >= width) return;
    int idx = (y * width + x) * 3;
    float r = rgb[idx], g = rgb[idx+1], b = rgb[idx+2];
    outbuf[idx]   = clamp(ccm[0]*r + ccm[1]*g + ccm[2]*b, 0.0, 1.0);
    outbuf[idx+1] = clamp(ccm[3]*r + ccm[4]*g + ccm[5]*b, 0.0, 1.0);
    outbuf[idx+2] = clamp(ccm[6]*r + ccm[7]*g + ccm[8]*b, 0.0, 1.0);
}
SHADER

cat > cs_tone.comp << 'SHADER'
#version 460
layout(local_size_x = 16, local_size_y = 16) in;
layout(set = 0, binding = 0, std430) readonly buffer IB { float rgb[]; };
layout(set = 0, binding = 1, std430) buffer OB { float outbuf[]; };
layout(push_constant) uniform P {
    int width; int height;
    float gamma_val; float contrast; float brightness; float saturation;
};
layout(set = 0, binding = 2, std430) readonly buffer LB { float lut[4096]; };
void main() {
    int y = int(gl_GlobalInvocationID.y);
    int x = int(gl_GlobalInvocationID.x);
    if (y >= height || x >= width) return;
    int idx = (y * width + x) * 3;
    float r = rgb[idx], g = rgb[idx+1], b = rgb[idx+2];
    int li = int(clamp(r, 0.0, 1.0) * 4095.0); r = lut[li];
    li = int(clamp(g, 0.0, 1.0) * 4095.0); g = lut[li];
    li = int(clamp(b, 0.0, 1.0) * 4095.0); b = lut[li];
    r = (r - 0.5) * contrast + 0.5 + brightness;
    g = (g - 0.5) * contrast + 0.5 + brightness;
    b = (b - 0.5) * contrast + 0.5 + brightness;
    float luma = 0.299*r + 0.587*g + 0.114*b;
    outbuf[idx]   = clamp(luma + (r - luma) * saturation, 0.0, 1.0);
    outbuf[idx+1] = clamp(luma + (g - luma) * saturation, 0.0, 1.0);
    outbuf[idx+2] = clamp(luma + (b - luma) * saturation, 0.0, 1.0);
}
SHADER

cat > cs_fcs.comp << 'SHADER'
#version 460
layout(local_size_x = 16, local_size_y = 16) in;
layout(set = 0, binding = 0, std430) readonly buffer IB { float rgb[]; };
layout(set = 0, binding = 1, std430) buffer OB { float outbuf[]; };
layout(push_constant) uniform P { int width; int height; float strength; float _pad; };
float lum(int y, int x) {
    if (y < 0 || y >= height || x < 0 || x >= width) return 0.0;
    int idx = (y * width + x) * 3;
    return 0.299*rgb[idx] + 0.587*rgb[idx+1] + 0.114*rgb[idx+2];
}
void main() {
    int y = int(gl_GlobalInvocationID.y);
    int x = int(gl_GlobalInvocationID.x);
    if (y >= height || x >= width) return;
    float e = 0.0;
    for (int ky = -1; ky <= 1; ky++)
        for (int kx = -2; kx <= 2; kx++) {
            if (ky == 0 && kx == 0) e += lum(y+ky, x+kx) * 1.0;
            else if ((kx & 1) == 0) e += lum(y+ky, x+kx) * (-0.125);
        }
    float es = clamp(abs(e) * strength * 0.125, 0.0, 1.0);
    float att = 1.0 - es;
    int idx = (y * width + x) * 3;
    float r = rgb[idx], g = rgb[idx+1], b = rgb[idx+2];
    float l = 0.299*r + 0.587*g + 0.114*b;
    outbuf[idx]   = l + (r - l) * att;
    outbuf[idx+1] = l + (g - l) * att;
    outbuf[idx+2] = l + (b - l) * att;
}
SHADER

cat > cs_ldci.comp << 'SHADER'
#version 460
layout(local_size_x = 16, local_size_y = 16) in;
layout(set = 0, binding = 0, std430) readonly buffer IB { float rgb[]; };
layout(set = 0, binding = 1, std430) buffer OB { float outbuf[]; };
layout(push_constant) uniform P { int width; int height; float strength; int radius; };
void main() {
    int y = int(gl_GlobalInvocationID.y);
    int x = int(gl_GlobalInvocationID.x);
    if (y >= height || x >= width) return;
    int idx = (y * width + x) * 3;
    float r = rgb[idx], g = rgb[idx+1], b = rgb[idx+2];
    float lum = 0.299*r + 0.587*g + 0.114*b;
    int r2 = radius;
    int x0 = max(x - r2, 0), x1 = min(x + r2, width - 1);
    int y0 = max(y - r2, 0), y1 = min(y + r2, height - 1);
    float sum = 0.0;
    int cnt = 0;
    for (int sy = y0; sy <= y1; sy++) {
        int base = (sy * width + x0) * 3;
        for (int sx = x0; sx <= x1; sx++) {
            sum += rgb[base]; base += 3; cnt++;
        }
    }
    float lmean = sum / float(max(cnt, 1));
    float lc = lum - lmean;
    float enhanced = lum + lc * strength;
    float scale = enhanced / max(lum, 0.0001);
    outbuf[idx]   = clamp(r * scale, 0.0, 1.0);
    outbuf[idx+1] = clamp(g * scale, 0.0, 1.0);
    outbuf[idx+2] = clamp(b * scale, 0.0, 1.0);
}
SHADER

cat > cs_ee.comp << 'SHADER'
#version 460
layout(local_size_x = 16, local_size_y = 16) in;
layout(set = 0, binding = 0, std430) readonly buffer IB { float rgb[]; };
layout(set = 0, binding = 1, std430) buffer OB { float outbuf[]; };
layout(push_constant) uniform P { int width; int height; float strength; float _pad; };
float ld(int y, int x, int c) {
    if (y < 0 || y >= height || x < 0 || x >= width) return 0.0;
    return rgb[(y * width + x) * 3 + c];
}
void main() {
    int y = int(gl_GlobalInvocationID.y);
    int x = int(gl_GlobalInvocationID.x);
    if (y >= height || x >= width) return;
    int idx = (y * width + x) * 3;
    for (int c = 0; c < 3; c++) {
        float center = ld(y, x, c);
        float edge = -ld(y-1,x,c) - ld(y,x-1,c) + 4.0*center - ld(y,x+1,c) - ld(y+1,x,c);
        outbuf[idx+c] = clamp(center + edge * strength, 0.0, 1.0);
    }
}
SHADER

echo "Shader sources created."

# Compile all to SPIR-V and generate C headers
for f in cs_bayer_to_rgb cs_blc_wb cs_ccm cs_tone cs_fcs cs_ldci cs_ee; do
    echo -n "Compiling $f... "
    glslangValidator --target-env vulkan1.2 ${f}.comp -o ${f}.spv 2>&1
    xxd -i ${f}.spv > ${f}_spv.h
    echo "$(wc -c < ${f}.spv) bytes SPIR-V, header generated"
done
echo "All shaders built."

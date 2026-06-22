# Makefile for Mesa GPU Compute Tests
#
# Tests:
#   mesa_offline_compute_test  - Vulkan compute shader benchmark (vector add + heavy FP)
#   mesa_heavy_pipeline_test   - Full HEAVY ISP pipeline (7-stage Vulkan compute)

CC      = gcc
CFLAGS  = -I$(PREFIX)/include -Wall -Wextra -O2
LDFLAGS = -L$(PREFIX)/lib
LIBS    = -lvulkan -lm -lpthread

# SPIR-V compilation
GLSLANG = glslangValidator

# All shader sources
SHADER_SRC  = cs_add.comp cs_heavy.comp \
              cs_bayer_to_rgb.comp cs_blc_wb.comp cs_ccm.comp \
              cs_tone.comp cs_ccm_tone.comp cs_fcs.comp cs_fcs_ldci_h.comp cs_ldci_h.comp cs_ldci_v.comp cs_ee.comp
SHADER_HEADERS = $(SHADER_SRC:.comp=_spv.h)

.PHONY: all clean shaders

all: mesa_offline_compute_test mesa_heavy_pipeline_test

# Build shader SPIR-V binaries and C headers
shaders: $(SHADER_HEADERS)

$(SHADER_HEADERS): %_spv.h: %.comp
	$(GLSLANG) --target-env vulkan1.2 $< -o $*.spv
	xxd -i $*.spv > $@

# Standalone compute benchmark
mesa_offline_compute_test: mesa_offline_compute_test.c cs_add_spv.h cs_heavy_spv.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

# Heavy ISP pipeline test
mesa_heavy_pipeline_test: mesa_heavy_pipeline_test.c $(SHADER_HEADERS) cs_ldci_h_spv.h cs_ldci_v_spv.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

clean:
	rm -f mesa_offline_compute_test mesa_heavy_pipeline_test *.spv *.o

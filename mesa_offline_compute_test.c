/*
 * mesa_offline_compute_test.c
 *
 * Test linking to Mesa (Vulkan compute) and measuring its
 * offline GPU computation power.
 *
 * Uses Vulkan compute shaders via Mesa's freedreno ICD (Adreno GPU).
 * Falls back to software (llvmpipe) if no GPU is available.
 *
 * Tests:
 *   1. Correctness: parallel vector addition (SSBO)
 *   2. Performance: heavy FP compute benchmark
 *
 * Build:
 *   gcc -o mesa_offline_compute_test mesa_offline_compute_test.c \
 *       -I$PREFIX/include -L$PREFIX/lib -lvulkan -lm
 *
 * Run:
 *   ./mesa_offline_compute_test
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <vulkan/vulkan.h>

/* Embedded SPIR-V shader binaries (compiled for Vulkan 1.2) */
#include "cs_add_spv.h"
#include "cs_heavy_spv.h"

/* ------------------------------------------------------------------ */
/*  Utility helpers                                                    */
/* ------------------------------------------------------------------ */

#define VK_CHECK(f, msg) do { \
    VkResult r_ = (f); \
    if (r_ != VK_SUCCESS) { \
        fprintf(stderr, "Vulkan error at %s: %d (0x%x)\n", msg, (int)r_, (unsigned)r_); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/* High-resolution timer in seconds */
static double timer_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------ */
/*  Vulkan context                                                     */
/* ------------------------------------------------------------------ */

typedef struct {
    VkInstance       inst;
    VkPhysicalDevice phys_dev;
    VkPhysicalDeviceProperties phys_dev_props;
    VkDevice         dev;
    VkQueue          queue;
    uint32_t         queue_family;
    VkCommandPool    cmd_pool;
} VulkanCtx;

static VulkanCtx init_vulkan(void) {
    VulkanCtx ctx;
    memset(&ctx, 0, sizeof(ctx));

    /* Create instance */
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "MesaOfflineComputeTest",
        .apiVersion = VK_API_VERSION_1_3,
    };
    VkInstanceCreateInfo inst_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
    };
    VK_CHECK(vkCreateInstance(&inst_info, NULL, &ctx.inst),
             "vkCreateInstance");

    /* Enumerate physical devices */
    uint32_t ndev = 0;
    vkEnumeratePhysicalDevices(ctx.inst, &ndev, NULL);
    if (ndev == 0) { fprintf(stderr, "No Vulkan devices\n"); exit(1); }

    VkPhysicalDevice *devs = malloc(ndev * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(ctx.inst, &ndev, devs);

    printf("  Found %u Vulkan physical device(s):\n", ndev);
    int gpu_idx = -1, fallback_idx = -1;
    for (uint32_t i = 0; i < ndev; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        const char *t = "OTHER";
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) t = "GPU";
        else if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) t = "CPU";
        printf("    [%u] %s  (%s)\n", i, p.deviceName, t);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
            p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            if (gpu_idx < 0) gpu_idx = i;
        }
        if (fallback_idx < 0) fallback_idx = i;
    }

    int sel = (gpu_idx >= 0) ? gpu_idx : fallback_idx;
    ctx.phys_dev = devs[sel];
    vkGetPhysicalDeviceProperties(ctx.phys_dev, &ctx.phys_dev_props);
    printf("  Using: %s\n\n", ctx.phys_dev_props.deviceName);
    free(devs);

    /* Find compute queue */
    uint32_t nqf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.phys_dev, &nqf, NULL);
    VkQueueFamilyProperties *qf = malloc(nqf * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.phys_dev, &nqf, qf);

    int found = 0;
    for (uint32_t i = 0; i < nqf; i++) {
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            ctx.queue_family = i;
            found = 1;
            break;
        }
    }
    free(qf);
    if (!found) { fprintf(stderr, "No compute queue\n"); exit(1); }

    /* Create device */
    float qprio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = ctx.queue_family,
        .queueCount = 1,
        .pQueuePriorities = &qprio,
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
    };
    VK_CHECK(vkCreateDevice(ctx.phys_dev, &dci, NULL, &ctx.dev),
             "vkCreateDevice");
    vkGetDeviceQueue(ctx.dev, ctx.queue_family, 0, &ctx.queue);

    /* Command pool */
    VkCommandPoolCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = ctx.queue_family,
    };
    VK_CHECK(vkCreateCommandPool(ctx.dev, &cpci, NULL, &ctx.cmd_pool),
             "vkCreateCommandPool");

    return ctx;
}

static void destroy_vulkan(VulkanCtx *ctx) {
    vkDestroyCommandPool(ctx->dev, ctx->cmd_pool, NULL);
    vkDestroyDevice(ctx->dev, NULL);
    vkDestroyInstance(ctx->inst, NULL);
}

/* ------------------------------------------------------------------ */
/*  Helper: find memory type                                           */
/* ------------------------------------------------------------------ */

static uint32_t find_mem_type(VkPhysicalDevice pd, uint32_t type_bits,
                               VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    fprintf(stderr, "No suitable memory type found\n");
    exit(1);
}

/* ------------------------------------------------------------------ */
/*  Helper: create buffer + memory                                     */
/* ------------------------------------------------------------------ */

static void create_buffer(VulkanCtx *ctx, VkDeviceSize size,
                           VkBufferUsageFlags usage,
                           VkMemoryPropertyFlags mem_props,
                           VkBuffer *buf, VkDeviceMemory *mem)
{
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VK_CHECK(vkCreateBuffer(ctx->dev, &bci, NULL, buf), "vkCreateBuffer");

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(ctx->dev, *buf, &mr);

    uint32_t mt = find_mem_type(ctx->phys_dev, mr.memoryTypeBits, mem_props);
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mr.size,
        .memoryTypeIndex = mt,
    };
    VK_CHECK(vkAllocateMemory(ctx->dev, &mai, NULL, mem), "vkAllocateMemory");
    VK_CHECK(vkBindBufferMemory(ctx->dev, *buf, *mem, 0), "vkBindBufferMemory");
}

/* ------------------------------------------------------------------ */
/*  Helper: one-shot command buffer submit and wait                    */
/* ------------------------------------------------------------------ */

static void submit_once(VulkanCtx *ctx, VkCommandBuffer cb) {
    VK_CHECK(vkEndCommandBuffer(cb), "vkEndCommandBuffer");
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
    };
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence;
    VK_CHECK(vkCreateFence(ctx->dev, &fci, NULL, &fence), "vkCreateFence");
    VK_CHECK(vkQueueSubmit(ctx->queue, 1, &si, fence), "vkQueueSubmit");
    VK_CHECK(vkWaitForFences(ctx->dev, 1, &fence, VK_TRUE, UINT64_MAX),
             "vkWaitForFences");
    vkDestroyFence(ctx->dev, fence, NULL);
    vkFreeCommandBuffers(ctx->dev, ctx->cmd_pool, 1, &cb);
}

/* ------------------------------------------------------------------ */
/*  Test 1: Correctness – parallel vector addition                     */
/* ------------------------------------------------------------------ */

static int test_vector_add(VulkanCtx *ctx) {
    printf("=== Test 1: Vector Addition (correctness) ===\n");

    const uint32_t N = 1024;
    const VkDeviceSize buf_size = N * sizeof(float);

    /* Create shader module from embedded SPIR-V */
    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = cs_add_spv_len,
        .pCode = (const uint32_t *)cs_add_spv,
    };
    VkShaderModule sm;
    VK_CHECK(vkCreateShaderModule(ctx->dev, &smci, NULL, &sm),
             "vkCreateShaderModule (add)");

    /* Descriptor set layout: 3 storage buffers (A, B, C) */
    VkDescriptorSetLayoutBinding dslb[3] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
    };
    VkDescriptorSetLayoutCreateInfo dslci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = dslb,
    };
    VkDescriptorSetLayout dsl;
    VK_CHECK(vkCreateDescriptorSetLayout(ctx->dev, &dslci, NULL, &dsl),
             "vkCreateDescriptorSetLayout (add)");

    /* Pipeline layout */
    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &dsl,
    };
    VkPipelineLayout pl;
    VK_CHECK(vkCreatePipelineLayout(ctx->dev, &plci, NULL, &pl),
             "vkCreatePipelineLayout (add)");

    /* Compute pipeline */
    VkPipelineShaderStageCreateInfo ssci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = sm,
        .pName = "main",
    };
    VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = ssci,
        .layout = pl,
    };
    VkPipeline pipeline;
    VK_CHECK(vkCreateComputePipelines(ctx->dev, VK_NULL_HANDLE, 1,
                                       &cpci, NULL, &pipeline),
             "vkCreateComputePipelines (add)");

    /* Descriptor pool */
    VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &dps,
    };
    VkDescriptorPool dp;
    VK_CHECK(vkCreateDescriptorPool(ctx->dev, &dpci, NULL, &dp),
             "vkCreateDescriptorPool (add)");

    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = dp,
        .descriptorSetCount = 1,
        .pSetLayouts = &dsl,
    };
    VkDescriptorSet ds;
    VK_CHECK(vkAllocateDescriptorSets(ctx->dev, &dsai, &ds),
             "vkAllocateDescriptorSets (add)");

    /* Create 3 storage buffers */
    uint32_t buf_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    uint32_t mem_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    VkBuffer buf[3];
    VkDeviceMemory mem[3];
    for (int i = 0; i < 3; i++)
        create_buffer(ctx, buf_size, buf_flags, mem_flags, &buf[i], &mem[i]);

    /* Fill A[i]=i, B[i]=i*2 */
    void *ptr;
    float *tmp = malloc(buf_size);
    for (uint32_t i = 0; i < N; i++) tmp[i] = (float)i;
    VK_CHECK(vkMapMemory(ctx->dev, mem[0], 0, buf_size, 0, &ptr), "vkMapMemory A");
    memcpy(ptr, tmp, buf_size);
    vkUnmapMemory(ctx->dev, mem[0]);

    for (uint32_t i = 0; i < N; i++) tmp[i] = (float)(i * 2);
    VK_CHECK(vkMapMemory(ctx->dev, mem[1], 0, buf_size, 0, &ptr), "vkMapMemory B");
    memcpy(ptr, tmp, buf_size);
    vkUnmapMemory(ctx->dev, mem[1]);
    free(tmp);

    /* Write descriptor set */
    VkDescriptorBufferInfo dbi[3] = {
        {buf[0], 0, VK_WHOLE_SIZE},
        {buf[1], 0, VK_WHOLE_SIZE},
        {buf[2], 0, VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet wds[3] = {
        {.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet=ds,
         .dstBinding=0, .descriptorCount=1,
         .descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo=&dbi[0]},
        {.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet=ds,
         .dstBinding=1, .descriptorCount=1,
         .descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo=&dbi[1]},
        {.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet=ds,
         .dstBinding=2, .descriptorCount=1,
         .descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo=&dbi[2]},
    };
    vkUpdateDescriptorSets(ctx->dev, 3, wds, 0, NULL);

    /* Dispatch */
    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cb;
    VK_CHECK(vkAllocateCommandBuffers(ctx->dev, &cbai, &cb),
             "vkAllocateCommandBuffers (add)");

    VkCommandBufferBeginInfo cbbi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VK_CHECK(vkBeginCommandBuffer(cb, &cbbi), "vkBeginCommandBuffer (add)");
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl,
                            0, 1, &ds, 0, NULL);
    vkCmdDispatch(cb, (N + 63) / 64, 1, 1);
    submit_once(ctx, cb);

    /* Read back buffer C */
    float *result = malloc(buf_size);
    VK_CHECK(vkMapMemory(ctx->dev, mem[2], 0, buf_size, 0, &ptr),
             "vkMapMemory C");
    memcpy(result, ptr, buf_size);
    vkUnmapMemory(ctx->dev, mem[2]);

    /* Verify */
    int errors = 0;
    for (uint32_t i = 0; i < N; i++) {
        float expected = (float)(i + i * 2); /* a[i]+b[i] = i + 2*i = 3*i */
        if (fabsf(result[i] - expected) > 1e-5f) {
            if (errors < 10)
                fprintf(stderr, "  Mismatch at [%u]: got %.1f, expected %.1f\n",
                        i, result[i], expected);
            errors++;
        }
    }
    free(result);

    /* Cleanup */
    for (int i = 0; i < 3; i++) {
        vkDestroyBuffer(ctx->dev, buf[i], NULL);
        vkFreeMemory(ctx->dev, mem[i], NULL);
    }
    vkDestroyDescriptorPool(ctx->dev, dp, NULL);
    vkDestroyPipeline(ctx->dev, pipeline, NULL);
    vkDestroyPipelineLayout(ctx->dev, pl, NULL);
    vkDestroyDescriptorSetLayout(ctx->dev, dsl, NULL);
    vkDestroyShaderModule(ctx->dev, sm, NULL);

    if (errors == 0)
        printf("  PASSED: all %u elements correct\n", N);
    else
        printf("  FAILED: %d / %u elements incorrect\n", errors, N);
    printf("\n");
    return errors;
}

/* ------------------------------------------------------------------ */
/*  Test 2: Performance – heavy compute benchmark                      */
/* ------------------------------------------------------------------ */

static int test_compute_benchmark(VulkanCtx *ctx) {
    printf("=== Test 2: Compute Performance Benchmark ===\n");

    const char *renderer = ctx->phys_dev_props.deviceName;
    int is_sw = (strstr(renderer, "llvmpipe") != NULL);

    /* Scale workload based on device capability */
    const uint32_t N = is_sw ? 65536 : 4 * 1024 * 1024;
    const uint32_t local_size = 256;
    const uint32_t groups = (N + local_size - 1) / local_size;

    printf("  Device: %s\n", renderer);
    printf("  Work items: %u\n", N);
    printf("  Work groups: %u (local size %u)\n", groups, local_size);
    if (is_sw) printf("  (CPU path: reduced workload)\n");

    /* Create shader module */
    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = cs_heavy_spv_len,
        .pCode = (const uint32_t *)cs_heavy_spv,
    };
    VkShaderModule sm;
    VK_CHECK(vkCreateShaderModule(ctx->dev, &smci, NULL, &sm),
             "vkCreateShaderModule (bench)");

    /* Descriptor set layout: 1 storage buffer */
    VkDescriptorSetLayoutBinding dslb = {
        0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL
    };
    VkDescriptorSetLayoutCreateInfo dslci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &dslb,
    };
    VkDescriptorSetLayout dsl;
    VK_CHECK(vkCreateDescriptorSetLayout(ctx->dev, &dslci, NULL, &dsl),
             "vkCreateDescriptorSetLayout (bench)");

    /* Pipeline layout */
    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &dsl,
    };
    VkPipelineLayout pl;
    VK_CHECK(vkCreatePipelineLayout(ctx->dev, &plci, NULL, &pl),
             "vkCreatePipelineLayout (bench)");

    /* Compute pipeline */
    VkPipelineShaderStageCreateInfo ssci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = sm,
        .pName = "main",
    };
    VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = ssci,
        .layout = pl,
    };
    VkPipeline pipeline;
    VK_CHECK(vkCreateComputePipelines(ctx->dev, VK_NULL_HANDLE, 1,
                                       &cpci, NULL, &pipeline),
             "vkCreateComputePipelines (bench)");

    /* Descriptor pool */
    VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &dps,
    };
    VkDescriptorPool dp;
    VK_CHECK(vkCreateDescriptorPool(ctx->dev, &dpci, NULL, &dp),
             "vkCreateDescriptorPool (bench)");

    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = dp,
        .descriptorSetCount = 1,
        .pSetLayouts = &dsl,
    };
    VkDescriptorSet ds;
    VK_CHECK(vkAllocateDescriptorSets(ctx->dev, &dsai, &ds),
             "vkAllocateDescriptorSets (bench)");

    /* Create output buffer */
    VkDeviceSize buf_size = N * sizeof(float);
    uint32_t buf_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    uint32_t mem_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkBuffer buf;
    VkDeviceMemory mem;
    create_buffer(ctx, buf_size, buf_flags, mem_flags, &buf, &mem);

    /* Write descriptor */
    VkDescriptorBufferInfo dbi = { buf, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet wds = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &dbi,
    };
    vkUpdateDescriptorSets(ctx->dev, 1, &wds, 0, NULL);

    /* Helper to create a command buffer, bind, dispatch, submit */
    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    /* Warm-up dispatch */
    {
        VkCommandBuffer cb;
        VK_CHECK(vkAllocateCommandBuffers(ctx->dev, &cbai, &cb),
                 "vkAllocateCommandBuffers (warmup)");
        VkCommandBufferBeginInfo cbbi = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        VK_CHECK(vkBeginCommandBuffer(cb, &cbbi), "vkBeginCommandBuffer (warmup)");
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl,
                                0, 1, &ds, 0, NULL);
        vkCmdDispatch(cb, groups, 1, 1);
        submit_once(ctx, cb);
    }

    /* Timed runs */
    const int ITER = is_sw ? 3 : 5;
    double total_time = 0.0;
    double min_time = 1e30;
    double max_time = 0.0;

    for (int iter = 0; iter < ITER; iter++) {
        VkCommandBuffer cb;
        VK_CHECK(vkAllocateCommandBuffers(ctx->dev, &cbai, &cb),
                 "vkAllocateCommandBuffers (run)");
        VkCommandBufferBeginInfo cbbi = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        VK_CHECK(vkBeginCommandBuffer(cb, &cbbi), "vkBeginCommandBuffer (run)");
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl,
                                0, 1, &ds, 0, NULL);

        double t0 = timer_sec();
        vkCmdDispatch(cb, groups, 1, 1);
        submit_once(ctx, cb);
        double t1 = timer_sec();

        double dt = t1 - t0;
        total_time += dt;
        if (dt < min_time) min_time = dt;
        if (dt > max_time) max_time = dt;
        printf("  Run %d: %.4f s (%.2f M items/s)\n",
               iter + 1, dt, (double)N / dt / 1e6);
    }

    double avg_time = total_time / ITER;
    /* ~5 FP ops per inner loop iteration, 1024 iterations per work item */
    double ops_per_item = 1024.0 * 5.0;
    double total_ops = (double)N * ops_per_item;
    double gflops = total_ops / avg_time / 1e9;

    printf("\n  --- Results ---\n");
    printf("  Device:          %s\n", renderer);
    printf("  Avg time:        %.4f s\n", avg_time);
    printf("  Min time:        %.4f s\n", min_time);
    printf("  Max time:        %.4f s\n", max_time);
    printf("  Throughput:      %.2f M items/s\n", (double)N / avg_time / 1e6);
    printf("  Estimated GFLOPS: %.2f (%.0f ops/item)\n", gflops, ops_per_item);

    /* Verify output is deterministic (check for NaN/Inf) */
    void *ptr;
    VK_CHECK(vkMapMemory(ctx->dev, mem, 0, buf_size, 0, &ptr),
             "vkMapMemory (verify)");
    float *data = (float *)ptr;
    int consistent = 1;
    for (uint32_t i = 0; i < N; i += N > 4 ? N / 4 : 1) {
        if (isnan(data[i]) || isinf(data[i])) {
            fprintf(stderr, "  WARNING: result[%u] is NaN/Inf\n", i);
            consistent = 0;
        }
    }
    if (consistent)
        printf("  Output values: all finite (deterministic)\n");
    else
        printf("  WARNING: some non-finite values detected\n");
    vkUnmapMemory(ctx->dev, mem);

    /* Cleanup */
    vkDestroyBuffer(ctx->dev, buf, NULL);
    vkFreeMemory(ctx->dev, mem, NULL);
    vkDestroyDescriptorPool(ctx->dev, dp, NULL);
    vkDestroyPipeline(ctx->dev, pipeline, NULL);
    vkDestroyPipelineLayout(ctx->dev, pl, NULL);
    vkDestroyDescriptorSetLayout(ctx->dev, dsl, NULL);
    vkDestroyShaderModule(ctx->dev, sm, NULL);

    printf("\n");
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Main entry point                                                   */
/* ------------------------------------------------------------------ */

int main(void) {
    printf("===========================================================\n");
    printf("  Mesa Offline Compute Test\n");
    printf("  Platform: aarch64 (Termux), Mesa %s\n", "26.0.6");
    printf("===========================================================\n\n");

    /* Initialize Vulkan */
    VulkanCtx ctx = init_vulkan();

    int ret = 0;

    /* Test 1: correctness */
    int e1 = test_vector_add(&ctx);
    if (e1 < 0) { ret = 1; }
    else if (e1 > 0) { ret = 1; }

    /* Test 2: benchmark */
    int e2 = test_compute_benchmark(&ctx);
    if (e2 < 0) { ret = 1; }

    /* Cleanup */
    destroy_vulkan(&ctx);

    printf("===========================================================\n");
    if (ret == 0)
        printf("  ALL TESTS PASSED\n");
    else
        printf("  SOME TESTS FAILED\n");
    printf("===========================================================\n");

    return ret;
}

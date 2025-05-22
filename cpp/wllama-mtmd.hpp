#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <stdio.h>
#include <cmath>

#include "llama.h"
#include "helpers/wcommon.h"
#include "helpers/wsampling.h"

// Include mtmd.h for multimodal support
#include "../llama.cpp/examples/llava/mtmd.h"

#include "glue.hpp"
#include "actions.hpp"

// Multimodal-specific action handlers
struct glue_msg_init_mtmd_req {
    std::string mmproj_path;
    bool use_gpu;
    int n_threads;
    std::string image_marker;
};

GLUE_HANDLER(glue_msg_init_mtmd_req, "imtm_req", mmproj_path, use_gpu, n_threads, image_marker);

struct glue_msg_init_mtmd_res {
    bool success;
    std::string error;
};

GLUE_HANDLER(glue_msg_init_mtmd_res, "imtm_res", success, error);

struct glue_msg_process_image_req {
    std::vector<uint8_t> image_data;
    int data_size;
    int width;
    int height;
    std::string prompt;
    bool use_cache;
};

GLUE_HANDLER(glue_msg_process_image_req, "proc_req", image_data, data_size, width, height, prompt, use_cache);

struct glue_msg_process_image_res {
    bool success;
    std::string error;
    std::string result;
};

GLUE_HANDLER(glue_msg_process_image_res, "proc_res", success, error, result);

// Global multimodal context
struct mtmd_context_wrapper {
    mtmd_context* ctx;
    bool initialized;
    
    mtmd_context_wrapper() : ctx(nullptr), initialized(false) {}
    
    ~mtmd_context_wrapper() {
        if (ctx) {
            mtmd_free(ctx);
            ctx = nullptr;
        }
        initialized = false;
    }
};

static mtmd_context_wrapper g_mtmd_ctx;

// Initialize multimodal context
action_handler<glue_msg_init_mtmd_res> action_init_mtmd(app_t& app, const char* req_raw) {
    PARSE_REQ(glue_msg_init_mtmd_req);
    
    glue_msg_init_mtmd_res res;
    res.success = false;
    
    if (!app.model) {
        res.error = "Text model not loaded. Call loadModel() first.";
        return action_handler<glue_msg_init_mtmd_res>(res);
    }
    
    // Cleanup existing context if any
    if (g_mtmd_ctx.ctx) {
        mtmd_free(g_mtmd_ctx.ctx);
        g_mtmd_ctx.ctx = nullptr;
        g_mtmd_ctx.initialized = false;
    }
    
    try {
        // Initialize multimodal context
        mtmd_context_params params;
        params.use_gpu = req.use_gpu;
        params.n_threads = req.n_threads > 0 ? req.n_threads : 1;
        params.verbosity = GGML_LOG_LEVEL_INFO;
        params.print_timings = true;
        
        if (!req.image_marker.empty()) {
            params.image_marker = req.image_marker.c_str();
        }
        
        g_mtmd_ctx.ctx = mtmd_init_from_file(req.mmproj_path.c_str(), app.model, params);
        
        if (!g_mtmd_ctx.ctx) {
            res.error = "Failed to initialize multimodal context";
            return action_handler<glue_msg_init_mtmd_res>(res);
        }
        
        g_mtmd_ctx.initialized = true;
        res.success = true;
    } catch (const std::exception& e) {
        res.error = std::string("Exception: ") + e.what();
        return action_handler<glue_msg_init_mtmd_res>(res);
    }
    
    return action_handler<glue_msg_init_mtmd_res>(res);
}

// Process an image
action_handler<glue_msg_process_image_res> action_process_image(app_t& app, const char* req_raw) {
    PARSE_REQ(glue_msg_process_image_req);
    
    glue_msg_process_image_res res;
    res.success = false;
    
    if (!app.model || !app.ctx) {
        res.error = "Text model not loaded. Call loadModel() first.";
        return action_handler<glue_msg_process_image_res>(res);
    }
    
    if (!g_mtmd_ctx.initialized || !g_mtmd_ctx.ctx) {
        res.error = "Multimodal context not initialized. Call initMultimodal() first.";
        return action_handler<glue_msg_process_image_res>(res);
    }
    
    if (req.image_data.empty() || req.width <= 0 || req.height <= 0) {
        res.error = "Invalid image data or dimensions";
        return action_handler<glue_msg_process_image_res>(res);
    }
    
    try {
        // Create a bitmap from the image data
        mtmd_bitmap bitmap;
        bitmap.nx = req.width;
        bitmap.ny = req.height;
        bitmap.data.resize(req.width * req.height * 3);
        
        // Copy image data
        if (req.image_data.size() >= req.width * req.height * 3) {
            std::memcpy(bitmap.data.data(), req.image_data.data(), req.width * req.height * 3);
        } else {
            res.error = "Image data size does not match dimensions";
            return action_handler<glue_msg_process_image_res>(res);
        }
        
        // Create a unique ID for the bitmap for caching
        std::string image_id = "img_" + std::to_string(std::hash<std::string>()(std::string((char*)req.image_data.data(), req.image_data.size())));
        bitmap.id = image_id;
        
        // Format the text with image marker
        std::string formatted_prompt = req.prompt;
        std::string image_marker = "<__image__>"; // Default image marker
        if (formatted_prompt.find(image_marker) == std::string::npos) {
            formatted_prompt += " " + image_marker;
        }
        
        // Create input text
        mtmd_input_text text;
        text.text = formatted_prompt;
        text.add_special = true;
        text.parse_special = true;
        
        // Tokenize the input
        std::vector<mtmd_bitmap> bitmaps;
        bitmaps.push_back(bitmap);
        
        mtmd_input_chunks chunks;
        int32_t tokenize_result = mtmd_tokenize(g_mtmd_ctx.ctx, chunks, text, bitmaps);
        
        if (tokenize_result != 0) {
            res.error = "Failed to tokenize input with image";
            return action_handler<glue_msg_process_image_res>(res);
        }
        
        // Clear KV cache if not using cache
        if (!req.use_cache) {
            llama_kv_cache_clear(app.ctx);
        }
        
        // Process the chunks
        llama_pos n_past = 0;
        int32_t n_batch = llama_n_batch(app.ctx);
        
        int32_t eval_result = mtmd_helper_eval(
            g_mtmd_ctx.ctx,
            app.ctx,
            chunks,
            n_past,
            0, // seq_id
            n_batch
        );
        
        if (eval_result != 0) {
            res.error = "Failed to evaluate chunks";
            return action_handler<glue_msg_process_image_res>(res);
        }
        
        // Update n_past
        n_past = n_past + mtmd_helper_get_n_tokens(chunks);
        
        // Sampling parameters
        wcommon_sampler_params sparams;
        sparams.temp = 0.7f;
        sparams.top_k = 40;
        sparams.top_p = 0.9f;
        sparams.n_prev = 64;
        
        // Initialize sampling context
        if (app.ctx_sampling) {
            wcommon_sampler_free(app.ctx_sampling);
        }
        app.ctx_sampling = wcommon_sampler_init(app.model, sparams);
        
        // Generate response (similar to completion generation)
        std::string generated_text;
        const int max_tokens = 1024; // Arbitrary limit
        
        for (int i = 0; i < max_tokens; i++) {
            // Sample next token
            llama_token id = wcommon_sampler_sample(app.ctx_sampling, app.ctx, n_past);
            
            if (id == llama_token_eos(app.model)) {
                break; // End of text
            }
            
            // Get token string
            std::string token_str = llama_token_to_piece(app.ctx, id);
            generated_text += token_str;
            
            // Update sampling context
            wcommon_sampler_accept(app.ctx_sampling, id);
            
            // Evaluate the new token
            llama_batch_clear(app.batch);
            llama_batch_add(app.batch, id, n_past, {0}, true);
            
            if (llama_decode(app.ctx, app.batch)) {
                res.error = "Failed to decode token";
                return action_handler<glue_msg_process_image_res>(res);
            }
            
            n_past += 1;
        }
        
        // Return the generated text
        res.success = true;
        res.result = generated_text;
        
    } catch (const std::exception& e) {
        res.error = std::string("Exception: ") + e.what();
    } catch (...) {
        res.error = "Unknown error occurred during image processing";
    }
    
    return action_handler<glue_msg_process_image_res>(res);
}

// Define the macro for including multimodal actions in wllama.cpp
#define WLLAMA_MTMD_ACTIONS                           \
  WLLAMA_ACTION(init_mtmd)                            \
  WLLAMA_ACTION(process_image)
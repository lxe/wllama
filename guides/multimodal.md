# Multimodal Support in wllama

This guide covers how to use the multimodal capabilities in wllama, allowing you to process images alongside text using large language models.

## Overview

Multimodal support enables wllama to work with both text and images, making it possible to:

- Process images and generate text descriptions
- Answer questions about image content
- Extract text from documents using OCR-like capabilities
- Analyze visual content in detail

## Getting Started

### Prerequisites

To use multimodal features, you need:

1. A compatible multimodal model (e.g., InternVL, Gemma 3, Qwen-VL)
2. The base text model (.gguf file)
3. The multimodal projector model (.gguf file, often named with a "mmproj" prefix)

### Installation

Multimodal support is included in wllama v2.4.0 and above. No additional installation steps are required beyond the standard wllama setup.

## Basic Usage

Here's a simple example of how to use multimodal capabilities:

```javascript
import { Wllama, ModelManager, createMultimodal } from '@wllama/wllama';

// Initialize wllama and model manager
const wllama = new Wllama({
  'single-thread/wllama.wasm': '/path/to/single-thread/wllama.wasm',
  'multi-thread/wllama.wasm': '/path/to/multi-thread/wllama.wasm'
});
const modelManager = new ModelManager();

// Load the text model
const model = await modelManager.downloadModel('https://example.com/model.gguf');
await wllama.loadModel(model);

// Create multimodal instance
const multimodal = createMultimodal(wllama);

// Initialize with multimodal projector model
const mmproj = await modelManager.downloadModel('https://example.com/mmproj.gguf');
await multimodal.initMultimodal(mmproj, {
  useGpu: true, // Use GPU acceleration if available
  nThreads: 4    // Number of threads to use
});

// Process an image
const imageData = {
  width: 640,
  height: 480,
  data: new Uint8Array(...) // RGB pixel data (width * height * 3 bytes)
};

const result = await multimodal.processImage(imageData, {
  prompt: "Describe this image in detail",
  useCache: false
});

console.log(result); // The model's description of the image
```

## Converting Images to the Required Format

To process images, you need to convert them to the format expected by the multimodal API:

```javascript
// From a canvas or ImageData object
function imageDataToMultimodalFormat(imageData) {
  const result = {
    width: imageData.width,
    height: imageData.height,
    data: new Uint8Array(imageData.width * imageData.height * 3)
  };
  
  // Convert RGBA to RGB
  for (let i = 0; i < imageData.width * imageData.height; i++) {
    result.data[i * 3] = imageData.data[i * 4];     // R
    result.data[i * 3 + 1] = imageData.data[i * 4 + 1]; // G
    result.data[i * 3 + 2] = imageData.data[i * 4 + 2]; // B
  }
  
  return result;
}

// Example usage with a canvas
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');
ctx.drawImage(imageElement, 0, 0);
const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
const processedImage = imageDataToMultimodalFormat(imageData);
```

## API Reference

### `createMultimodal(wllama)`

Creates a new multimodal instance from an existing Wllama instance.

**Parameters:**
- `wllama`: An initialized Wllama instance

**Returns:**
- A new WllamaMultimodal instance

### `WllamaMultimodal.initMultimodal(mmproj, options)`

Initializes multimodal functionality with the provided projector model.

**Parameters:**
- `mmproj`: Path to or instance of a multimodal projector model
- `options`: Configuration options (optional)
  - `useGpu`: Whether to use GPU acceleration (default: true)
  - `nThreads`: Number of threads to use for processing (default: 1)
  - `imageMarker`: Custom image marker (default: "<__image__>")

**Returns:**
- Promise that resolves when initialization is complete

### `WllamaMultimodal.processImage(imageData, options)`

Processes an image and generates a text description.

**Parameters:**
- `imageData`: Object containing image data
  - `width`: Image width in pixels
  - `height`: Image height in pixels
  - `data`: Uint8Array of RGB pixel data (width * height * 3 bytes)
- `options`: Processing options
  - `prompt`: Text prompt to include with the image (default: "Describe this image.")
  - `useCache`: Whether to use KV cache (default: false)

**Returns:**
- Promise that resolves to the generated text description

### `WllamaMultimodal.isMultimodalReady()`

Checks if multimodal functionality is initialized and ready to use.

**Returns:**
- Boolean indicating whether multimodal is ready

## Supported Models

The multimodal functionality supports various models, including:

- InternVL (1B, 2B, 8B, 14B)
- Gemma 3 (4B, 12B, 27B)
- Qwen-VL (2B, 7B)
- Qwen 2.5 VL (3B, 7B, 32B, 72B)
- SmolVLM (256M, 500M, 2.2B)
- Pixtral (12B)
- Mistral Small (24B)
- LLaVA (various versions)

## Troubleshooting

### Common Issues

1. **"Multimodal functionality not initialized"**: Ensure you've called `initMultimodal()` before attempting to process images.

2. **"Text model not loaded"**: You must load the base text model with `wllama.loadModel()` before initializing multimodal functionality.

3. **"Failed to process image"**: Check that your image data is in the correct format (RGB, not RGBA) and dimensions are accurate.

4. **"Context cache overflow"**: Reduce the complexity of your prompt or increase the context size when loading the model.

### Performance Tips

- For better performance with large images, consider resizing them before processing
- Use `useGpu: true` when available for faster image processing
- Set an appropriate number of threads based on your hardware
- Clear the KV cache between processing multiple images to prevent memory issues

## Advanced Usage

### Processing Multiple Images

```javascript
// Process multiple images sequentially
async function processImages(images, prompt) {
  const results = [];
  
  for (const image of images) {
    await multimodal.processImage(image, { prompt });
    results.push(result);
  }
  
  return results;
}
```

### Custom Image Markers

Some models use specific image markers. You can customize this when initializing:

```javascript
await multimodal.initMultimodal(mmproj, {
  imageMarker: "<image>" // Custom marker for your model
});
```

## Implementation Details

The multimodal functionality in wllama is implemented on top of llama.cpp's mtmd (multimodal) library. It integrates with the existing wllama architecture to provide a seamless experience for processing both text and images.

The implementation includes:

- C++ bindings to llama.cpp's mtmd library
- TypeScript interfaces for working with images
- Utility functions for image processing
- Integration with the existing wllama text processing pipeline
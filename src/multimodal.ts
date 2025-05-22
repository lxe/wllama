import { Wllama, WllamaError } from './wllama';
import { ProxyToWorker } from './worker';

/**
 * Options for initializing multimodal functionality
 */
export interface MultimodalOptions {
  /**
   * Whether to use GPU for mmproj model. Default: true
   */
  useGpu?: boolean;
  
  /**
   * Number of threads to use for processing. Default: 1
   */
  nThreads?: number;
  
  /**
   * Custom image marker. Default: "<__image__>"
   */
  imageMarker?: string;
}

/**
 * Options for processing an image
 */
export interface ProcessImageOptions {
  /**
   * Text prompt to include with the image
   */
  prompt: string;
  
  /**
   * Whether to use the KV cache. Default: false
   */
  useCache?: boolean;
}

/**
 * Represents an image in RGB format
 */
export interface ImageData {
  /**
   * Width of the image
   */
  width: number;
  
  /**
   * Height of the image
   */
  height: number;
  
  /**
   * RGB pixel data (width * height * 3 bytes)
   */
  data: Uint8Array;
}

/**
 * A class that extends Wllama with multimodal capabilities
 */
export class WllamaMultimodal {
  private wllama: Wllama;
  private isMultimodalEnabled: boolean = false;
  
  /**
   * Creates a new WllamaMultimodal instance
   * @param wllama An existing Wllama instance
   */
  constructor(wllama: Wllama) {
    this.wllama = wllama;
  }
  
  /**
   * Initialize multimodal functionality
   * @param mmproj Path to the multimodal projector model
   * @param options Additional options
   */
  async initMultimodal(mmproj: string, options: MultimodalOptions = {}): Promise<void> {
    if (!this.wllama.isModelLoaded()) {
      throw new WllamaError('Text model not loaded. Call loadModel() first.', 'model_not_loaded');
    }
    
    const result = await this.wllama.proxy.wllamaAction('init_mtmd', {
      _name: 'imtm_req',
      mmproj_path: mmproj,
      use_gpu: options.useGpu !== undefined ? options.useGpu : true,
      n_threads: options.nThreads || 1,
      image_marker: options.imageMarker || '<__image__>'
    });
    
    if (!result.success) {
      throw new WllamaError(`Failed to initialize multimodal: ${result.error}`, 'unknown_error');
    }
    
    this.isMultimodalEnabled = true;
  }
  
  /**
   * Process an image and generate a text description
   * @param image The image data to process
   * @param options Additional options
   * @returns The generated text description
   */
  async processImage(image: ImageData, options: ProcessImageOptions): Promise<string> {
    if (!this.isMultimodalEnabled) {
      throw new WllamaError('Multimodal functionality not initialized. Call initMultimodal() first.', 'unknown_error');
    }
    
    // Convert image data to raw bytes for transmission
    const imgData = Array.from(image.data);
    
    const result = await this.wllama.proxy.wllamaAction('process_image', {
      _name: 'proc_req',
      image_data: imgData,
      data_size: imgData.length,
      width: image.width,
      height: image.height,
      prompt: options.prompt || 'Describe this image.',
      use_cache: options.useCache || false
    });
    
    if (!result.success) {
      throw new WllamaError(`Failed to process image: ${result.error}`, 'inference_error');
    }
    
    return result.result;
  }
  
  /**
   * Check if multimodal functionality is enabled
   * @returns true if multimodal is enabled
   */
  isMultimodalReady(): boolean {
    return this.isMultimodalEnabled;
  }
}

/**
 * Create a new WllamaMultimodal instance from an existing Wllama instance
 * @param wllama Existing Wllama instance
 * @returns New WllamaMultimodal instance
 */
export function createMultimodal(wllama: Wllama): WllamaMultimodal {
  return new WllamaMultimodal(wllama);
}
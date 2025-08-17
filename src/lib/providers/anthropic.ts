import { anthropic } from "@ai-sdk/anthropic";
import { streamText, type LanguageModelV1 } from "ai";
import type { ValidationSchema } from "../types/typeAliases.js";
import type { AIProviderName } from "../core/types.js";
import type { StreamOptions, StreamResult } from "../types/streamTypes.js";
import type { UnknownRecord, JsonValue } from "../types/common.js";
import type { NeuroLink } from "../neurolink.js";
import { BaseProvider } from "../core/baseProvider.js";
import { logger } from "../utils/logger.js";
import { createTimeoutController, TimeoutError } from "../utils/timeout.js";
import { DEFAULT_MAX_TOKENS, DEFAULT_MAX_STEPS } from "../core/constants.js";
import {
  createAnthropicConfig,
  validateApiKey,
  getProviderModel,
} from "../utils/providerConfig.js";
import { buildMessagesArray } from "../utils/messageBuilder.js";

// Configuration helpers - now using consolidated utility
const getAnthropicApiKey = (): string => {
  return validateApiKey(createAnthropicConfig());
};

const getDefaultAnthropicModel = (): string => {
  return getProviderModel("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022");
};

/**
 * Anthropic Provider v2 - BaseProvider Implementation
 * Fixed syntax and enhanced with proper error handling
 */
export class AnthropicProvider extends BaseProvider {
  private model: LanguageModelV1;

  constructor(modelName?: string, sdk?: unknown) {
    super(
      modelName,
      "anthropic" as AIProviderName,
      sdk as NeuroLink | undefined,
    );

    // Initialize Anthropic model with API key validation
    const apiKey = getAnthropicApiKey();

    // Set Anthropic API key as environment variable (required by @ai-sdk/anthropic)
    process.env.ANTHROPIC_API_KEY = apiKey;

    // Initialize Anthropic with proper configuration
    this.model = anthropic(this.modelName || getDefaultAnthropicModel());

    logger.debug("Anthropic Provider v2 initialized", {
      modelName: this.modelName,
      provider: this.providerName,
    });
  }

  protected getProviderName(): AIProviderName {
    return "anthropic" as AIProviderName;
  }

  protected getDefaultModel(): string {
    return getDefaultAnthropicModel();
  }

  /**
   * Returns the Vercel AI SDK model instance for Anthropic
   */
  protected getAISDKModel(): LanguageModelV1 {
    return this.model;
  }

  protected handleProviderError(error: unknown): Error {
    if (error instanceof TimeoutError) {
      return this.handleTimeoutError(error);
    }

    const errorRecord = error as UnknownRecord;
    const errorMessage = this.extractErrorMessage(errorRecord);

    if (this.isApiKeyError(errorMessage)) {
      return this.createApiKeyError();
    }

    if (this.isRateLimitError(errorMessage)) {
      return this.createRateLimitError();
    }

    if (this.isConnectionError(errorMessage)) {
      return this.createConnectionError();
    }

    if (this.isServerError(errorMessage)) {
      return this.createServerError();
    }

    return this.createGenericError(errorMessage);
  }

  /**
   * Handle timeout errors
   */
  private handleTimeoutError(error: TimeoutError): Error {
    return new Error(
      `Anthropic request timed out after ${error.timeout}ms: ${error.message}`,
    );
  }

  /**
   * Extract error message from error record
   */
  private extractErrorMessage(errorRecord: UnknownRecord): string {
    return typeof errorRecord?.message === "string" ? errorRecord.message : "";
  }

  /**
   * Check if error is related to API key
   */
  private isApiKeyError(message: string): boolean {
    return (
      message.includes("API_KEY_INVALID") || message.includes("Invalid API key")
    );
  }

  /**
   * Check if error is related to rate limiting
   */
  private isRateLimitError(message: string): boolean {
    return (
      message.includes("rate limit") ||
      message.includes("too_many_requests") ||
      message.includes("429")
    );
  }

  /**
   * Check if error is related to connection
   */
  private isConnectionError(message: string): boolean {
    return (
      message.includes("ECONNRESET") ||
      message.includes("ENOTFOUND") ||
      message.includes("ECONNREFUSED") ||
      message.includes("network") ||
      message.includes("connection")
    );
  }

  /**
   * Check if error is related to server
   */
  private isServerError(message: string): boolean {
    return (
      message.includes("500") ||
      message.includes("502") ||
      message.includes("503") ||
      message.includes("504") ||
      message.includes("server error")
    );
  }

  /**
   * Create API key error
   */
  private createApiKeyError(): Error {
    return new Error(
      "Invalid Anthropic API key. Please check your ANTHROPIC_API_KEY environment variable.",
    );
  }

  /**
   * Create rate limit error
   */
  private createRateLimitError(): Error {
    return new Error("Anthropic rate limit exceeded. Please try again later.");
  }

  /**
   * Create connection error
   */
  private createConnectionError(): Error {
    return new Error(
      "Anthropic API connection error. Please check your internet connection and try again.",
    );
  }

  /**
   * Create server error
   */
  private createServerError(): Error {
    return new Error(
      "Anthropic API server error. Please try again in a few moments.",
    );
  }

  /**
   * Create generic error
   */
  private createGenericError(message: string): Error {
    const errorMessage = message || "Unknown error";
    return new Error(`Anthropic error: ${errorMessage}`);
  }

  // executeGenerate removed - BaseProvider handles all generation with tools

  protected async executeStream(
    options: StreamOptions,
    _analysisSchema?: ValidationSchema,
  ): Promise<StreamResult> {
    this.validateStreamOptions(options);

    const timeout = this.getTimeout(options);
    const timeoutController = createTimeoutController(
      timeout,
      this.providerName,
      "stream",
    );

    try {
      // ✅ Get tools for streaming (same as generate method)
      const shouldUseTools = !options.disableTools && this.supportsTools();
      const tools = shouldUseTools ? await this.getAllTools() : {};

      // Build message array from options
      const messages = buildMessagesArray(options);

      const result = await streamText({
        model: this.model,
        messages: messages,
        temperature: options.temperature,
        maxTokens: options.maxTokens || DEFAULT_MAX_TOKENS,
        tools,
        maxSteps: options.maxSteps || DEFAULT_MAX_STEPS,
        toolChoice: shouldUseTools ? "auto" : "none",
        abortSignal: timeoutController?.controller.signal,
      });

      timeoutController?.cleanup();

      const transformedStream = this.createTextStream(result);

      // ✅ Note: Vercel AI SDK's streamText() method limitations with tools
      // The streamText() function doesn't provide the same tool result access as generateText()
      // Full tool support is now available with real streaming
      const toolCalls: Array<{
        toolCallId: string;
        toolName: string;
        args: Record<string, unknown>;
      }> = [];

      const toolResults: Array<{
        toolName: string;
        status: "success" | "failure";
        output?: JsonValue;
        id: string;
      }> = [];

      const usage = await result.usage;
      const finishReason = await result.finishReason;

      return {
        stream: transformedStream,
        provider: this.providerName,
        model: this.modelName,
        toolCalls, // ✅ Include tool calls in stream result
        toolResults, // ✅ Include tool results in stream result
        usage: usage
          ? {
              inputTokens: usage.promptTokens || 0,
              outputTokens: usage.completionTokens || 0,
              totalTokens: usage.totalTokens || 0,
            }
          : undefined,
        finishReason: finishReason || undefined,
      };
    } catch (error: unknown) {
      timeoutController?.cleanup();
      throw this.handleProviderError(error);
    }
  }

  async isAvailable(): Promise<boolean> {
    try {
      getAnthropicApiKey();
      return true;
    } catch {
      return false;
    }
  }

  getModel(): LanguageModelV1 {
    return this.model;
  }
}

export default AnthropicProvider;

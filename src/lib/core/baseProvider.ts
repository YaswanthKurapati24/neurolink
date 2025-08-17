import { z } from "zod";
import type {
  ZodUnknownSchema,
  ValidationSchema,
  StandardRecord,
} from "../types/typeAliases.js";
import type { Tool, LanguageModelV1 } from "ai";
import type {
  AIProvider,
  TextGenerationOptions,
  TextGenerationResult,
  EnhancedGenerateResult,
  AnalyticsData,
  AIProviderName,
  EvaluationData,
} from "../core/types.js";
import type { StreamOptions, StreamResult } from "../types/streamTypes.js";
import type { JsonValue, JsonObject, UnknownRecord } from "../types/common.js";
import type { ToolDefinition, ToolResult, ToolArgs } from "../types/tools.js";
import { logger } from "../utils/logger.js";
import { DEFAULT_MAX_STEPS, STEP_LIMITS } from "../core/constants.js";
import { directAgentTools } from "../agent/directTools.js";
import { getSafeMaxTokens } from "../utils/tokenLimits.js";
import { createTimeoutController, TimeoutError } from "../utils/timeout.js";
import { shouldDisableBuiltinTools } from "../utils/toolUtils.js";
import { buildMessagesArray } from "../utils/messageBuilder.js";
import type { GenerateResult } from "../types/generateTypes.js";
import type { NeuroLink } from "../neurolink.js";
import type { ExternalMCPToolInfo } from "../types/externalMcp.js";
import { getKeysAsString, getKeyCount } from "../utils/transformationUtils.js";
import {
  validateStreamOptions as validateStreamOpts,
  validateTextGenerationOptions,
  ValidationError,
  createValidationSummary,
} from "../utils/parameterValidation.js";

// Union type for tools that can be either AI SDK tools or external MCP tools
type ExtendedTool = Tool & Partial<ExternalMCPToolInfo>;

// Interface for AI SDK generate result with steps (extends GenerateResult)
interface AISDKGenerateResult extends GenerateResult {
  steps?: Array<{
    toolCalls?: Array<{
      toolName?: string;
      name?: string;
      [key: string]: unknown;
    }>;
    [key: string]: unknown;
  }>;
  [key: string]: unknown;
}

/**
 * Abstract base class for all AI providers
 * Tools are integrated as first-class citizens - always available by default
 */
export abstract class BaseProvider implements AIProvider {
  protected readonly modelName: string;
  protected readonly providerName: AIProviderName;
  protected readonly defaultTimeout: number = 30000; // 30 seconds

  // Tools are conditionally included based on centralized configuration
  protected readonly directTools = shouldDisableBuiltinTools()
    ? {}
    : directAgentTools;
  protected mcpTools?: Record<string, Tool>; // MCP tools loaded dynamically when available
  protected customTools?: Map<string, unknown>; // Custom tools from registerTool()
  protected toolExecutor?: (
    toolName: string,
    params: unknown,
  ) => Promise<unknown>; // Tool executor from setupToolExecutor
  protected sessionId?: string;
  protected userId?: string;
  protected neurolink?: NeuroLink; // Reference to actual NeuroLink instance for MCP tools

  constructor(
    modelName?: string,
    providerName?: AIProviderName,
    neurolink?: NeuroLink,
  ) {
    this.modelName = modelName || this.getDefaultModel();
    this.providerName = providerName || this.getProviderName();
    this.neurolink = neurolink;
  }

  /**
   * Check if this provider supports tool/function calling
   * Override in subclasses to disable tools for specific providers or models
   * @returns true by default, providers can override to return false
   */
  supportsTools(): boolean {
    return true;
  }

  // ===================
  // PUBLIC API METHODS
  // ===================

  /**
   * Primary streaming method - implements AIProvider interface
   * When tools are involved, falls back to generate() with synthetic streaming
   */
  async stream(
    optionsOrPrompt: StreamOptions | string,
    analysisSchema?: ValidationSchema,
  ): Promise<StreamResult> {
    const options = this.normalizeStreamOptions(optionsOrPrompt);

    // CRITICAL FIX: Always prefer real streaming over fake streaming
    // Try real streaming first, use fake streaming only as fallback
    try {
      const realStreamResult = await this.executeStream(
        options,
        analysisSchema,
      );

      // If real streaming succeeds, return it (with tools support via Vercel AI SDK)
      return realStreamResult;
    } catch (realStreamError) {
      logger.warn(
        `Real streaming failed for ${this.providerName}, falling back to fake streaming:`,
        realStreamError,
      );

      // Fallback to fake streaming only if real streaming fails AND tools are enabled
      if (!options.disableTools && this.supportsTools()) {
        try {
          // Convert stream options to text generation options
          const textOptions: TextGenerationOptions = {
            prompt: options.input?.text || "",
            systemPrompt: options.systemPrompt,
            temperature: options.temperature,
            maxTokens: options.maxTokens,
            disableTools: false,
            maxSteps: options.maxSteps || 5,
            provider: options.provider as AIProviderName | undefined,
            model: options.model,
            // 🔧 FIX: Include analytics and evaluation options from stream options
            enableAnalytics: options.enableAnalytics,
            enableEvaluation: options.enableEvaluation,
            evaluationDomain: options.evaluationDomain,
            toolUsageContext: options.toolUsageContext,
            context: options.context as Record<string, JsonValue> | undefined,
          };

          const result = await this.generate(textOptions, analysisSchema);

          // Create a synthetic stream from the generate result that simulates progressive delivery
          return {
            stream: (async function* (): AsyncGenerator<{ content: string }> {
              if (result?.content) {
                // Split content into words for more natural streaming
                const words = result.content.split(/(\s+)/); // Keep whitespace
                let buffer = "";

                for (let i = 0; i < words.length; i++) {
                  buffer += words[i];

                  // Yield chunks of roughly 5-10 words or at punctuation
                  const shouldYield =
                    i === words.length - 1 || // Last word
                    buffer.length > 50 || // Buffer getting long
                    /[.!?;,]\s*$/.test(buffer); // End of sentence/clause

                  if (shouldYield && buffer.trim()) {
                    yield { content: buffer };
                    buffer = "";

                    // Small delay to simulate streaming (1-10ms)
                    await new Promise((resolve) =>
                      setTimeout(resolve, Math.random() * 9 + 1),
                    );
                  }
                }

                // Yield all remaining content
                if (buffer.trim()) {
                  yield { content: buffer };
                }
              }
            })(),
            usage: result?.usage,
            provider: result?.provider,
            model: result?.model,
            toolCalls: result?.toolCalls?.map((call) => ({
              toolName: call.toolName,
              parameters: call.args,
              id: call.toolCallId,
            })),
            toolResults: result?.toolResults
              ? result.toolResults.map((tr) => ({
                  toolName:
                    ((tr as UnknownRecord).toolName as string) || "unknown",
                  status: (((tr as UnknownRecord).status as string) === "error"
                    ? "failure"
                    : "success") as "success" | "failure",
                  result: (tr as UnknownRecord).result,
                  error: (tr as UnknownRecord).error as string | undefined,
                }))
              : undefined,
            // 🔧 FIX: Include analytics and evaluation from generate result
            analytics: result?.analytics,
            evaluation: result?.evaluation,
          };
        } catch (error) {
          logger.error(
            `Fake streaming fallback failed for ${this.providerName}:`,
            error,
          );
          throw this.handleProviderError(error);
        }
      } else {
        // If real streaming failed and no tools are enabled, re-throw the original error
        logger.error(
          `Real streaming failed for ${this.providerName}:`,
          realStreamError,
        );
        throw this.handleProviderError(realStreamError);
      }
    }
  }

  /**
   * Text generation method - implements AIProvider interface
   * Tools are always available unless explicitly disabled
   */
  async generate(
    optionsOrPrompt: TextGenerationOptions | string,
    _analysisSchema?: ValidationSchema,
  ): Promise<EnhancedGenerateResult | null> {
    const options = this.normalizeTextOptions(optionsOrPrompt);
    this.validateOptions(options);
    const startTime = Date.now();

    try {
      const tools = await this.setupToolsForGeneration(options);
      const result = await this.executeAIGeneration(options, tools);
      const enhancedResult = await this.processGenerationResult(
        result,
        options,
        tools,
      );
      return await this.enhanceResult(enhancedResult, options, startTime);
    } catch (error) {
      logger.error(`Generate failed for ${this.providerName}:`, error);
      throw this.handleProviderError(error);
    }
  }

  /**
   * Set up tools for AI generation
   */
  private async setupToolsForGeneration(
    options: TextGenerationOptions,
  ): Promise<Record<string, Tool>> {
    const shouldUseTools = !options.disableTools && this.supportsTools();
    if (!shouldUseTools) {
      return {};
    }

    const baseTools = await this.getAllTools();
    const tools = { ...baseTools, ...(options.tools || {}) };

    logger.debug(`[BaseProvider.generate] Tools for ${this.providerName}:`, {
      directTools: getKeyCount(baseTools),
      directToolNames: getKeysAsString(baseTools),
      externalTools: getKeyCount(options.tools || {}),
      externalToolNames: getKeysAsString(options.tools || {}),
      totalTools: getKeyCount(tools),
      totalToolNames: getKeysAsString(tools),
    });

    return tools;
  }

  /**
   * Execute AI text generation using Vercel AI SDK
   */
  private async executeAIGeneration(
    options: TextGenerationOptions,
    tools: Record<string, Tool>,
  ): Promise<AISDKGenerateResult> {
    const { generateText } = await import("ai");
    const model = await this.getAISDKModel();
    const messages = buildMessagesArray(options);

    const result = await generateText({
      model,
      messages,
      tools,
      maxSteps: options.maxSteps || DEFAULT_MAX_STEPS,
      toolChoice: Object.keys(tools).length > 0 ? "auto" : "none",
      temperature: options.temperature,
      maxTokens: options.maxTokens || 8192,
    });

    // Transform AI SDK result to our expected format
    return {
      content: result.text,
      provider: this.providerName,
      model: options.model || "unknown",
      usage: result.usage
        ? {
            promptTokens: result.usage.promptTokens,
            completionTokens: result.usage.completionTokens,
            totalTokens: result.usage.totalTokens,
          }
        : undefined,
      steps: result.steps,
      // Spread other properties but let our specific ones override
      text: result.text,
      reasoning: result.reasoning,
      files: result.files,
      reasoningDetails: result.reasoningDetails,
      finishReason: result.finishReason,
      warnings: result.warnings,
      logprobs: result.logprobs,
      request: result.request,
      response: result.response,
      toolCalls: result.toolCalls,
      toolResults: result.toolResults,
      experimental_providerMetadata: result.experimental_providerMetadata,
    } as AISDKGenerateResult;
  }

  /**
   * Process AI generation result and extract tool information
   */
  private async processGenerationResult(
    result: AISDKGenerateResult,
    options: TextGenerationOptions,
    tools: Record<string, Tool>,
  ): Promise<EnhancedGenerateResult> {
    const toolsUsed = this.extractToolsUsed(result);
    const toolExecutions = this.extractToolExecutions(result);

    return {
      content: (result as { text?: string }).text || "No content",
      usage: {
        inputTokens:
          (result.usage as { promptTokens?: number; inputTokens?: number })
            ?.promptTokens ||
          (result.usage as { inputTokens?: number })?.inputTokens ||
          0,
        outputTokens:
          (result.usage as { completionTokens?: number; outputTokens?: number })
            ?.completionTokens ||
          (result.usage as { outputTokens?: number })?.outputTokens ||
          0,
        totalTokens: result.usage?.totalTokens || 0,
      },
      provider: this.providerName,
      model: this.modelName,
      toolCalls: this.formatToolCalls(
        (result as { toolCalls?: unknown[] }).toolCalls || [],
      ) as Array<{
        toolCallId: string;
        toolName: string;
        args: StandardRecord;
      }>,
      toolResults: result.toolResults as ToolResult[],
      toolsUsed,
      toolExecutions,
      availableTools: this.formatAvailableTools(tools) as Array<{
        name: string;
        description: string;
        parameters: StandardRecord;
      }>,
    };
  }

  /**
   * Extract tool names from AI SDK result
   */
  private extractToolsUsed(result: AISDKGenerateResult): string[] {
    const toolsUsed: string[] = [];

    // Check direct tool calls (fallback)
    if (result.toolCalls && result.toolCalls.length > 0) {
      toolsUsed.push(
        ...result.toolCalls.map((tc) => {
          return (
            ((tc as UnknownRecord).toolName as string) ||
            ((tc as UnknownRecord).name as string) ||
            "unknown"
          );
        }),
      );
    }

    // Check steps for tool calls (primary source for multi-step)
    const steps = (result as unknown as AISDKGenerateResult).steps;
    if (steps && Array.isArray(steps)) {
      for (const step of steps) {
        if (step?.toolCalls && Array.isArray(step.toolCalls)) {
          toolsUsed.push(
            ...step.toolCalls.map((tc) => tc.toolName || tc.name || "unknown"),
          );
        }
      }
    }

    return [...new Set(toolsUsed)];
  }

  /**
   * Extract tool executions with arguments and results
   */
  private extractToolExecutions(
    result: AISDKGenerateResult,
  ): Array<{ name: string; input: StandardRecord; output: unknown }> {
    const toolExecutions: Array<{
      name: string;
      input: StandardRecord;
      output: unknown;
    }> = [];
    const toolCallArgsMap = new Map<string, StandardRecord>();

    const steps = (result as unknown as AISDKGenerateResult).steps;
    if (!steps || !Array.isArray(steps)) {
      return toolExecutions;
    }

    for (const step of steps) {
      // Collect tool calls and arguments
      if (step?.toolCalls && Array.isArray(step.toolCalls)) {
        this.collectToolCallArguments(step.toolCalls, toolCallArgsMap);
      }

      // Process tool results
      if (step?.toolResults && Array.isArray(step.toolResults)) {
        this.processToolResults(
          step.toolResults,
          toolCallArgsMap,
          toolExecutions,
        );
      }
    }

    return toolExecutions;
  }

  /**
   * Collect tool call arguments for matching with results
   */
  private collectToolCallArguments(
    toolCalls: Array<{ [key: string]: unknown }>,
    toolCallArgsMap: Map<string, StandardRecord>,
  ): void {
    for (const toolCall of toolCalls) {
      const tcRecord = toolCall as UnknownRecord;
      const toolName =
        (tcRecord.toolName as string) || (tcRecord.name as string) || "unknown";
      const toolId =
        (tcRecord.toolCallId as string) || (tcRecord.id as string) || toolName;

      let callArgs: StandardRecord = {};
      if (tcRecord.args) {
        callArgs = tcRecord.args as StandardRecord;
      } else if (tcRecord.arguments) {
        callArgs = tcRecord.arguments as StandardRecord;
      } else if (tcRecord.parameters) {
        callArgs = tcRecord.parameters as StandardRecord;
      }

      toolCallArgsMap.set(toolId, callArgs);
      toolCallArgsMap.set(toolName, callArgs);
    }
  }

  /**
   * Process tool results and match with call arguments
   */
  private processToolResults(
    toolResults: Array<{ [key: string]: unknown }>,
    toolCallArgsMap: Map<string, StandardRecord>,
    toolExecutions: Array<{
      name: string;
      input: StandardRecord;
      output: unknown;
    }>,
  ): void {
    for (const toolResult of toolResults) {
      const trRecord = toolResult as UnknownRecord;
      const toolName = (trRecord.toolName as string) || "unknown";
      const toolId = (trRecord.toolCallId as string) || (trRecord.id as string);

      let toolArgs: StandardRecord = {};
      if (trRecord.args) {
        toolArgs = trRecord.args as StandardRecord;
      } else if (trRecord.arguments) {
        toolArgs = trRecord.arguments as StandardRecord;
      } else if (trRecord.parameters) {
        toolArgs = trRecord.parameters as StandardRecord;
      } else if (trRecord.input) {
        toolArgs = trRecord.input as StandardRecord;
      } else {
        toolArgs = toolCallArgsMap.get(toolId || toolName) || {};
      }

      toolExecutions.push({
        name: toolName,
        input: toolArgs,
        output: (trRecord.result as unknown) || "success",
      });
    }
  }

  /**
   * Format tool calls for the result
   */
  private formatToolCalls(toolCalls: unknown[]): unknown[] {
    if (!toolCalls) {
      return [];
    }

    return toolCalls.map((tc) => ({
      toolCallId:
        ((tc as UnknownRecord).toolCallId as string) ||
        ((tc as UnknownRecord).id as string) ||
        "unknown",
      toolName:
        ((tc as UnknownRecord).toolName as string) ||
        ((tc as UnknownRecord).name as string) ||
        "unknown",
      args:
        ((tc as UnknownRecord).args as StandardRecord) ||
        ((tc as UnknownRecord).parameters as StandardRecord) ||
        {},
    }));
  }

  /**
   * Format available tools for the result
   */
  private formatAvailableTools(tools: Record<string, Tool>): unknown[] {
    return Object.keys(tools).map((name) => {
      const tool = tools[name] as ExtendedTool;
      return {
        name,
        description: tool.description || "No description available",
        parameters: tool.parameters || {},
        server: tool.serverId || "direct",
      };
    });
  }
  /**
   * Alias for generate method - implements AIProvider interface
   */
  async gen(
    optionsOrPrompt: TextGenerationOptions | string,
    analysisSchema?: ValidationSchema,
  ): Promise<EnhancedGenerateResult | null> {
    return this.generate(optionsOrPrompt, analysisSchema);
  }

  /**
   * BACKWARD COMPATIBILITY: Legacy generateText method
   * Converts EnhancedGenerateResult to TextGenerationResult format
   * Ensures existing scripts using createAIProvider().generateText() continue to work
   */
  async generateText(
    options: TextGenerationOptions,
  ): Promise<TextGenerationResult> {
    // Validate required parameters for backward compatibility
    if (
      !options.prompt ||
      typeof options.prompt !== "string" ||
      options.prompt.trim() === ""
    ) {
      throw new Error(
        "GenerateText options must include prompt as a non-empty string",
      );
    }

    // Call the main generate method
    const result = await this.generate(options);

    if (!result) {
      throw new Error("Generation failed: No result returned");
    }

    // Convert EnhancedGenerateResult to TextGenerationResult format
    return {
      content: result.content || "",
      provider: result.provider || this.providerName,
      model: result.model || this.modelName,
      usage: result.usage || {
        promptTokens: 0,
        completionTokens: 0,
        totalTokens: 0,
      },
      responseTime: 0, // BaseProvider doesn't track response time directly
      toolsUsed: result.toolsUsed || [],
      enhancedWithTools: !!(result.toolsUsed && result.toolsUsed.length > 0),
      analytics: result.analytics,
      evaluation: result.evaluation,
    };
  }

  // ===================
  // ABSTRACT METHODS - MUST BE IMPLEMENTED BY SUBCLASSES
  // ===================

  /**
   * Provider-specific streaming implementation (only used when tools are disabled)
   */
  protected abstract executeStream(
    options: StreamOptions,
    analysisSchema?: ValidationSchema,
  ): Promise<StreamResult>;

  /**
   * Get the provider name
   */
  protected abstract getProviderName(): AIProviderName;

  /**
   * Get the default model for this provider
   */
  protected abstract getDefaultModel(): string;

  /**
   * REQUIRED: Every provider MUST implement this method
   * Returns the Vercel AI SDK model instance for this provider
   */
  protected abstract getAISDKModel():
    | LanguageModelV1
    | Promise<LanguageModelV1>;

  // ===================
  // TOOL MANAGEMENT
  // ===================

  /**
   * Get all available tools - direct tools are ALWAYS available
   * MCP tools are added when available (without blocking)
   */
  protected async getAllTools(): Promise<Record<string, Tool>> {
    const tools: Record<string, Tool> = {
      ...this.directTools, // Always include direct tools
    };

    logger.debug(`[BaseProvider] getAllTools called for ${this.providerName}`, {
      neurolinkAvailable: !!this.neurolink,
      neurolinkType: typeof this.neurolink,
      directToolsCount: getKeyCount(this.directTools),
    });
    logger.debug(
      `[BaseProvider] Direct tools: ${getKeysAsString(this.directTools)}`,
    );

    // Add tools from various sources
    await this.addCustomToolsFromExecutor(tools);
    await this.addInMemoryServerTools(tools);
    await this.addExternalMCPTools(tools);
    this.addMCPTools(tools);

    logger.debug(
      `[BaseProvider] getAllTools returning tools: ${getKeysAsString(tools)}`,
    );

    return tools;
  }

  /**
   * Add custom tools from setupToolExecutor if available
   */
  private async addCustomToolsFromExecutor(
    tools: Record<string, Tool>,
  ): Promise<void> {
    if (!this.customTools || this.customTools.size === 0) {
      return;
    }

    logger.debug(
      `[BaseProvider] Loading ${this.customTools.size} custom tools from setupToolExecutor`,
    );

    for (const [toolName, toolDef] of this.customTools.entries()) {
      await this.processCustomTool(tools, toolName, toolDef);
    }
  }

  /**
   * Process a single custom tool definition
   */
  private async processCustomTool(
    tools: Record<string, Tool>,
    toolName: string,
    toolDef: unknown,
  ): Promise<void> {
    logger.debug(`[BaseProvider] Processing custom tool: ${toolName}`, {
      toolDef: typeof toolDef,
      hasExecute:
        toolDef && typeof toolDef === "object" && "execute" in toolDef,
      hasName: toolDef && typeof toolDef === "object" && "name" in toolDef,
    });

    if (
      toolDef &&
      typeof toolDef === "object" &&
      "execute" in toolDef &&
      typeof (toolDef as StandardRecord).execute === "function"
    ) {
      try {
        const { tool: createAISDKTool } = await import("ai");

        const typedToolDef = toolDef as {
          name: string;
          description?: string;
          inputSchema?: unknown;
          execute: Function;
        };

        tools[toolName] = createAISDKTool({
          description: typedToolDef.description || `Custom tool ${toolName}`,
          parameters: z.object({}), // Use empty schema for custom tools
          execute: async (params) => {
            logger.debug(`[BaseProvider] Executing custom tool: ${toolName}`, {
              params,
            });
            // Use the tool executor if available (from setupToolExecutor)
            if (this.toolExecutor) {
              return await this.toolExecutor(toolName, params);
            } else {
              return await typedToolDef.execute(params);
            }
          },
        });

        logger.debug(
          `[BaseProvider] Successfully added custom tool: ${toolName}`,
        );
      } catch (error) {
        logger.error(
          `[BaseProvider] Failed to add custom tool: ${toolName}`,
          error,
        );
      }
    } else {
      logger.warn(`[BaseProvider] Invalid custom tool format: ${toolName}`, {
        toolDef: typeof toolDef,
        hasExecute:
          toolDef && typeof toolDef === "object" && "execute" in toolDef,
        executeType:
          toolDef && typeof toolDef === "object" && "execute" in toolDef
            ? typeof (toolDef as StandardRecord).execute
            : "N/A",
      });
    }
  }

  /**
   * Add custom tools from NeuroLink in-memory servers
   */
  private async addInMemoryServerTools(
    tools: Record<string, Tool>,
  ): Promise<void> {
    if (
      !this.neurolink ||
      typeof this.neurolink.getInMemoryServers !== "function"
    ) {
      return;
    }

    logger.debug(`[BaseProvider] NeuroLink check passed, loading custom tools`);

    try {
      const inMemoryServers = this.neurolink.getInMemoryServers();
      logger.debug(`[BaseProvider] Got servers:`, inMemoryServers.size);
      logger.debug(
        `[BaseProvider] Loading custom tools from SDK, found ${inMemoryServers.size} servers`,
      );

      if (inMemoryServers && inMemoryServers.size > 0) {
        await this.processInMemoryServers(tools, inMemoryServers);
      }
    } catch (error) {
      logger.debug(
        `Failed to load custom tools for ${this.providerName}:`,
        error,
      );
      // Not an error - custom tools are optional
    }
  }

  /**
   * Process in-memory servers and convert their tools
   */
  private async processInMemoryServers(
    tools: Record<string, Tool>,
    inMemoryServers: Map<string, unknown>,
  ): Promise<void> {
    for (const [_serverId, serverConfig] of inMemoryServers) {
      if (serverConfig && (serverConfig as { tools?: unknown }).tools) {
        // Handle tools array from MCPServerInfo
        const serverTools = (serverConfig as { tools: unknown[] }).tools;
        const toolEntries = serverTools.map((tool) => [
          (tool as { name: string }).name,
          tool,
        ]);

        for (const [toolName, toolInfo] of toolEntries as [
          string,
          ToolDefinition,
        ][]) {
          if (toolInfo && typeof toolInfo.execute === "function") {
            await this.convertInMemoryTool(tools, toolName, toolInfo);
          }
        }
      }
    }
  }

  /**
   * Convert an in-memory tool to AI SDK format
   */
  private async convertInMemoryTool(
    tools: Record<string, Tool>,
    toolName: string,
    toolInfo: ToolDefinition,
  ): Promise<void> {
    logger.debug(`[BaseProvider] Converting custom tool: ${toolName}`);

    try {
      const { tool: createAISDKTool } = await import("ai");

      // Validate optional schemas if present
      const isZodSchema = (s: unknown): boolean =>
        typeof s === "object" &&
        s !== null &&
        typeof (s as { parse?: unknown }).parse === "function";

      tools[toolName] = createAISDKTool({
        description: toolInfo.description || `Tool ${toolName}`,
        parameters: isZodSchema(toolInfo.parameters)
          ? (toolInfo.parameters as z.ZodSchema)
          : z.object({}),
        execute: async (params) => {
          const result = await toolInfo.execute(params as ToolArgs);

          // Handle MCP-style results
          if (result && typeof result === "object" && "success" in result) {
            if ((result as { success: boolean }).success) {
              return (result as { data: unknown }).data;
            } else {
              const resultError = (result as { error?: unknown }).error;
              const errorMsg =
                typeof resultError === "string"
                  ? resultError
                  : resultError &&
                      typeof resultError === "object" &&
                      "message" in resultError
                    ? String((resultError as { message: unknown }).message)
                    : "Tool execution failed";
              throw new Error(errorMsg);
            }
          }
          return result;
        },
      });
    } catch (toolCreationError) {
      logger.error(`Failed to create tool: ${toolName}`, toolCreationError);
    }
  }

  /**
   * Add external MCP tools from NeuroLink
   */
  private async addExternalMCPTools(
    tools: Record<string, Tool>,
  ): Promise<void> {
    if (
      !this.neurolink ||
      typeof this.neurolink.getExternalMCPTools !== "function"
    ) {
      logger.debug(`[BaseProvider] No external MCP tool interface available`, {
        hasNeuroLink: !!this.neurolink,
        hasGetExternalMCPTools:
          this.neurolink &&
          typeof this.neurolink.getExternalMCPTools === "function",
      });
      return;
    }

    try {
      logger.debug(
        `[BaseProvider] Loading external MCP tools from NeuroLink via direct tool access`,
      );

      const externalTools = this.neurolink.getExternalMCPTools() || [];
      logger.debug(
        `[BaseProvider] Found ${externalTools.length} external MCP tools`,
      );

      for (const tool of externalTools) {
        await this.convertExternalMCPTool(tools, tool);
      }

      logger.debug(`[BaseProvider] External MCP tools loading complete`, {
        totalToolsAdded: externalTools.length,
      });
    } catch (error) {
      logger.error(
        `[BaseProvider] Failed to load external MCP tools for ${this.providerName}:`,
        error,
      );
      // Not an error - external tools are optional
    }
  }

  /**
   * Convert an external MCP tool to AI SDK format
   */
  private async convertExternalMCPTool(
    tools: Record<string, Tool>,
    tool: {
      name: string;
      description?: string;
      inputSchema?: unknown;
      serverId?: string;
    },
  ): Promise<void> {
    logger.debug(`[BaseProvider] Converting external MCP tool: ${tool.name}`);

    try {
      const { tool: createAISDKTool } = await import("ai");

      tools[tool.name] = createAISDKTool({
        description: tool.description || `External MCP tool ${tool.name}`,
        parameters: await this.convertMCPSchemaToZod(
          tool.inputSchema as StandardRecord | undefined,
        ),
        execute: async (params) => {
          logger.debug(
            `[BaseProvider] Executing external MCP tool: ${tool.name}`,
            { params },
          );

          // Execute via NeuroLink's direct tool execution
          if (
            this.neurolink &&
            typeof this.neurolink.executeExternalMCPTool === "function"
          ) {
            return await this.neurolink.executeExternalMCPTool(
              tool.serverId || "unknown",
              tool.name,
              params as JsonObject,
            );
          } else {
            throw new Error(
              `Cannot execute external MCP tool: NeuroLink executeExternalMCPTool not available`,
            );
          }
        },
      });

      logger.debug(
        `[BaseProvider] Successfully added external MCP tool: ${tool.name}`,
      );
    } catch (toolCreationError) {
      logger.error(
        `Failed to create external MCP tool: ${tool.name}`,
        toolCreationError,
      );
    }
  }

  /**
   * Add MCP tools if available
   */
  private addMCPTools(tools: Record<string, Tool>): void {
    // MCP tools loading simplified - removed functionCalling dependency
    if (!this.mcpTools) {
      // Set empty tools object - MCP tools are handled at a higher level
      this.mcpTools = {};
    }

    // Add MCP tools if available
    if (this.mcpTools) {
      Object.assign(tools, this.mcpTools);
    }
  }

  /**
   * Convert MCP JSON Schema to Zod schema for AI SDK tools
   * Handles common MCP schema patterns safely
   */
  private async convertMCPSchemaToZod(
    inputSchema?: StandardRecord,
  ): Promise<ZodUnknownSchema> {
    const { z } = await import("zod");

    if (!inputSchema || typeof inputSchema !== "object") {
      return z.object({});
    }

    try {
      const schema = inputSchema as StandardRecord;
      const zodFields: Record<string, ZodUnknownSchema> = {};

      // Handle JSON Schema properties
      if (schema.properties && typeof schema.properties === "object") {
        const required = new Set(
          Array.isArray(schema.required) ? schema.required : [],
        );

        for (const [propName, propDef] of Object.entries(schema.properties)) {
          const zodType = this.convertJsonSchemaTypeToZod(
            propDef as StandardRecord,
            z,
            required.has(propName),
          );
          zodFields[propName] = zodType;
        }
      }

      return getKeyCount(zodFields) > 0 ? z.object(zodFields) : z.object({});
    } catch (error) {
      logger.warn(
        `Failed to convert MCP schema to Zod, using empty schema:`,
        error,
      );
      return z.object({});
    }
  }

  /**
   * Convert individual JSON Schema type to Zod type
   */
  private convertJsonSchemaTypeToZod(
    prop: StandardRecord,
    z: typeof import("zod").z,
    isRequired: boolean,
  ): ZodUnknownSchema {
    let zodType: ZodUnknownSchema;

    // Convert based on JSON Schema type
    switch (prop.type) {
      case "string":
        zodType = z.string();
        break;
      case "number":
      case "integer":
        zodType = z.number();
        break;
      case "boolean":
        zodType = z.boolean();
        break;
      case "array":
        zodType = z.array(z.unknown());
        break;
      case "object":
        zodType = z.object({});
        break;
      default:
        // Unknown type, use string as fallback
        zodType = z.string();
    }

    // Add description if available
    if (prop.description && typeof prop.description === "string") {
      zodType = zodType.describe(prop.description);
    }

    // Make optional if not required
    if (!isRequired) {
      zodType = zodType.optional();
    }

    return zodType;
  }

  /**
   * Set session context for MCP tools
   */
  public setSessionContext(sessionId?: string, userId?: string): void {
    this.sessionId = sessionId;
    this.userId = userId;
  }

  /**
   * Provider-specific error handling
   */
  protected abstract handleProviderError(error: unknown): Error;

  // ===================
  // CONSOLIDATED PROVIDER METHODS - MOVED FROM INDIVIDUAL PROVIDERS
  // ===================

  /**
   * Execute operation with timeout and proper cleanup
   * Consolidates identical timeout handling from 8/10 providers
   */
  protected async executeWithTimeout<T>(
    operation: () => Promise<T>,
    options: { timeout?: number | string; operationType?: string },
  ): Promise<T> {
    const timeout = this.getTimeout(
      options as StreamOptions | TextGenerationOptions,
    );
    const timeoutController = createTimeoutController(
      timeout,
      this.providerName,
      (options.operationType as "generate" | "stream") || "generate",
    );

    try {
      if (timeoutController) {
        return await Promise.race([
          operation(),
          new Promise<never>((_, reject) => {
            timeoutController.controller.signal.addEventListener(
              "abort",
              () => {
                reject(
                  new TimeoutError(
                    `${this.providerName} operation timed out`,
                    timeoutController.timeoutMs,
                    this.providerName,
                    (options.operationType as "generate" | "stream") ||
                      "generate",
                  ),
                );
              },
            );
          }),
        ]);
      } else {
        return await operation();
      }
    } finally {
      timeoutController?.cleanup();
    }
  }

  /**
   * Validate stream options - consolidates validation from 7/10 providers
   */
  protected validateStreamOptions(options: StreamOptions): void {
    const validation = validateStreamOpts(options);

    if (!validation.isValid) {
      const summary = createValidationSummary(validation);
      throw new ValidationError(
        `Stream options validation failed: ${summary}`,
        "options",
        "VALIDATION_FAILED",
        validation.suggestions,
      );
    }

    // Log warnings if any
    if (validation.warnings.length > 0) {
      logger.warn("Stream options validation warnings:", validation.warnings);
    }

    // Additional BaseProvider-specific validation
    if (options.maxSteps !== undefined) {
      if (
        options.maxSteps < STEP_LIMITS.min ||
        options.maxSteps > STEP_LIMITS.max
      ) {
        throw new ValidationError(
          `maxSteps must be between ${STEP_LIMITS.min} and ${STEP_LIMITS.max}`,
          "maxSteps",
          "OUT_OF_RANGE",
          [
            `Use a value between ${STEP_LIMITS.min} and ${STEP_LIMITS.max} for optimal performance`,
          ],
        );
      }
    }
  }

  /**
   * Create text stream transformation - consolidates identical logic from 7/10 providers
   */
  protected createTextStream(result: {
    textStream: AsyncIterable<string>;
  }): AsyncGenerator<{ content: string }> {
    return (async function* (): AsyncGenerator<{ content: string }> {
      for await (const chunk of result.textStream) {
        yield { content: chunk };
      }
    })();
  }

  /**
   * Create standardized stream result - consolidates result structure
   */
  protected createStreamResult(
    stream: AsyncGenerator<{ content: string }>,
    additionalProps: Partial<StreamResult> = {},
  ): StreamResult {
    return {
      stream,
      provider: this.providerName,
      model: this.modelName,
      ...additionalProps,
    };
  }

  /**
   * Create stream analytics - consolidates analytics from 4/10 providers
   */
  protected async createStreamAnalytics(
    result: UnknownRecord,
    startTime: number,
    options: StreamOptions,
  ): Promise<UnknownRecord | undefined> {
    try {
      const { createAnalytics } = await import("./analytics.js");
      const analytics = createAnalytics(
        this.providerName,
        this.modelName,
        result,
        Date.now() - startTime,
        {
          requestId: `${this.providerName}-stream-${Date.now()}`,
          streamingMode: true,
          ...options.context,
        },
      );
      return analytics as unknown as UnknownRecord;
    } catch (error) {
      logger.warn(`Analytics creation failed for ${this.providerName}:`, error);
      return undefined;
    }
  }

  /**
   * Handle common error patterns - consolidates error handling from multiple providers
   */
  protected handleCommonErrors(error: unknown): Error | null {
    if (error instanceof TimeoutError) {
      return new Error(
        `${this.providerName} request timed out after ${error.timeout}ms. Consider increasing timeout or using a lighter model.`,
      );
    }

    const message = error instanceof Error ? error.message : String(error);

    // Common API key errors
    if (
      message.includes("API_KEY_INVALID") ||
      message.includes("Invalid API key") ||
      message.includes("authentication") ||
      message.includes("unauthorized")
    ) {
      return new Error(
        `Invalid API key for ${this.providerName}. Please check your API key environment variable.`,
      );
    }

    // Common rate limit errors
    if (
      message.includes("rate limit") ||
      message.includes("quota") ||
      message.includes("429")
    ) {
      return new Error(
        `Rate limit exceeded for ${this.providerName}. Please wait before making more requests.`,
      );
    }

    return null; // Not a common error, let provider handle it
  }

  /**
   * Set up tool executor for a provider to enable actual tool execution
   * Consolidates identical setupToolExecutor logic from neurolink.ts (used in 4 places)
   * @param sdk - The NeuroLinkSDK instance for tool execution
   * @param functionTag - Function name for logging
   */
  setupToolExecutor(
    sdk: {
      customTools: Map<string, unknown>;
      executeTool: (toolName: string, params: unknown) => Promise<unknown>;
    },
    functionTag: string,
  ): void {
    // Store custom tools for use in getAllTools()
    this.customTools = sdk.customTools;
    this.toolExecutor = sdk.executeTool;

    logger.debug(`[${functionTag}] Setting up tool executor for provider`, {
      providerType: this.constructor.name,
      availableCustomTools: sdk.customTools.size,
      customToolsStored: !!this.customTools,
      toolExecutorStored: !!this.toolExecutor,
    });

    // Note: Tool execution will be handled through getAllTools() -> AI SDK tools
    // The custom tools are converted to AI SDK format in getAllTools() method
  }

  // ===================
  // TEMPLATE METHODS - COMMON FUNCTIONALITY
  // ===================

  protected normalizeTextOptions(
    optionsOrPrompt: TextGenerationOptions | string,
  ): TextGenerationOptions {
    if (typeof optionsOrPrompt === "string") {
      const safeMaxTokens = getSafeMaxTokens(this.providerName, this.modelName);
      return {
        prompt: optionsOrPrompt,
        provider: this.providerName,
        model: this.modelName,
        maxTokens: safeMaxTokens,
      };
    }

    // Handle both prompt and input.text formats
    const prompt = optionsOrPrompt.prompt || optionsOrPrompt.input?.text || "";
    const modelName = optionsOrPrompt.model || this.modelName;
    const providerName = optionsOrPrompt.provider || this.providerName;

    // Apply safe maxTokens based on provider and model
    const safeMaxTokens = getSafeMaxTokens(
      providerName,
      modelName,
      optionsOrPrompt.maxTokens,
    );

    return {
      ...optionsOrPrompt,
      prompt,
      provider: providerName,
      model: modelName,
      maxTokens: safeMaxTokens,
    };
  }

  protected normalizeStreamOptions(
    optionsOrPrompt: StreamOptions | string,
  ): StreamOptions {
    if (typeof optionsOrPrompt === "string") {
      const safeMaxTokens = getSafeMaxTokens(this.providerName, this.modelName);
      return {
        input: { text: optionsOrPrompt },
        provider: this.providerName,
        model: this.modelName,
        maxTokens: safeMaxTokens,
      };
    }

    const modelName = optionsOrPrompt.model || this.modelName;
    const providerName = optionsOrPrompt.provider || this.providerName;

    // Apply safe maxTokens based on provider and model
    const safeMaxTokens = getSafeMaxTokens(
      providerName,
      modelName,
      optionsOrPrompt.maxTokens,
    );

    return {
      ...optionsOrPrompt,
      provider: providerName,
      model: modelName,
      maxTokens: safeMaxTokens,
    };
  }

  protected async enhanceResult(
    result: EnhancedGenerateResult,
    options: TextGenerationOptions,
    startTime: number,
  ): Promise<EnhancedGenerateResult> {
    const responseTime = Date.now() - startTime;
    let enhancedResult = { ...result };

    if (options.enableAnalytics) {
      try {
        logger.debug(`Creating analytics for ${this.providerName}...`);
        const analytics = await this.createAnalytics(
          result,
          responseTime,
          options,
        );
        logger.debug(`Analytics created:`, analytics);
        enhancedResult = { ...enhancedResult, analytics };
      } catch (error) {
        logger.warn(
          `Analytics creation failed for ${this.providerName}:`,
          error,
        );
      }
    }

    if (options.enableEvaluation) {
      try {
        const evaluation = await this.createEvaluation(result, options);
        enhancedResult = { ...enhancedResult, evaluation };
      } catch (error) {
        logger.warn(
          `Evaluation creation failed for ${this.providerName}:`,
          error,
        );
      }
    }

    return enhancedResult;
  }

  protected async createAnalytics(
    result: EnhancedGenerateResult,
    responseTime: number,
    options: TextGenerationOptions,
  ): Promise<AnalyticsData> {
    const { createAnalytics } = await import("./analytics.js");
    return createAnalytics(
      this.providerName,
      this.modelName,
      result,
      responseTime,
      options.context,
    );
  }

  protected async createEvaluation(
    result: EnhancedGenerateResult,
    options: TextGenerationOptions,
  ): Promise<EvaluationData> {
    const { evaluateResponse } = await import("../core/evaluation.js");
    const evaluation = await evaluateResponse(result.content, options.prompt);
    return evaluation as EvaluationData;
  }

  protected validateOptions(options: TextGenerationOptions): void {
    const validation = validateTextGenerationOptions(options);

    if (!validation.isValid) {
      const summary = createValidationSummary(validation);
      throw new ValidationError(
        `Text generation options validation failed: ${summary}`,
        "options",
        "VALIDATION_FAILED",
        validation.suggestions,
      );
    }

    // Log warnings if any
    if (validation.warnings.length > 0) {
      logger.warn(
        "Text generation options validation warnings:",
        validation.warnings,
      );
    }

    // Additional BaseProvider-specific validation
    if (options.maxSteps !== undefined) {
      if (
        options.maxSteps < STEP_LIMITS.min ||
        options.maxSteps > STEP_LIMITS.max
      ) {
        throw new ValidationError(
          `maxSteps must be between ${STEP_LIMITS.min} and ${STEP_LIMITS.max}`,
          "maxSteps",
          "OUT_OF_RANGE",
          [
            `Use a value between ${STEP_LIMITS.min} and ${STEP_LIMITS.max} for optimal performance`,
          ],
        );
      }
    }
  }

  protected getProviderInfo(): { provider: string; model: string } {
    return {
      provider: this.providerName,
      model: this.modelName,
    };
  }
  /**
   * Get timeout value in milliseconds
   */
  public getTimeout(options: TextGenerationOptions | StreamOptions): number {
    if (!options.timeout) {
      return this.defaultTimeout;
    }

    if (typeof options.timeout === "number") {
      return options.timeout;
    }

    // Parse string timeout (e.g., '30s', '2m', '1h')
    const timeoutStr = options.timeout.toLowerCase();
    const value = parseInt(timeoutStr);

    if (timeoutStr.includes("h")) {
      return value * 60 * 60 * 1000;
    } else if (timeoutStr.includes("m")) {
      return value * 60 * 1000;
    } else if (timeoutStr.includes("s")) {
      return value * 1000;
    }

    return this.defaultTimeout;
  }

  /**
   * Utility method to chunk large prompts into smaller pieces
   * @param prompt The prompt to chunk
   * @param maxChunkSize Maximum size per chunk (default: 900,000 characters)
   * @param overlap Overlap between chunks to maintain context (default: 100 characters)
   * @returns Array of prompt chunks
   */
  static chunkPrompt(
    prompt: string,
    maxChunkSize: number = 900000,
    overlap: number = 100,
  ): string[] {
    if (prompt.length <= maxChunkSize) {
      return [prompt];
    }

    const chunks: string[] = [];
    let start = 0;

    while (start < prompt.length) {
      const end = Math.min(start + maxChunkSize, prompt.length);
      chunks.push(prompt.slice(start, end));

      // Break if we've reached the end
      if (end >= prompt.length) {
        break;
      }

      // Move start forward, accounting for overlap
      const nextStart = end - overlap;

      // Ensure we make progress (avoid infinite loops)
      if (nextStart <= start) {
        start = end;
      } else {
        start = Math.max(nextStart, 0);
      }
    }

    return chunks;
  }
}

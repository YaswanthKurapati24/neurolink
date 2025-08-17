/**
 * Utility functions for AI provider management
 * Consolidated from providerUtils-fixed.ts
 */
import { AIProviderFactory } from "../core/factory.js";
import { logger } from "./logger.js";
import type { UnknownRecord } from "../types/common.js";
import type { AIProviderName } from "../core/types.js";
import { ProviderHealthChecker } from "./providerHealth.js";

/**
 * Get the best available provider based on real-time availability checks
 * Enhanced version consolidated from providerUtils-fixed.ts
 * @param requestedProvider - Optional preferred provider name
 * @returns The best provider name to use
 */
export async function getBestProvider(
  requestedProvider?: string,
): Promise<string> {
  // Check requested provider FIRST - explicit user choice overrides defaults
  if (requestedProvider && requestedProvider !== "auto") {
    // For explicit provider requests, check health first
    try {
      const health = await ProviderHealthChecker.checkProviderHealth(
        requestedProvider as AIProviderName,
        { includeConnectivityTest: false, cacheResults: true },
      );

      if (health.isHealthy) {
        logger.debug(
          `[getBestProvider] Using healthy explicitly requested provider: ${requestedProvider}`,
        );
        return requestedProvider;
      } else {
        logger.warn(
          `[getBestProvider] Requested provider ${requestedProvider} is unhealthy, finding alternative`,
          { error: health.error },
        );
      }
    } catch (error) {
      logger.warn(
        `[getBestProvider] Health check failed for ${requestedProvider}, using anyway`,
        { error: error instanceof Error ? error.message : String(error) },
      );
      return requestedProvider; // Return anyway for explicit requests
    }
  }

  // Use health checker to get best available provider
  const healthyProvider = await ProviderHealthChecker.getBestHealthyProvider();

  if (healthyProvider) {
    logger.debug(
      `[getBestProvider] Selected healthy provider: ${healthyProvider}`,
    );
    return healthyProvider;
  }

  // Fallback to legacy provider checking if health system fails
  logger.warn(
    "[getBestProvider] Health system failed, falling back to legacy checking",
  );

  // Check for explicit default provider in env (only when no provider requested)
  if (
    process.env.DEFAULT_PROVIDER &&
    (await isProviderAvailable(process.env.DEFAULT_PROVIDER))
  ) {
    logger.debug(
      `[getBestProvider] Using default provider from env: ${process.env.DEFAULT_PROVIDER}`,
    );
    return process.env.DEFAULT_PROVIDER;
  }

  // Special case for Ollama - prioritize local when available
  if (process.env.OLLAMA_BASE_URL && process.env.OLLAMA_MODEL) {
    try {
      if (await isProviderAvailable("ollama")) {
        logger.debug(`[getBestProvider] Prioritizing working local Ollama`);
        return "ollama"; // Prioritize working local AI
      }
    } catch {
      // Fall through to cloud providers
    }
  }

  /**
   * Provider priority order rationale:
   * - Vertex (Google Cloud AI) is prioritized first for its enterprise-grade reliability and advanced model capabilities.
   * - Google AI follows as second priority for comprehensive Google AI ecosystem support.
   * - OpenAI maintains high priority due to its consistent reliability and broad model support.
   * - Other providers are ordered based on a combination of reliability, feature set, and historical performance in our use cases.
   * - Ollama is kept as a fallback for local deployments when available.
   * Please update this comment if the order is changed in the future, and document the rationale for maintainability.
   */
  const providers = [
    "vertex", // Prioritize Google Cloud AI (Vertex) first
    "google-ai", // Google AI ecosystem support
    "openai", // Reliable with broad model support
    "anthropic",
    "bedrock",
    "azure",
    "mistral",
    "huggingface",
    "ollama", // Keep as fallback
  ];

  for (const provider of providers) {
    if (await isProviderAvailable(provider)) {
      logger.debug(`[getBestProvider] Selected provider: ${provider}`);
      return provider;
    }
  }

  throw new Error(
    "No available AI providers. Please check your configurations.",
  );
}

/**
 * Check if a provider is truly available by performing a quick authentication test.
 * Enhanced function consolidated from providerUtils-fixed.ts
 * @param providerName - The name of the provider to check.
 * @returns True if the provider is available and authenticated.
 */
async function isProviderAvailable(providerName: string): Promise<boolean> {
  if (!hasProviderEnvVars(providerName) && providerName !== "ollama") {
    return false;
  }

  if (providerName === "ollama") {
    try {
      const response = await fetch("http://localhost:11434/api/tags", {
        method: "GET",
        signal: AbortSignal.timeout(2000),
      });
      if (response.ok) {
        const { models } = await response.json();
        const defaultOllamaModel = "llama3.2:latest";
        return models.some((m: UnknownRecord) => m.name === defaultOllamaModel);
      }
      return false;
    } catch {
      return false;
    }
  }

  try {
    const provider = await AIProviderFactory.createProvider(providerName);
    await provider.generate({ prompt: "test", maxTokens: 1 });
    return true;
  } catch {
    return false;
  }
}

/**
 * Validation results for environment variables
 */
export interface EnvVarValidationResult {
  isValid: boolean;
  missingVars: string[];
  invalidVars: string[];
  warnings: string[];
}

/**
 * Google Cloud Project ID validation regex
 * Format requirements:
 * - Must start with a lowercase letter
 * - Can contain lowercase letters, numbers, and hyphens
 * - Must end with a lowercase letter or number
 * - Total length must be 6-30 characters
 */
const GOOGLE_CLOUD_PROJECT_ID_REGEX = /^[a-z][a-z0-9-]{4,28}[a-z0-9]$/;

/**
 * Validate environment variable values for a provider
 * Addresses GitHub Copilot comment about adding environment variable validation
 * @param provider - Provider name to validate
 * @returns Validation result with detailed information
 */
export function validateProviderEnvVars(
  provider: string,
): EnvVarValidationResult {
  const result: EnvVarValidationResult = {
    isValid: true,
    missingVars: [],
    invalidVars: [],
    warnings: [],
  };

  const normalizedProvider = provider.toLowerCase();

  // Group validation by provider type
  if (isAwsProvider(normalizedProvider)) {
    validateAwsCredentials(result);
  } else if (isVertexProvider(normalizedProvider)) {
    validateVertexCredentials(result);
  } else if (isOpenAiProvider(normalizedProvider)) {
    validateOpenAICredentials(result);
  } else if (isAnthropicProvider(normalizedProvider)) {
    validateAnthropicCredentials(result);
  } else if (isAzureProvider(normalizedProvider)) {
    validateAzureCredentials(result);
  } else if (isGoogleAiProvider(normalizedProvider)) {
    validateGoogleAICredentials(result);
  } else if (isHuggingFaceProvider(normalizedProvider)) {
    validateHuggingFaceCredentials(result);
  } else if (isMistralProvider(normalizedProvider)) {
    validateMistralCredentials(result);
  } else if (isOllamaProvider(normalizedProvider)) {
    // Ollama doesn't require environment variables
  } else if (isLiteLlmProvider(normalizedProvider)) {
    // LiteLLM validation can be added if needed
  } else {
    result.isValid = false;
    result.warnings.push(`Unknown provider: ${provider}`);
  }

  result.isValid =
    result.missingVars.length === 0 && result.invalidVars.length === 0;
  return result;
}

/**
 * Check if provider is AWS-based
 */
function isAwsProvider(provider: string): boolean {
  return ["bedrock", "amazon", "aws"].includes(provider);
}

/**
 * Check if provider is Vertex-based
 */
function isVertexProvider(provider: string): boolean {
  return ["vertex", "googlevertex", "google", "gemini"].includes(provider);
}

/**
 * Check if provider is OpenAI-based
 */
function isOpenAiProvider(provider: string): boolean {
  return ["openai", "gpt"].includes(provider);
}

/**
 * Check if provider is Anthropic-based
 */
function isAnthropicProvider(provider: string): boolean {
  return ["anthropic", "claude"].includes(provider);
}

/**
 * Check if provider is Azure-based
 */
function isAzureProvider(provider: string): boolean {
  return ["azure", "azureopenai"].includes(provider);
}

/**
 * Check if provider is Google AI-based
 */
function isGoogleAiProvider(provider: string): boolean {
  return ["google-ai", "google-studio"].includes(provider);
}

/**
 * Check if provider is HuggingFace-based
 */
function isHuggingFaceProvider(provider: string): boolean {
  return ["huggingface", "hugging-face", "hf"].includes(provider);
}

/**
 * Check if provider is Mistral-based
 */
function isMistralProvider(provider: string): boolean {
  return ["mistral", "mistral-ai", "mistralai"].includes(provider);
}

/**
 * Check if provider is Ollama-based
 */
function isOllamaProvider(provider: string): boolean {
  return ["ollama", "local", "local-ollama"].includes(provider);
}

/**
 * Check if provider is LiteLLM-based
 */
function isLiteLlmProvider(provider: string): boolean {
  return provider === "litellm";
}

/**
 * Validate AWS credentials with flexible validation
 * Note: AWS credential formats can vary, so validation is kept reasonably flexible
 */
function validateAwsCredentials(result: EnvVarValidationResult): void {
  const accessKeyId = process.env.AWS_ACCESS_KEY_ID;
  const secretAccessKey = process.env.AWS_SECRET_ACCESS_KEY;
  const region = process.env.AWS_REGION || process.env.AWS_DEFAULT_REGION;

  if (!accessKeyId) {
    result.missingVars.push("AWS_ACCESS_KEY_ID");
  } else if (!/^[A-Z0-9]{16,}$/.test(accessKeyId)) {
    // Flexible validation: at least 16 uppercase alphanumeric characters
    result.invalidVars.push(
      "AWS_ACCESS_KEY_ID (should be uppercase alphanumeric characters, typically 20 chars)",
    );
  }

  if (!secretAccessKey) {
    result.missingVars.push("AWS_SECRET_ACCESS_KEY");
  } else if (!/^[A-Za-z0-9+/]{30,}$/.test(secretAccessKey)) {
    // Flexible validation: at least 30 base64 characters (can vary in length)
    result.invalidVars.push(
      "AWS_SECRET_ACCESS_KEY (should be base64 characters, typically 40+ chars)",
    );
  }

  if (!region) {
    result.warnings.push("AWS_REGION not set, will use default region");
  }
}

/**
 * Validate Google Vertex credentials
 */
function validateVertexCredentials(result: EnvVarValidationResult): void {
  const projectId =
    process.env.GOOGLE_CLOUD_PROJECT_ID ||
    process.env.VERTEX_PROJECT_ID ||
    process.env.GOOGLE_VERTEX_PROJECT ||
    process.env.GOOGLE_CLOUD_PROJECT;

  const hasCredentials =
    process.env.GOOGLE_APPLICATION_CREDENTIALS ||
    process.env.GOOGLE_SERVICE_ACCOUNT_KEY ||
    (process.env.GOOGLE_AUTH_CLIENT_EMAIL &&
      process.env.GOOGLE_AUTH_PRIVATE_KEY);

  if (!projectId) {
    result.missingVars.push("GOOGLE_CLOUD_PROJECT_ID (or variant)");
  } else if (!GOOGLE_CLOUD_PROJECT_ID_REGEX.test(projectId)) {
    result.invalidVars.push(
      "Project ID format invalid (must be 6-30 lowercase letters, digits, hyphens)",
    );
  }

  if (!hasCredentials) {
    result.missingVars.push(
      "Google credentials (GOOGLE_APPLICATION_CREDENTIALS or explicit auth)",
    );
  }

  if (
    process.env.GOOGLE_AUTH_CLIENT_EMAIL &&
    !isValidEmail(process.env.GOOGLE_AUTH_CLIENT_EMAIL)
  ) {
    result.invalidVars.push("GOOGLE_AUTH_CLIENT_EMAIL (invalid email format)");
  }
}

/**
 * Validate OpenAI credentials
 */
function validateOpenAICredentials(result: EnvVarValidationResult): void {
  const apiKey = process.env.OPENAI_API_KEY;

  if (!apiKey) {
    result.missingVars.push("OPENAI_API_KEY");
  } else if (!/^sk-[A-Za-z0-9]{48,}$/.test(apiKey)) {
    result.invalidVars.push(
      "OPENAI_API_KEY (should start with 'sk-' followed by 48+ characters)",
    );
  }
}

/**
 * Validate Anthropic credentials
 */
function validateAnthropicCredentials(result: EnvVarValidationResult): void {
  const apiKey = process.env.ANTHROPIC_API_KEY;

  if (!apiKey) {
    result.missingVars.push("ANTHROPIC_API_KEY");
  } else if (!/^sk-ant-[A-Za-z0-9-_]{95,}$/.test(apiKey)) {
    result.invalidVars.push(
      "ANTHROPIC_API_KEY (should start with 'sk-ant-' followed by 95+ characters)",
    );
  }
}

/**
 * Validate Azure credentials
 */
function validateAzureCredentials(result: EnvVarValidationResult): void {
  const apiKey = process.env.AZURE_OPENAI_API_KEY;
  const endpoint = process.env.AZURE_OPENAI_ENDPOINT;

  if (!apiKey) {
    result.missingVars.push("AZURE_OPENAI_API_KEY");
  } else if (!/^[a-f0-9]{32}$/.test(apiKey)) {
    result.invalidVars.push(
      "AZURE_OPENAI_API_KEY (should be 32 hexadecimal characters)",
    );
  }

  if (!endpoint) {
    result.missingVars.push("AZURE_OPENAI_ENDPOINT");
  } else if (!isValidUrl(endpoint)) {
    result.invalidVars.push(
      "AZURE_OPENAI_ENDPOINT (should be a valid HTTPS URL)",
    );
  }
}

/**
 * Validate Google AI credentials
 */
function validateGoogleAICredentials(result: EnvVarValidationResult): void {
  const apiKey =
    process.env.GOOGLE_AI_API_KEY || process.env.GOOGLE_GENERATIVE_AI_API_KEY;

  if (!apiKey) {
    result.missingVars.push(
      "GOOGLE_AI_API_KEY (or GOOGLE_GENERATIVE_AI_API_KEY)",
    );
  } else if (!/^[A-Za-z0-9_-]{39}$/.test(apiKey)) {
    result.invalidVars.push(
      "GOOGLE_AI_API_KEY (should be 39 alphanumeric characters with dashes/underscores)",
    );
  }
}

/**
 * Validate HuggingFace credentials
 */
function validateHuggingFaceCredentials(result: EnvVarValidationResult): void {
  const apiKey = process.env.HUGGINGFACE_API_KEY || process.env.HF_TOKEN;

  if (!apiKey) {
    result.missingVars.push("HUGGINGFACE_API_KEY (or HF_TOKEN)");
  } else if (!/^hf_[A-Za-z0-9]{37}$/.test(apiKey)) {
    result.invalidVars.push(
      "HUGGINGFACE_API_KEY (should start with 'hf_' followed by 37 characters)",
    );
  }
}

/**
 * Validate Mistral credentials
 */
function validateMistralCredentials(result: EnvVarValidationResult): void {
  const apiKey = process.env.MISTRAL_API_KEY;

  if (!apiKey) {
    result.missingVars.push("MISTRAL_API_KEY");
  } else if (!/^[A-Za-z0-9]{32,}$/.test(apiKey)) {
    result.invalidVars.push(
      "MISTRAL_API_KEY (should be 32+ alphanumeric characters)",
    );
  }
}

/**
 * Helper function to validate email format
 */
function isValidEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

/**
 * Helper function to validate URL format
 */
function isValidUrl(url: string): boolean {
  try {
    const parsedUrl = new URL(url);
    return parsedUrl.protocol === "https:";
  } catch {
    return false;
  }
}

/**
 * Check if a provider has the minimum required environment variables
 * NOTE: This only checks if variables exist, not if they're valid
 * For validation, use validateProviderEnvVars instead
 * @param provider - Provider name to check
 * @returns True if the provider has required environment variables
 */
export function hasProviderEnvVars(provider: string): boolean {
  const normalizedProvider = provider.toLowerCase();

  if (hasAwsProviderEnvVars(normalizedProvider)) {
    return true;
  }
  if (hasVertexProviderEnvVars(normalizedProvider)) {
    return true;
  }
  if (hasOpenAiProviderEnvVars(normalizedProvider)) {
    return true;
  }
  if (hasAnthropicProviderEnvVars(normalizedProvider)) {
    return true;
  }
  if (hasAzureProviderEnvVars(normalizedProvider)) {
    return true;
  }
  if (hasGoogleAiProviderEnvVars(normalizedProvider)) {
    return true;
  }
  if (hasHuggingFaceProviderEnvVars(normalizedProvider)) {
    return true;
  }
  if (hasMistralProviderEnvVars(normalizedProvider)) {
    return true;
  }
  if (hasOllamaProviderEnvVars(normalizedProvider)) {
    return true;
  }
  if (hasLiteLlmProviderEnvVars(normalizedProvider)) {
    return true;
  }

  return false;
}

/**
 * Check AWS provider environment variables
 */
function hasAwsProviderEnvVars(provider: string): boolean {
  if (!["bedrock", "amazon", "aws"].includes(provider)) {
    return false;
  }
  return !!(process.env.AWS_ACCESS_KEY_ID && process.env.AWS_SECRET_ACCESS_KEY);
}

/**
 * Check Vertex provider environment variables
 */
function hasVertexProviderEnvVars(provider: string): boolean {
  if (!["vertex", "googlevertex", "google", "gemini"].includes(provider)) {
    return false;
  }

  const hasProject = !!(
    process.env.GOOGLE_CLOUD_PROJECT_ID ||
    process.env.VERTEX_PROJECT_ID ||
    process.env.GOOGLE_VERTEX_PROJECT ||
    process.env.GOOGLE_CLOUD_PROJECT
  );

  const hasCredentials = !!(
    process.env.GOOGLE_APPLICATION_CREDENTIALS ||
    process.env.GOOGLE_SERVICE_ACCOUNT_KEY ||
    (process.env.GOOGLE_AUTH_CLIENT_EMAIL &&
      process.env.GOOGLE_AUTH_PRIVATE_KEY)
  );

  return hasProject && hasCredentials;
}

/**
 * Check OpenAI provider environment variables
 */
function hasOpenAiProviderEnvVars(provider: string): boolean {
  if (!["openai", "gpt"].includes(provider)) {
    return false;
  }
  return !!process.env.OPENAI_API_KEY;
}

/**
 * Check Anthropic provider environment variables
 */
function hasAnthropicProviderEnvVars(provider: string): boolean {
  if (!["anthropic", "claude"].includes(provider)) {
    return false;
  }
  return !!process.env.ANTHROPIC_API_KEY;
}

/**
 * Check Azure provider environment variables
 */
function hasAzureProviderEnvVars(provider: string): boolean {
  if (!["azure", "azureopenai"].includes(provider)) {
    return false;
  }
  return !!process.env.AZURE_OPENAI_API_KEY;
}

/**
 * Check Google AI provider environment variables
 */
function hasGoogleAiProviderEnvVars(provider: string): boolean {
  if (!["google-ai", "google-studio"].includes(provider)) {
    return false;
  }
  return !!(
    process.env.GOOGLE_AI_API_KEY || process.env.GOOGLE_GENERATIVE_AI_API_KEY
  );
}

/**
 * Check HuggingFace provider environment variables
 */
function hasHuggingFaceProviderEnvVars(provider: string): boolean {
  if (!["huggingface", "hugging-face", "hf"].includes(provider)) {
    return false;
  }
  return !!(process.env.HUGGINGFACE_API_KEY || process.env.HF_TOKEN);
}

/**
 * Check Mistral provider environment variables
 */
function hasMistralProviderEnvVars(provider: string): boolean {
  if (!["mistral", "mistral-ai", "mistralai"].includes(provider)) {
    return false;
  }
  return !!process.env.MISTRAL_API_KEY;
}

/**
 * Check Ollama provider environment variables
 */
function hasOllamaProviderEnvVars(provider: string): boolean {
  if (!["ollama", "local", "local-ollama"].includes(provider)) {
    return false;
  }
  return true; // Ollama doesn't require environment variables
}

/**
 * Check LiteLLM provider environment variables
 */
function hasLiteLlmProviderEnvVars(provider: string): boolean {
  if (provider !== "litellm") {
    return false;
  }
  return true; // LiteLLM proxy availability will be checked during usage
}

/**
 * Get available provider names
 * @returns Array of available provider names
 */
export function getAvailableProviders(): string[] {
  return [
    "bedrock",
    "vertex",
    "openai",
    "anthropic",
    "azure",
    "google-ai",
    "huggingface",
    "ollama",
    "mistral",
  ];
}

/**
 * Validate provider name
 * @param provider - Provider name to validate
 * @returns True if provider name is valid
 */
export function isValidProvider(provider: string): boolean {
  return getAvailableProviders().includes(provider.toLowerCase());
}

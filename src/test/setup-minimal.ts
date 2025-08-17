/**
 * Minimal test setup for NeuroLink test infrastructure
 * Created to resolve missing dependency blocking all tests
 */

// Mock environment variables for testing
process.env.NODE_ENV = "test";

// Basic test configuration
export const testConfig = {
  timeout: 10000,
  retries: 1,
  environment: "test",
};

// Mock provider configurations for tests
export const mockProviderConfig = {
  openai: { apiKey: "test-key" },
  anthropic: { apiKey: "test-key" },
  google: { apiKey: "test-key" },
  ollama: { baseUrl: "http://localhost:11434" },
};

// Test utility functions
export function createMockProvider(name: string): {
  name: string;
  generate: () => Promise<{ content: string }>;
  getConfig: () => { apiKey?: string; baseUrl?: string };
} {
  return {
    name,
    generate: async (): Promise<{ content: string }> => ({
      content: `Mock response from ${name}`,
    }),
    getConfig: (): { apiKey?: string; baseUrl?: string } =>
      mockProviderConfig[name as keyof typeof mockProviderConfig],
  };
}

// Setup logging for tests
export function setupTestLogging(): void {
  // Suppress logs during testing unless DEBUG=1
  if (!process.env.DEBUG) {
    console.log = (): void => {};
    console.debug = (): void => {};
  }
}

// Initialize test environment
setupTestLogging();

export default {
  testConfig,
  mockProviderConfig,
  createMockProvider,
  setupTestLogging,
};

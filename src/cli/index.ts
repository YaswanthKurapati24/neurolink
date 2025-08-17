#!/usr/bin/env node

/**
 * NeuroLink CLI
 *
 * Professional CLI experience with minimal maintenance overhead.
 * Features: Spinners, colors, batch processing, provider testing, rich help
 */

import yargs from "yargs";
import { hideBin } from "yargs/helpers";
import _ora from "ora";
import chalk from "chalk";

import { addOllamaCommands } from "./commands/ollama.js";
import { addSageMakerCommands } from "./commands/sagemaker.js";
import { CLICommandFactory } from "./factories/commandFactory.js";

// Load environment variables from .env file
try {
  // Try to import and configure dotenv
  const { config } = await import("dotenv");
  config(); // Load .env from current working directory
} catch {
  // dotenv is not available (dev dependency only) - this is fine for production
  // Environment variables should be set externally in production
}

// Utility Functions (Simple, Zero Maintenance)

// Manual pre-validation for unknown flags
const args = hideBin(process.argv);

// Enhanced CLI with Professional UX
const cli = yargs(args)
  .scriptName("neurolink")
  .usage("Usage: $0 <command> [options]")
  .version()
  .help()
  .alias("h", "help")
  .alias("V", "version")
  .strictOptions()
  .strictCommands()
  .demandCommand(1, "")
  .epilogue("For more info: https://github.com/juspay/neurolink")
  .showHelpOnFail(true, "Specify --help for available options")
  .middleware((argv: { noColor?: boolean; [key: string]: unknown }) => {
    // Handle no-color option globally
    if (argv.noColor || process.env.NO_COLOR || !process.stdout.isTTY) {
      process.env.FORCE_COLOR = "0";
    }

    // Handle custom config file
    if (argv.configFile) {
      process.env.NEUROLINK_CONFIG_FILE = argv.configFile as string;
    }

    // Control SDK logging based on debug flag
    if (argv.debug) {
      process.env.NEUROLINK_DEBUG = "true";
    } else {
      // Always set to false when debug is not enabled (including when not provided)
      process.env.NEUROLINK_DEBUG = "false";
    }

    // Keep existing quiet middleware
    if (
      process.env.NEUROLINK_QUIET === "true" &&
      typeof argv.quiet === "undefined"
    ) {
      argv.quiet = true;
    }
  })
  .fail((msg, err, yargsInstance) => {
    const exitProcess = (): void => {
      if (!process.exitCode) {
        process.exit(1);
      }
    };

    if (err) {
      // Error likely from an async command handler (e.g., via handleError)
      // handleError already prints and calls process.exit(1).
      // If we're here, it means handleError's process.exit might not have been caught by the top-level async IIFE.
      // Or, it's a synchronous yargs error during parsing that yargs itself throws.
      const alreadyExitedByHandleError =
        (err as Error & { exitCode?: number })?.exitCode !== undefined;
      // A simple heuristic: if the error message doesn't look like one of our handled generic messages,
      // it might be a direct yargs parsing error.
      const isLikelyYargsInternalError =
        err.message && // Ensure err.message exists
        !err.message.includes("Authentication error") &&
        !err.message.includes("Network error") &&
        !err.message.includes("Authorization error") &&
        !err.message.includes("Permission denied") && // from config export
        !err.message.includes("Invalid or unparseable JSON"); // from config import

      if (!alreadyExitedByHandleError) {
        process.stderr.write(
          chalk.red(
            `CLI Error: ${err.message || msg || "An unexpected error occurred."}\n`,
          ),
        );
        // If it's a yargs internal parsing error, show help.
        if (isLikelyYargsInternalError && msg) {
          yargsInstance.showHelp((h) => {
            process.stderr.write(h + "\n");
            exitProcess();
          });
          return;
        }
        exitProcess();
      }
      return; // Exit was already called or error handled
    }

    // Yargs parsing/validation error (msg is present, err is null)
    if (msg) {
      let processedMsg = `Error: ${msg}\n`;
      if (
        msg.includes("Not enough non-option arguments") ||
        msg.includes("Missing required argument") ||
        msg.includes("Unknown command")
      ) {
        process.stderr.write(chalk.red(processedMsg)); // Print error first
        yargsInstance.showHelp((h) => {
          process.stderr.write("\n" + h + "\n");
          exitProcess();
        });
        return; // Exit happens in callback
      } else if (
        msg.includes("Unknown argument") ||
        msg.includes("Invalid values")
      ) {
        processedMsg = `Error: ${msg}\nUse --help to see available options.\n`;
      }
      process.stderr.write(chalk.red(processedMsg));
    } else {
      // No specific message, but failure occurred (e.g. demandCommand failed silently)
      yargsInstance.showHelp((h) => {
        process.stderr.write(h + "\n");
        exitProcess();
      });
      return; // Exit happens in callback
    }
    exitProcess(); // Default exit
  })

  // Generate Command (Primary) - Using CLICommandFactory
  .command(CLICommandFactory.createGenerateCommand())

  // Stream Text Command - Using CLICommandFactory
  .command(CLICommandFactory.createStreamCommand())

  // Batch Processing Command - Using CLICommandFactory
  .command(CLICommandFactory.createBatchCommand())

  // Provider Command Group - Using CLICommandFactory
  .command(CLICommandFactory.createProviderCommands())

  // Status command alias - Using CLICommandFactory
  .command(CLICommandFactory.createStatusCommand())

  // Models Command Group - Using CLICommandFactory
  .command(CLICommandFactory.createModelsCommands())

  // MCP Command Group - Using CLICommandFactory
  .command(CLICommandFactory.createMCPCommands())

  // Discover Command - Using CLICommandFactory
  .command(CLICommandFactory.createDiscoverCommand())

  // Configuration Command Group - Using CLICommandFactory
  .command(CLICommandFactory.createConfigCommands())

  // Get Best Provider Command - Using CLICommandFactory
  .command(CLICommandFactory.createBestProviderCommand())

  // Completion Command - Using CLICommandFactory
  .command(CLICommandFactory.createCompletionCommand());

// Add Ollama Commands
addOllamaCommands(cli);

// Add SageMaker Commands
addSageMakerCommands(cli);

// Execute CLI
(async (): Promise<void> => {
  try {
    // Parse and execute commands
    await cli.parse();
  } catch (error) {
    // Global error handler - should not reach here due to fail() handler
    process.stderr.write(
      chalk.red(`Unexpected CLI error: ${(error as Error).message}\n`),
    );
    process.exit(1);
  }
})();

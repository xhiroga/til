// run-adapter.js
const path = require('path');
const fs = require('fs');
// Adjust the path based on where this script is run relative to the core package
const { exec2xAdapter, exec3xAdapter } = require('./extensions/playable-ads-adapter/playable-adapter-core-b6c0debc.js');

// --- Argument Parsing (Simple) ---
const args = process.argv.slice(2);
let cocosVersion = null;
let buildPath = null;
let configPath = null;
let mode = 'parallel'; // Default mode

for (let i = 0; i < args.length; i++) {
  if (args[i] === '--version' && args[i + 1]) {
    cocosVersion = args[i + 1];
    i++;
  } else if (args[i] === '--buildPath' && args[i + 1]) {
    buildPath = path.resolve(args[i + 1]); // Resolve to absolute path
    i++;
  } else if (args[i] === '--config' && args[i + 1]) {
    configPath = path.resolve(args[i + 1]); // Resolve to absolute path
    i++;
  } else if (args[i] === '--mode' && args[i + 1]) {
    mode = args[i + 1];
    i++;
  }
}

if (!cocosVersion || !buildPath) {
  console.error('Error: --version and --buildPath are required.');
  console.log('Usage: node run-adapter.js --version <2|3> --buildPath <path/to/build> [--config <path/to/.adapterrc>] [--mode <parallel|serial>]');
  process.exit(1);
}

if (cocosVersion !== '2' && cocosVersion !== '3') {
  console.error('Error: --version must be 2 or 3.');
  process.exit(1);
}

// --- Build Options ---
const options = {
  buildFolderPath: buildPath,
  adapterBuildConfig: null,
};

if (configPath) {
  try {
    const configContent = fs.readFileSync(configPath, 'utf-8');
    options.adapterBuildConfig = JSON.parse(configContent);
    console.log(`Loaded config from: ${configPath}`);
  } catch (err) {
    console.error(`Error reading or parsing config file at ${configPath}:`, err);
    process.exit(1);
  }
} else {
    // If configPath is not provided, expect the core library to find .adapterrc in buildFolderPath
    console.log('Config path not provided, attempting to find .adapterrc in build folder.');
}


// --- Execute Adapter ---
(async () => {
  try {
    console.log(`Running adapter for Cocos v${cocosVersion}...`);
    console.log(`Build Path: ${options.buildFolderPath}`);
    console.log(`Mode: ${mode}`);

    if (cocosVersion === '2') {
      await exec2xAdapter(options, { mode });
    } else {
      await exec3xAdapter(options, { mode });
    }
    console.log('Adapter finished successfully!');
  } catch (error) {
    console.error('Adapter failed:', error);
    process.exit(1);
  }
})();
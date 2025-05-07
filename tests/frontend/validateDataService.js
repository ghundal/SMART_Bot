/**
 * validateDataService.js
 * A validation script for DataService that supports ES modules
 */

// Load necessary modules for Node.js environment
const fs = require('fs');
const path = require('path');

// Setup global fetch mock for testing
global.fetch = function mockFetch() {
  return Promise.resolve({
    ok: true,
    json: () => Promise.resolve({}),
    text: () => Promise.resolve('')
  });
};

// Setup localStorage mock
global.localStorage = {
  _data: {},
  getItem(key) {
    return this._data[key];
  },
  setItem(key, value) {
    this._data[key] = value;
  },
  removeItem(key) {
    delete this._data[key];
  },
  clear() {
    this._data = {};
  }
};

// Set environment variables
global.process = global.process || {};
global.process.env = global.process.env || {};
global.process.env.NEXT_PUBLIC_API_URL = 'http://localhost:9000';

/**
 * Simple validation by parsing the file and checking for expected functions
 */
async function validateDataServiceFile() {
  console.log('🔍 Starting DataService validation tests...');

  // Path to DataService.js
  const dataServicePath = path.resolve(__dirname, '../../src/frontend/services/DataService.js');

  try {
    // Check if the file exists
    if (!fs.existsSync(dataServicePath)) {
      console.error(`❌ DataService.js file not found at ${dataServicePath}`);
      return false;
    }

    // Read the file contents
    const fileContents = fs.readFileSync(dataServicePath, 'utf8');
    console.log('✅ DataService.js file found and read successfully');

    // Validation Results
    const results = {
      passed: 0,
      failed: 0,
      errors: []
    };

    // Test Helper
    function test(name, testFn) {
      try {
        console.log(`\n🧪 Testing: ${name}`);
        testFn();
        console.log(`✅ PASSED: ${name}`);
        results.passed++;
      } catch (error) {
        console.error(`❌ FAILED: ${name}`);
        console.error(`  Error: ${error.message}`);
        results.failed++;
        results.errors.push({ name, error: error.message });
      }
    }

    // 1. Test for expected functions in the file
    test('Expected Functions Presence', () => {
      const expectedFunctions = [
        'GetChats',
        'GetChat',
        'StartChatWithLLM',
        'ContinueChatWithLLM',
        'DeleteChat',
        'SubmitQuery',
        'uuid',
        'getSessionId'
      ];

      let foundFunctions = 0;

      expectedFunctions.forEach(funcName => {
        // Check for function definition or export
        const functionPatterns = [
          new RegExp(`function\\s+${funcName}\\s*\\(`),              // function declaration
          new RegExp(`const\\s+${funcName}\\s*=\\s*\\(?.*\\)\\s*=>`), // arrow function
          new RegExp(`export\\s+const\\s+${funcName}\\s*=`),          // exported const
          new RegExp(`export\\s+function\\s+${funcName}\\s*\\(`),     // exported function
          new RegExp(`export\\s+\\{.*\\b${funcName}\\b.*\\}`),        // named export
          new RegExp(`${funcName}\\s*:\\s*function`)                  // object method
        ];

        const found = functionPatterns.some(pattern => pattern.test(fileContents));

        if (found) {
          foundFunctions++;
          console.log(`  - Found function: ${funcName}`);
        } else {
          console.log(`  - Could not find function: ${funcName}`);
        }
      });

      if (foundFunctions === 0) {
        throw new Error('No expected functions found in DataService module');
      }

      console.log(`  Found ${foundFunctions}/${expectedFunctions.length} expected functions`);

      if (foundFunctions < expectedFunctions.length / 2) {
        throw new Error(`Only found ${foundFunctions}/${expectedFunctions.length} expected functions`);
      }
    });

    // 2. Test for fetch usage
    test('API Communication Pattern', () => {
      const fetchPattern = /fetch\s*\(\s*`\${API_URL}/;
      const foundFetch = fetchPattern.test(fileContents);

      if (!foundFetch) {
        throw new Error('Could not find fetch API calls with API_URL usage');
      }

      console.log('  - Found proper fetch API usage with API_URL');
    });

    // Print summary
    console.log('\n📊 Validation Summary:');
    console.log(`✅ Passed: ${results.passed}`);
    console.log(`❌ Failed: ${results.failed}`);

    if (results.failed > 0) {
      console.log('\n❌ Failed Tests:');
      results.errors.forEach(({ name, error }) => {
        console.log(`  - ${name}: ${error}`);
      });
      return false;
    }

    return true;
  } catch (error) {
    console.error('❌ Error during validation:', error.message);
    return false;
  }
}

// Execute validation if called directly
if (require.main === module) {
  validateDataServiceFile().then(success => {
    console.log('\n🏁 Validation complete!');
    process.exit(success ? 0 : 1);
  }).catch(error => {
    console.error('❌ Unexpected error:', error);
    process.exit(1);
  });
}

module.exports = validateDataServiceFile;

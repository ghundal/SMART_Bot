/**
 * validateReportsService.js
 * A validation script for ReportsService that uses file reading to verify structure
 */

const fs = require('fs');
const path = require('path');

// Set environment variables
global.process = global.process || {};
global.process.env = global.process.env || {};
global.process.env.NEXT_PUBLIC_API_URL = 'http://localhost:9000';

/**
 * Validate the ReportsService.js file
 */
async function validateReportsService() {
  console.log('🔍 Starting ReportsService validation tests...');

  // Path to ReportsService.js
  const reportsServicePath = path.resolve(
    __dirname,
    '../../src/frontend/services/ReportsService.js',
  );

  try {
    // Check if the file exists
    if (!fs.existsSync(reportsServicePath)) {
      console.error(`❌ ReportsService.js file not found at ${reportsServicePath}`);
      return false;
    }

    // Read the file contents
    const fileContents = fs.readFileSync(reportsServicePath, 'utf8');
    console.log('✅ ReportsService.js file found and read successfully');

    // Validation Results
    const results = {
      passed: 0,
      failed: 0,
      errors: [],
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
        'getSystemStats',
        'getUserCount',
        'getQueryCount',
        'getTopDocuments',
        'getQueryActivity',
        'getTopKeywords',
        'getTopPhrases',
        'getUserActivity',
        'getDailyActiveUsers',
      ];

      let foundFunctions = 0;

      expectedFunctions.forEach((funcName) => {
        // Check for function definition or export
        const functionPatterns = [
          new RegExp(`function\\s+${funcName}\\s*\\(`), // function declaration
          new RegExp(`const\\s+${funcName}\\s*=\\s*\\(?.*\\)\\s*=>`), // arrow function
          new RegExp(`export\\s+const\\s+${funcName}\\s*=`), // exported const
          new RegExp(`export\\s+function\\s+${funcName}\\s*\\(`), // exported function
          new RegExp(`export\\s+\\{.*\\b${funcName}\\b.*\\}`), // named export
          new RegExp(`${funcName}\\s*:\\s*function`), // object method
        ];

        const found = functionPatterns.some((pattern) => pattern.test(fileContents));

        if (found) {
          foundFunctions++;
          console.log(`  - Found function: ${funcName}`);
        } else {
          console.log(`  - Could not find function: ${funcName}`);
        }
      });

      if (foundFunctions === 0) {
        throw new Error('No expected functions found in ReportsService module');
      }

      console.log(`  Found ${foundFunctions}/${expectedFunctions.length} expected functions`);

      if (foundFunctions < expectedFunctions.length / 2) {
        throw new Error(
          `Only found ${foundFunctions}/${expectedFunctions.length} expected functions`,
        );
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

    // 3. Test for error handling
    test('Error Handling', () => {
      const errorHandlingPattern =
        /if\s*\(\s*!\s*response\.ok\s*\)\s*\{[\s\S]*?throw\s+new\s+Error/;
      const foundErrorHandling = errorHandlingPattern.test(fileContents);

      if (!foundErrorHandling) {
        throw new Error('Could not find proper error handling for fetch responses');
      }

      console.log('  - Found proper error handling for fetch responses');
    });

    // 4. Test for ReportsService object export
    test('Service Object Export', () => {
      const serviceObjectPattern = /const\s+ReportsService\s*=\s*\{[\s\S]*?\};/;
      const exportDefaultPattern = /export\s+default\s+ReportsService/;

      const foundServiceObject = serviceObjectPattern.test(fileContents);
      const foundExportDefault = exportDefaultPattern.test(fileContents);

      if (!foundServiceObject) {
        throw new Error('Could not find ReportsService object definition');
      }

      if (!foundExportDefault) {
        throw new Error('Could not find export default for ReportsService');
      }

      console.log('  - Found ReportsService object and default export');
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
  validateReportsService()
    .then((success) => {
      console.log('\n🏁 Validation complete!');
      process.exit(success ? 0 : 1);
    })
    .catch((error) => {
      console.error('❌ Unexpected error:', error);
      process.exit(1);
    });
}

module.exports = validateReportsService;

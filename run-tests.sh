#!/bin/bash
# run-all-tests.sh - Run all tests for E115_SMART (unit, integration, validation)

# Set colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Create necessary directories if they don't exist
mkdir -p tests/frontend

# Keep track of all results
PYTHON_RESULT=0
DATASERVICE_UNIT_RESULT=0
DATASERVICE_INTEGRATION_RESULT=0
DATASERVICE_VALIDATION_RESULT=0
REPORTSSERVICE_UNIT_RESULT=0
REPORTSSERVICE_INTEGRATION_RESULT=0
REPORTSSERVICE_VALIDATION_RESULT=0

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}🔍 RUNNING ALL E115_SMART TESTS 🔍${NC}"
echo -e "${BLUE}=======================================${NC}"

# 1. Run Python tests
echo -e "\n${BLUE}🐍 Running Python tests...${NC}"
python -m pytest
PYTHON_RESULT=$?

if [ $PYTHON_RESULT -eq 0 ]; then
  echo -e "${GREEN}✅ Python tests passed!${NC}"
else
  echo -e "${RED}❌ Python tests failed!${NC}"
fi

# 2. Run DataService unit tests
echo -e "\n${BLUE}🧪 Running DataService unit tests...${NC}"
npx jest tests/frontend/DataService.test.js --testEnvironment=node --transform='babel-jest'
DATASERVICE_UNIT_RESULT=$?

if [ $DATASERVICE_UNIT_RESULT -eq 0 ]; then
  echo -e "${GREEN}✅ DataService unit tests passed!${NC}"
else
  echo -e "${RED}❌ DataService unit tests failed!${NC}"
fi

# 3. Run DataService integration tests
echo -e "\n${BLUE}🔄 Running DataService integration tests...${NC}"
npx jest tests/frontend/DataService.integration.test.js --testEnvironment=node --transform='babel-jest'
DATASERVICE_INTEGRATION_RESULT=$?

if [ $DATASERVICE_INTEGRATION_RESULT -eq 0 ]; then
  echo -e "${GREEN}✅ DataService integration tests passed!${NC}"
else
  echo -e "${RED}❌ DataService integration tests failed!${NC}"
fi

# 4. Run DataService validation script
echo -e "\n${BLUE}🔍 Running DataService validation...${NC}"
node tests/frontend/validateDataService.js
DATASERVICE_VALIDATION_RESULT=$?

if [ $DATASERVICE_VALIDATION_RESULT -eq 0 ]; then
  echo -e "${GREEN}✅ DataService validation passed!${NC}"
else
  echo -e "${RED}❌ DataService validation failed!${NC}"
fi

# 5. Run ReportsService unit tests
echo -e "\n${BLUE}🧪 Running ReportsService unit tests...${NC}"
npx jest tests/frontend/ReportsService.test.js --testEnvironment=node --transform='babel-jest'
REPORTSSERVICE_UNIT_RESULT=$?

if [ $REPORTSSERVICE_UNIT_RESULT -eq 0 ]; then
  echo -e "${GREEN}✅ ReportsService unit tests passed!${NC}"
else
  echo -e "${RED}❌ ReportsService unit tests failed!${NC}"
fi

# 6. Run ReportsService integration tests
echo -e "\n${BLUE}🔄 Running ReportsService integration tests...${NC}"
npx jest tests/frontend/ReportsService.integration.test.js --testEnvironment=node --transform='babel-jest'
REPORTSSERVICE_INTEGRATION_RESULT=$?

if [ $REPORTSSERVICE_INTEGRATION_RESULT -eq 0 ]; then
  echo -e "${GREEN}✅ ReportsService integration tests passed!${NC}"
else
  echo -e "${RED}❌ ReportsService integration tests failed!${NC}"
fi

# 7. Run ReportsService validation script
echo -e "\n${BLUE}🔍 Running ReportsService validation...${NC}"
node tests/frontend/validateReportsService.js
REPORTSSERVICE_VALIDATION_RESULT=$?

if [ $REPORTSSERVICE_VALIDATION_RESULT -eq 0 ]; then
  echo -e "${GREEN}✅ ReportsService validation passed!${NC}"
else
  echo -e "${RED}❌ ReportsService validation failed!${NC}"
fi

# Output overall results
echo -e "\n${BLUE}=======================================${NC}"
echo -e "${BLUE}📊 TEST RESULTS SUMMARY 📊${NC}"
echo -e "${BLUE}=======================================${NC}"
echo -e "Python Tests: $([ $PYTHON_RESULT -eq 0 ] && echo "${GREEN}PASSED${NC}" || echo "${RED}FAILED${NC}")"
echo -e "DataService Tests:"
echo -e "  ⮑ Unit: $([ $DATASERVICE_UNIT_RESULT -eq 0 ] && echo "${GREEN}PASSED${NC}" || echo "${RED}FAILED${NC}")"
echo -e "  ⮑ Integration: $([ $DATASERVICE_INTEGRATION_RESULT -eq 0 ] && echo "${GREEN}PASSED${NC}" || echo "${RED}FAILED${NC}")"
echo -e "  ⮑ Validation: $([ $DATASERVICE_VALIDATION_RESULT -eq 0 ] && echo "${GREEN}PASSED${NC}" || echo "${RED}FAILED${NC}")"
echo -e "ReportsService Tests:"
echo -e "  ⮑ Unit: $([ $REPORTSSERVICE_UNIT_RESULT -eq 0 ] && echo "${GREEN}PASSED${NC}" || echo "${RED}FAILED${NC}")"
echo -e "  ⮑ Integration: $([ $REPORTSSERVICE_INTEGRATION_RESULT -eq 0 ] && echo "${GREEN}PASSED${NC}" || echo "${RED}FAILED${NC}")"
echo -e "  ⮑ Validation: $([ $REPORTSSERVICE_VALIDATION_RESULT -eq 0 ] && echo "${GREEN}PASSED${NC}" || echo "${RED}FAILED${NC}")"

# Determine overall pass/fail
if [ $PYTHON_RESULT -eq 0 ] &&
   [ $DATASERVICE_UNIT_RESULT -eq 0 ] &&
   [ $DATASERVICE_INTEGRATION_RESULT -eq 0 ] &&
   [ $DATASERVICE_VALIDATION_RESULT -eq 0 ] &&
   [ $REPORTSSERVICE_UNIT_RESULT -eq 0 ] &&
   [ $REPORTSSERVICE_INTEGRATION_RESULT -eq 0 ] &&
   [ $REPORTSSERVICE_VALIDATION_RESULT -eq 0 ]; then
  echo -e "\n${GREEN}🎉 ALL TESTS PASSED! 🎉${NC}"
  exit 0
else
  echo -e "\n${RED}❌ SOME TESTS FAILED. Please fix the issues before committing.${NC}"
  exit 1
fi

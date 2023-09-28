cmake_minimum_required(VERSION 3.12)

# Check if Python is installed
find_package (Python3 COMPONENTS Interpreter)
find_program(PYTHON_EXECUTABLE python)
if(NOT PYTHON_EXECUTABLE)
    message(FATAL_ERROR "Python is not installed. Please install Python and try again.")
endif()

# Set the project directory
set(PROJECT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/..")

# Create a virtual environment
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -m venv beyond_defer_venv
    WORKING_DIRECTORY ${PROJECT_DIRECTORY}
    RESULT_VARIABLE result
)

# If the result is 0, the the command is failed
if (result)
    message(FATAL_ERROR "Failed to create a virtual environment.")
endif()


# Activate the virtual environment
if(WIN32)
    set(vfolder "beyond_defer_venv\\Scripts\\")
else()
    set(vfolder "$ENV{HOME}/beyond_defer_venv/bin/")
endif()
set(ENV{VIRTUAL_ENV} vfolder)
set (Python3_FIND_VIRTUALENV FIRST)
unset (Python3_EXECUTABLE)
find_package (Python3 COMPONENTS Interpreter Development)

# Install dependencies from requirements.txt
execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt requirements.txt
    COMMAND $ENV{HOME}/beyond_defer_venv/bin/pip install -r requirements.txt
    RESULT_VARIABLE result
)
# print results

message(STATUS "result: ${result}")

if(result)
    message(FATAL_ERROR "Failed to install dependencies from requirements.txt.")
endif()


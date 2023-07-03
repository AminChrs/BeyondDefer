cmake_minimum_required(VERSION 3.12)

# Check if Python is installed
find_program(PYTHON_EXECUTABLE python)
if(NOT PYTHON_EXECUTABLE)
    message(FATAL_ERROR "Python is not installed. Please install Python and try again.")
endif()

# Set the project directory
set(PROJECT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/..")

# Create a virtual environment
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -m venv myvenv
    WORKING_DIRECTORY ${PROJECT_DIRECTORY}
    RESULT_VARIABLE result
)

# If the result is 0, the the command is failed
if (result)
    message(FATAL_ERROR "Failed to create a virtual environment.")
endif()
# Activate the virtual environment
if(WIN32)
    set(vfolder "myenv\\Scripts\\")
else()
    set(vfolder "./myenv/bin/")
endif()
# change directory to vfolder
execute_process()
message(STATUS "result of activation: ${result}")

# Install dependencies from requirements.txt
execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt requirements.txt
    COMMAND pip install -r requirements.txt
    RESULT_VARIABLE result
)
# print results

message(STATUS "result: ${result}")

if(result)
    message(FATAL_ERROR "Failed to install dependencies from requirements.txt.")
endif()

# Deactivate the virtual environment
if(WIN32)
    set(deactivate_command "myenv\\Scripts\\deactivate.bat")
else()
    set(deactivate_command "deactivate")
endif()

execute_process(
    COMMAND ${activate_command} && ${deactivate_command}
    WORKING_DIRECTORY ${PROJECT_DIRECTORY}
    RESULT_VARIABLE result
)
if(result)
    message(FATAL_ERROR "Failed to deactivate the virtual environment.")
endif()
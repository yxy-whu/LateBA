cmake_minimum_required(VERSION 3.7)

if ("${CMAKE_BUILD_TYPE}" MATCHES "Debug")
  set(DEBUG ON)
endif ("${CMAKE_BUILD_TYPE}" MATCHES "Debug")

# To match Makefiles and have just one build type per configured build
# dir, we collapse VS generator configs to a single choice.
# This must be done prior to the project() command and the var
# must be set in the cache.
if ("${CMAKE_GENERATOR}" MATCHES "Visual Studio")
  if (DEBUG)
    set(CMAKE_CONFIGURATION_TYPES "Debug" CACHE STRING "" FORCE)
  else ()
    # Go w/ debug info (i#1392)
    set(CMAKE_CONFIGURATION_TYPES "RelWithDebInfo" CACHE STRING "" FORCE)
  endif ()
  # we want to use the <VAR>_<config> variants of config-dependent properties
  string(TOUPPER "${CMAKE_CONFIGURATION_TYPES}" upper)
  set(location_suffix "_${upper}")
else ("${CMAKE_GENERATOR}" MATCHES "Visual Studio")
  set(location_suffix "")
endif ("${CMAKE_GENERATOR}" MATCHES "Visual Studio")

project(DynamoRIO_tractFunc)

if ("${CMAKE_VERSION}" VERSION_EQUAL "3.9" OR
   "${CMAKE_VERSION}" VERSION_GREATER "3.9")
  # i#1375: We are ok w/ the new policy of SKIP_BUILD_RPATH not affecting install names.
  cmake_policy(SET CMP0068 NEW)
endif ()

set(output_dir "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${output_dir}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${output_dir}")
if ("${CMAKE_GENERATOR}" MATCHES "Visual Studio")
  # we don't support the Debug and Release subdirs
  foreach (config ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER "${config}" config_upper)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config_upper}
      "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${config_upper}
      "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${config_upper}
      "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
  endforeach ()
endif ()

if (BUILD_TESTS)
  # For automated sanity tests we can't have msgboxes or output
  set(SHOW_RESULTS_DEFAULT OFF)
else (BUILD_TESTS)
  set(SHOW_RESULTS_DEFAULT ON)
endif (BUILD_TESTS)

option(SHOW_RESULTS
  "Display client results in pop-up (Windows) or console message (Linux)"
  ${SHOW_RESULTS_DEFAULT})
if (SHOW_RESULTS)
  add_definitions(-DSHOW_RESULTS)
endif (SHOW_RESULTS)

option(SHOW_SYMBOLS "Use symbol lookup in clients that support it" ON)
if (SHOW_SYMBOLS)
  add_definitions(-DSHOW_SYMBOLS)
endif (SHOW_SYMBOLS)
if (NOT DEFINED GENERATE_PDBS)
  # support running tests over ssh where pdb building is problematic
  set(GENERATE_PDBS ON)
endif (NOT DEFINED GENERATE_PDBS)

# For VERBOSE option to recorde instructions in function
option(VERBOSE "Display more information about function that support it" ON)
if(VERBOSE)
    add_definitions(-DVERBOSE)
endif(VERBOSE)

# i#379: We usually want to build the samples with optimizations to improve the
# chances of inlining, but it's nice to be able to turn that off easily.  A
# release build should already have optimizations, so this should only really
# affect debug builds.
option(OPTIMIZE_SAMPLES
  "Build samples with optimizations to increase the chances of clean call inlining (overrides debug flags)"
  ON)
if (WIN32)
  set(OPT_CFLAGS "/O2")
else (WIN32)
  set(OPT_CFLAGS "-O2")
endif (WIN32)

if (DEBUG)
  set(OPT_CFLAGS "${OPT_CFLAGS} -DDEBUG")
endif (DEBUG)

# For C clients that only rely on the DR API and not on any 3rd party
# library routines, we could shrink the size of the client binary
# by disabling libc via "set(DynamoRIO_USE_LIBC OFF)".

if (NOT DEFINED DynamoRIO_DIR)
  set(DynamoRIO_DIR "${PROJECT_SOURCE_DIR}/../cmake" CACHE PATH
    "DynamoRIO installation's cmake directory")
endif (NOT DEFINED DynamoRIO_DIR)

find_package(DynamoRIO 9.0)
if (NOT DynamoRIO_FOUND)
  message(FATAL_ERROR "DynamoRIO package required to build")
endif(NOT DynamoRIO_FOUND)

if (WIN32)
  # disable stack protection: "unresolved external symbol ___security_cookie"
  # disable the warning "unreferenced formal parameter" #4100
  # disable the warning "conditional expression is constant" #4127
  # disable the warning "cast from function pointer to data pointer" #4054
  set(CL_CFLAGS "/GS- /wd4100 /wd4127 /wd4054")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CL_CFLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CL_CFLAGS}")
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif (WIN32)

if (OPTIMIZE_SAMPLES)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OPT_CFLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPT_CFLAGS}")
endif ()

function (add_sample_client name source_file_list extension_list)
  add_library(${name} SHARED ${source_file_list})
  configure_DynamoRIO_client(${name})
  foreach (ext ${extension_list})
    use_DynamoRIO_extension(${name} ${ext})
  endforeach (ext)

  # Provide a hint for how to use the client
  if (NOT DynamoRIO_INTERNAL OR NOT "${CMAKE_GENERATOR}" MATCHES "Ninja")
    DynamoRIO_get_full_path(path ${name} "${location_suffix}")
    add_custom_command(TARGET ${name}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND}
      ARGS -E echo "Usage: pass to drconfig or drrun: -c ${path}"
      VERBATIM)
  endif ()
  get_property(sample_list GLOBAL PROPERTY DynamoRIO_sample_list) # NON-PUBLIC
  set_property(GLOBAL PROPERTY DynamoRIO_sample_list              # NON-PUBLIC
    "${sample_list};${name}")                                     # NON-PUBLIC
  set(srcs ${srcs} ${source_file_list} PARENT_SCOPE)              # NON-PUBLIC
  set(tgts ${tgts} ${name} PARENT_SCOPE)                          # NON-PUBLIC

  # Some samples actually have more tests than just simple sanity # NON-PUBLIC
  # tests. These tests require that results are shown. In order   # NON-PUBLIC
  # to avoid short-circuiting the SHOW_RESULTS option, we build   # NON-PUBLIC
  # an additional sample (which shows results) for testing        # NON-PUBLIC
  # purposes.                                                     # NON-PUBLIC
  # FIXME i#4372 i#4392: Ultimately, when we remove all sanity    # NON-PUBLIC
  # checks and have proper tests for the samples, we will         # NON-PUBLIC
  # remove the SHOW_RESULTS option.                               # NON-PUBLIC
  if (BUILD_TESTS)                                                # NON-PUBLIC
    add_library(${name}_test SHARED ${source_file_list})          # NON-PUBLIC
    configure_DynamoRIO_client(${name}_test)                      # NON-PUBLIC
    foreach (ext ${extension_list})                               # NON-PUBLIC
      use_DynamoRIO_extension(${name}_test ${ext})                # NON-PUBLIC
    endforeach (ext)                                              # NON-PUBLIC
    target_compile_definitions(${name}_test PRIVATE -DSHOW_RESULTS)    # NON-PUBLIC
    set(tgts ${tgts} ${name}_test PARENT_SCOPE)                        # NON-PUBLIC
  endif ()                                                             # NON-PUBLIC

endfunction (add_sample_client)

function (add_sample_standalone name source_file_list)
  add_executable(${name} ${source_file_list})

  configure_DynamoRIO_standalone(${name})

  # Provide a hint for running
  if (NOT DynamoRIO_INTERNAL OR NOT "${CMAKE_GENERATOR}" MATCHES "Ninja")
    if (UNIX)
      set(FIND_MSG "(set LD_LIBRARY_PATH)")
    else (UNIX)
      set(FIND_MSG "(set PATH or copy to same directory)")
    endif (UNIX)
    add_custom_command(TARGET ${name}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND}
      ARGS -E echo "Make sure the loader finds the dynamorio library ${FIND_MSG}"
      VERBATIM)
  endif ()
endfunction (add_sample_standalone)

###########################################################################

# As we'll be calling configure_DynamoRIO_{client,standalone} from within
# a function scope, we must set the global vars ahead of time:
configure_DynamoRIO_global(OFF ON)

# Use ;-separated lists for source files and extensions.

# FIXME i#4372: Add functioning tests for samples that go beyond  # NON-PUBLIC
# sanity checks.                                                  # NON-PUBLIC
if(X86)
    add_sample_client(tract_func "tract_func.c;utils.c" "drmgr;drsyms;drx;drcovlib")
endif(X86)
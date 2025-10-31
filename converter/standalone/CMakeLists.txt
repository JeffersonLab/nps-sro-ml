file(GLOB STANDALONE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/sources/*.cpp)

foreach(executable ${STANDALONE_SOURCES})
    get_filename_component(exec_stem ${executable} NAME_WE) 
    add_executable(${exec_stem}.exe ${executable} )

    target_link_libraries(
        ${exec_stem}.exe
        PRIVATE
        NPSConverter
        ${ROOT_LIBRARIES}
        ${TORCH_LIBRARIES}
    )

    set_target_properties(
        ${exec_stem}.exe 
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY 
        ${CMAKE_BINARY_DIR}
    )

endforeach(executable)


# function: FetchAndExtractDependency
function(FetchAndExtractDependency DEPENDENCY_NAME DEPENDENCY_URL DEPENDENCY_PREFIX HASH_SHA256)
    include(ExternalProject)
    message(STATUS "Downloading and extracting ${DEPENDENCY_NAME} Setting...")

    # add ExternalProject
    ExternalProject_Add( ${DEPENDENCY_NAME}
        PREFIX                      ${DEPENDENCY_PREFIX}
        DOWNLOAD_EXTRACT_TIMESTAMP  true
        URL                         ${DEPENDENCY_URL}
        URL_HASH                    ${HASH_SHA256}
        DOWNLOAD_NO_PROGRESS        1
        CONFIGURE_COMMAND           ""
        BUILD_COMMAND               ""
        INSTALL_COMMAND             ""
    )

    message(STATUS "Download and extraction of ${DEPENDENCY_NAME} setting completed.")
endfunction()
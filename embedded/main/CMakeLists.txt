# gtcrn_int8.bin is required and must be produced by tools/export_int8.py.
# gtcrn_testvec.bin is optional; if present, the boot self-test is compiled in.
set(GTCRN_EMBED "gtcrn_int8.bin")
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/gtcrn_testvec.bin")
    list(APPEND GTCRN_EMBED "gtcrn_testvec.bin")
endif()

idf_component_register(
    SRCS        "app_main.c" "audio_io.c" "wav.c" "usb_audio.c"
    INCLUDE_DIRS "."
    REQUIRES    gtcrn esp_driver_i2s esp_driver_spi esp_timer fatfs sdmmc vfs usb_host_uac esp_driver_uart
    EMBED_FILES ${GTCRN_EMBED})

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/gtcrn_testvec.bin")
    target_compile_definitions(${COMPONENT_LIB} PRIVATE GTCRN_HAVE_TESTVEC=1)
endif()

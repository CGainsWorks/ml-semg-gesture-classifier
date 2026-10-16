/*
 * The model and the fixed input windows, embedded in flash byte for byte.
 *
 * .incbin rather than a generated C array: nothing has to be regenerated when
 * the model changes and what ends up in flash is provably the committed file.
 * TFLite Micro wants the model 16-byte aligned.
 */

    .section .rodata.model, "a"
    .balign 16
    .global g_model
    .global g_model_end
g_model:
    .incbin MODEL_TFLITE_PATH
g_model_end:

    .section .rodata.windows, "a"
    .balign 16
    .global g_windows
    .global g_windows_end
g_windows:
    .incbin WINDOWS_BIN_PATH
g_windows_end:
